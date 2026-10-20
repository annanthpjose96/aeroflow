"""FastAPI app: REST + WebSocket + Prometheus + dashboard."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__
from .engine import Engine
from .nlsearch import answer

STATIC = Path(__file__).parent / "static"


def create_app(engine: Engine, run_engine: bool = True, interval_s: float = 0.0) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        if run_engine:
            engine.start(interval_s)
        yield
        if run_engine:
            engine.stop()

    app = FastAPI(title="AeroFlow", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__, "fps": round(engine.fps, 1)}

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return (STATIC / "dashboard.html").read_text(encoding="utf-8")

    @app.get("/api/layout")
    def layout():
        return engine.layout()

    @app.get("/api/state")
    def state():
        return engine.state()

    @app.get("/api/zones/{zone_id}/history")
    def history(zone_id: str, minutes: float | None = None):
        ref = engine.store.latest_ts()
        since = None if (minutes is None or ref is None) else ref - minutes * 60
        return engine.store.series(zone_id, since)

    @app.get("/api/alerts")
    def alerts(limit: int = Query(50, le=500), type: str | None = None, zone: str | None = None):
        return engine.store.events(type_=type, zone_id=zone, limit=limit)

    @app.get("/api/heatmap")
    def heatmap():
        with engine.lock:
            return {"cell_m": engine.cfg["analytics"]["heatmap_cell_m"], "grid": engine.analytics.heatmap()}

    @app.get("/api/forecast")
    def forecast():
        with engine.lock:
            return engine.analytics.forecast

    @app.get("/api/search")
    def search(q: str = Query(..., min_length=2, max_length=300)):
        return answer(q, engine.store, engine.cfg.zones)

    @app.get("/api/sop/ask")
    def sop_ask(q: str = Query(..., min_length=3, max_length=300)):
        if engine.sop is None:
            raise HTTPException(503, "SOP retrieval is disabled or no SOP folder was found")
        return engine.sop.ask(q)

    @app.get("/api/sop/search")
    def sop_search(q: str = Query(..., min_length=3, max_length=300), k: int = Query(3, ge=1, le=10)):
        if engine.sop is None:
            raise HTTPException(503, "SOP retrieval is disabled or no SOP folder was found")
        return engine.sop.search(q, k=k)

    @app.get("/metrics")
    def prom():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        try:
            while True:
                await sock.send_json(engine.state())
                await asyncio.sleep(0.4)
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app
