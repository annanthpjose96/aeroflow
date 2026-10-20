from fastapi.testclient import TestClient

from aeroflow.api import create_app
from aeroflow.engine import Engine
from aeroflow.sources import SimSource
from aeroflow.store import Store


def make(cfg, steps=200):
    eng = Engine(cfg, SimSource(cfg), Store(":memory:"))
    for _ in range(steps):
        eng.step()
    return TestClient(create_app(eng, run_engine=False))


def test_endpoints(cfg):
    c = make(cfg)
    assert c.get("/health").json()["status"] == "ok"
    assert "AeroFlow" in c.get("/").text
    layout = c.get("/api/layout").json()
    assert {z["id"] for z in layout["zones"]} == {"check_in", "security_queue", "gate_5", "restricted"}
    st = c.get("/api/state").json()
    assert st["tracks"] and st["zones"]
    assert c.get("/api/heatmap").json()["grid"]
    assert c.get("/api/zones/security_queue/history").json()
    assert "answer" in c.get("/api/search", params={"q": "queue length now"}).json()
    assert c.get("/api/search", params={"q": "x"}).status_code == 422
    assert "aeroflow_zone_count" in c.get("/metrics").text


def test_websocket_streams_state(cfg):
    c = make(cfg, steps=20)
    with c.websocket_connect("/ws") as ws:
        assert "zones" in ws.receive_json()


def test_dashboard_served_regardless_of_system_default_encoding(cfg, monkeypatch):
    # regression: on Windows the default text encoding is cp1252, which cannot decode the
    # non-ASCII characters in dashboard.html unless the file is read explicitly as UTF-8.
    import pathlib

    orig = pathlib.Path.read_text

    def cp1252_default(self, encoding=None, errors=None):
        return orig(self, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(pathlib.Path, "read_text", cp1252_default)
    assert "AeroFlow" in make(cfg, steps=5).get("/").text
