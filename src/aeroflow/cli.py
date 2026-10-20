"""Command line interface: `aeroflow run | eval | search | purge`."""

from __future__ import annotations

import argparse
import json
import time

from .config import load_config


def main(argv=None) -> None:
    p = argparse.ArgumentParser("aeroflow", description="Multi-camera passenger flow & queue analytics")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="start pipeline + API + dashboard")
    r.add_argument("--config", default="configs/default.yaml")
    r.add_argument("--source", choices=["sim", "rtsp"], default="sim")
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=8000)
    e = sub.add_parser("eval", help="evaluate ID consistency + alerts on the simulator")
    e.add_argument("--config", default="configs/default.yaml")
    e.add_argument("--seconds", type=float, default=600)
    s = sub.add_parser("search", help="ask a question about stored data")
    s.add_argument("question")
    s.add_argument("--config", default="configs/default.yaml")
    o = sub.add_parser("sop", help="ask a procedure question (RAG over SOPs)")
    o.add_argument("question")
    o.add_argument("--config", default="configs/default.yaml")
    g = sub.add_parser("purge", help="enforce data retention now")
    g.add_argument("--config", default="configs/default.yaml")
    a = p.parse_args(argv)
    cfg = load_config(a.config)

    if a.cmd == "run":
        import uvicorn

        from .api import create_app
        from .engine import build_engine
        from .events import make_bus

        eng = build_engine(cfg, a.source, bus=make_bus(cfg["events"]))
        uvicorn.run(
            create_app(eng, interval_s=cfg["sim"]["dt"] / cfg["sim"]["speed"] if a.source == "sim" else 0.0),
            host=a.host,
            port=a.port,
        )
    elif a.cmd == "eval":
        from .evaluation import evaluate

        print(json.dumps(evaluate(cfg, a.seconds), indent=2))
    elif a.cmd == "search":
        from .nlsearch import answer
        from .store import Store

        print(json.dumps(answer(a.question, Store(cfg["storage"]["db_path"]), cfg.zones), indent=2, default=str))
    elif a.cmd == "sop":
        from .rag import SopIndex, find_sop_dir

        d = find_sop_dir(cfg.raw)
        if d is None:
            raise SystemExit("SOP folder not found (set rag.sop_dir)")
        print(json.dumps(SopIndex(d, cfg["rag"]["backend"], cfg["rag"]["top_k"]).ask(a.question), indent=2))
    elif a.cmd == "purge":
        from .store import Store

        cutoff = time.time() - cfg["storage"]["retention_days"] * 86400
        print(f"deleted {Store(cfg['storage']['db_path']).purge(cutoff)} rows")


if __name__ == "__main__":
    main()
