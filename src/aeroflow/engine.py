"""Pipeline engine: detections -> local tracks -> floor projection -> global IDs -> analytics."""

from __future__ import annotations

import threading
import time

from . import metrics as M
from .analytics import Analytics
from .config import Config
from .datatypes import Observation
from .events import EventBus, NullBus
from .fusion import GlobalFusion
from .geometry import Homography
from .store import Store
from .tracking import CameraTracker


class Engine:
    def __init__(self, cfg: Config, source, store: Store, bus: EventBus | None = None, sop=None):
        self.cfg, self.source, self.store, self.sop = cfg, source, store, sop
        self.bus = bus or NullBus()
        self.prefix = cfg["events"]["topic_prefix"]
        tcfg, fcfg = cfg["tracking"], cfg["fusion"]
        self.homs = {c.id: Homography(c.image_points, c.floor_points) for c in cfg.cameras}
        self.trackers = {c.id: CameraTracker(c.id, **tcfg) for c in cfg.cameras}
        self.fusion = GlobalFusion(**fcfg)
        self.analytics = Analytics(cfg)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.ts = 0.0
        self._last_ts: float | None = None
        self._last_snapshot = -1e18
        self.fps = 0.0
        self.last_assignments: list[tuple[int | None, int, str]] = []  # (gt_id, gid, cls) for evaluation
        self.recent_alerts: list[dict] = []
        self._active: list = []

    def step(self) -> list[dict]:
        """Pull one batch of detections from the configured source and process it."""
        ts, dets = self.source.read()
        return self.process(ts, dets)

    def process(self, ts: float, dets: list) -> list[dict]:
        """Run one pipeline tick on externally supplied detections (e.g. from a DeepStream probe)."""
        t_wall = time.perf_counter()
        with self.lock:
            dt = 0.0 if self._last_ts is None else max(ts - self._last_ts, 0.0)
            self._last_ts, self.ts = ts, ts
            by_cam: dict[str, list] = {c: [] for c in self.trackers}
            for d in dets:
                by_cam[d.camera_id].append(d)
            obs: list[Observation] = []
            for cid, trk in self.trackers.items():
                seen, dead = trk.update(by_cam[cid])
                self.fusion.forget(cid, dead)
                for t in seen:
                    pos = self.homs[cid].to_floor(*t.foot)
                    obs.append(Observation(cid, t.local_id, t.cls, pos, t.embedding, t.aspect, ts, t.gt_id))
            fused = self.fusion.update(ts, obs)
            self.last_assignments = [(o.gt_id, gid, o.cls) for o, gid in fused]
            self._active = self.fusion.active(ts)
            new_alerts = self.analytics.update(ts, dt, self._active)

            for al in new_alerts:
                if self.sop is not None:
                    zone = self.analytics.zone_by_id.get(al.get("zone"))
                    al["recommended"] = self.sop.for_alert(al, zone.name if zone else None)
                self.store.add_event(al)
                self.bus.publish(f"{self.prefix}/alerts", {k: v for k, v in al.items() if k != "key"})
                M.ALERTS.labels(al["type"]).inc()
                self.recent_alerts = ([al] + self.recent_alerts)[:50]
            interval = self.cfg["analytics"]["snapshot_interval_s"]
            if ts - self._last_snapshot >= interval:
                self._last_snapshot = ts
                for zid, s in self.analytics.zone_stats.items():
                    self.store.add_snapshot(ts, zid, s["count"], s["est_wait_s"], s["density"])
                    M.ZONE_COUNT.labels(zid).set(s["count"])
                    if s["est_wait_s"] is not None:
                        M.ZONE_WAIT.labels(zid).set(s["est_wait_s"])
                    fc = self.analytics.forecast.get(zid)
                    if fc and fc["pred_wait_s"] is not None:
                        M.ZONE_PRED_WAIT.labels(zid).set(fc["pred_wait_s"])
                self.bus.publish(f"{self.prefix}/zones", {"ts": ts, "zones": list(self.analytics.zone_stats.values())})
            for cls in ("person", "bag"):
                M.TRACKS.labels(cls).set(sum(1 for t in self._active if t.cls == cls))
        dur = time.perf_counter() - t_wall
        M.STEP_LATENCY.observe(dur)
        self.fps = 0.9 * self.fps + 0.1 * (1 / dur) if self.fps else 1 / max(dur, 1e-6)
        M.FPS.set(self.fps)
        return new_alerts

    def state(self) -> dict:
        with self.lock:
            return {
                "ts": self.ts,
                "fps": round(self.fps, 1),
                "tracks": [
                    {
                        "gid": t.gid,
                        "cls": t.cls,
                        "x": float(t.pos[0]),
                        "y": float(t.pos[1]),
                        "fallen": bool(t.fallen),
                        "cams": sorted(t.cameras),
                    }
                    for t in self._active
                ],
                "zones": list(self.analytics.zone_stats.values()),
                "alerts": [{k: v for k, v in a.items() if k != "key"} for a in self.analytics.active_alerts.values()],
                "forecast": self.analytics.forecast,
            }

    def layout(self) -> dict:
        return {
            "floor": self.cfg["floor"],
            "zones": [
                {"id": z.id, "name": z.name, "type": z.type, "polygon": z.polygon, "capacity": z.capacity}
                for z in self.cfg.zones
            ],
            "cameras": [{"id": c.id, "floor_polygon": c.floor_points} for c in self.cfg.cameras],
        }

    def start(self, interval_s: float = 0.0) -> None:
        def loop():
            while not self.stop_event.is_set():
                t = time.perf_counter()
                try:
                    self.step()
                except Exception:  # keep serving; the error is visible in logs
                    import logging

                    logging.getLogger("aeroflow").exception("pipeline step failed")
                    time.sleep(0.5)
                self.stop_event.wait(max(0.0, interval_s - (time.perf_counter() - t)))

        self.thread = threading.Thread(target=loop, daemon=True, name="aeroflow-engine")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.source.close()
        self.bus.close()


def build_engine(
    cfg: Config, source_kind: str = "sim", store: Store | None = None, bus: EventBus | None = None
) -> Engine:
    from .rag import SopIndex, find_sop_dir
    from .sources import RtspSource, SimSource

    store = store or Store(cfg["storage"]["db_path"])
    src = SimSource(cfg) if source_kind == "sim" else RtspSource(cfg)
    sop = None
    if cfg["rag"]["enabled"]:
        d = find_sop_dir(cfg.raw)
        if d:
            sop = SopIndex(d, cfg["rag"]["backend"], cfg["rag"]["top_k"])
    return Engine(cfg, src, store, bus, sop)
