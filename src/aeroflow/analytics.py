"""Zone analytics: occupancy, dwell, queue wait, heatmap, forecasting and alert rules."""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import Config
from .datatypes import GlobalTrack
from .forecast import Holt
from .geometry import point_in_polygon, polygon_area
from .summaries import summarize

SOFT_ALERTS = {"open_lane", "long_wait", "capacity_exceeded", "crowd_density", "loitering"}


class Analytics:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.a = cfg["analytics"]
        self.zones = cfg.zones
        self.zone_by_id = {z.id: z for z in cfg.zones}
        self.areas = {z.id: polygon_area(z.polygon) for z in cfg.zones}
        w, h, c = cfg["floor"]["width"], cfg["floor"]["height"], self.a["heatmap_cell_m"]
        self.heat = np.zeros((int(np.ceil(h / c)), int(np.ceil(w / c))))
        self.inside: dict[str, dict[int, list[float]]] = {z.id: {} for z in cfg.zones}
        self.exits: dict[str, deque] = {z.id: deque() for z in cfg.zones}
        self.forecasters = {z.id: Holt() for z in cfg.zones if z.type == "queue"}
        self._last_sample = -1e18
        self.bags: dict[int, dict] = {}
        self.fall_since: dict[int, float] = {}
        self.active_alerts: dict[tuple, dict] = {}
        self.zone_stats: dict[str, dict] = {}
        self.forecast: dict[str, dict] = {}
        self.t0: float | None = None
        self._last_raised: dict[tuple, float] = {}

    def update(self, ts: float, dt: float, tracks: list[GlobalTrack]) -> list[dict]:
        a = self.a
        if self.t0 is None:
            self.t0 = ts
        persons = [t for t in tracks if t.cls == "person"]
        bags = [t for t in tracks if t.cls == "bag"]
        seen: set = set()
        new: list[dict] = []

        def raise_(key, type_, severity, zone=None, pos=None, **payload):
            if key in self.active_alerts:
                seen.add(key)
                return
            if type_ in SOFT_ALERTS and ts - self._last_raised.get(key, -1e18) < a["soft_alert_cooldown_s"]:
                return  # flap suppression for threshold-style alerts
            seen.add(key)
            self._last_raised[key] = ts
            al = {
                "key": key,
                "type": type_,
                "severity": severity,
                "zone": zone,
                "ts": ts,
                "pos": None if pos is None else [float(pos[0]), float(pos[1])],
                "payload": payload,
            }
            zname = self.zone_by_id[zone].name if zone in self.zone_by_id else None
            al["summary"] = summarize(al, zname)
            self.active_alerts[key] = al
            new.append(al)

        def lim(key, value):  # hysteresis: once raised, only clear when 15% below the limit
            return value * (0.85 if key in self.active_alerts else 1.0)

        c = a["heatmap_cell_m"]
        for p in persons:
            ix, iy = int(p.pos[0] // c), int(p.pos[1] // c)
            if 0 <= iy < self.heat.shape[0] and 0 <= ix < self.heat.shape[1]:
                self.heat[iy, ix] += dt

        sample_now = ts - self._last_sample >= a["forecast_sample_s"]
        for z in self.zones:
            present = {p.gid: p for p in persons if point_in_polygon(p.pos, z.polygon)}
            ins = self.inside[z.id]
            for gid in present:
                if gid in ins:
                    ins[gid][1] = ts
                else:
                    ins[gid] = [ts, ts]
            for gid in list(ins):
                if gid not in present and ts - ins[gid][1] >= a["exit_grace_s"]:
                    entered, last = ins.pop(gid)
                    self.exits[z.id].append((ts, last - entered))
            win = a["throughput_window_s"]
            while self.exits[z.id] and self.exits[z.id][0][0] < ts - win:
                self.exits[z.id].popleft()
            elapsed = max(min(win, ts - self.t0), 60.0)
            n_exits = len(self.exits[z.id])
            throughput = n_exits / elapsed  # people per second
            count = len(present)
            dwell_now = [ts - ins[g][0] for g in present]
            stat = {
                "id": z.id,
                "name": z.name,
                "type": z.type,
                "count": count,
                "density": count / self.areas[z.id] if self.areas[z.id] else 0.0,
                "avg_dwell_s": float(np.mean(dwell_now)) if dwell_now else 0.0,
                "throughput_per_min": throughput * 60,
                "est_wait_s": None,
                "capacity": z.capacity,
            }
            if z.type == "queue":
                if count == 0:
                    stat["est_wait_s"] = 0.0
                elif n_exits >= a["min_exits_for_wait"]:
                    stat["est_wait_s"] = count / throughput
                else:
                    stat["est_wait_s"] = max(dwell_now)
                fc = self.forecasters[z.id]
                if sample_now:
                    fc.update(count)
                steps = int(a["forecast_horizon_s"] / a["forecast_sample_s"])
                pred = fc.predict(steps)
                if pred is not None:
                    cap = z.capacity * 3 if z.capacity else 1e9
                    pred = min(pred, cap)
                    pw = pred / throughput if n_exits >= a["min_exits_for_wait"] else None
                    self.forecast[z.id] = {"horizon_s": a["forecast_horizon_s"], "pred_count": pred, "pred_wait_s": pw}
                    if (
                        z.max_wait_s
                        and pw is not None
                        and pw > lim(("open_lane", z.id), z.max_wait_s)
                        and (stat["est_wait_s"] or 0) <= z.max_wait_s
                    ):
                        raise_(
                            ("open_lane", z.id),
                            "open_lane",
                            "info",
                            z.id,
                            pred_wait_s=pw,
                            horizon_s=a["forecast_horizon_s"],
                        )
            self.zone_stats[z.id] = stat

            if z.capacity and count > lim(("capacity", z.id), z.capacity):
                raise_(("capacity", z.id), "capacity_exceeded", "warning", z.id, count=count, capacity=z.capacity)
            if z.max_density and stat["density"] > z.max_density:
                raise_(
                    ("density", z.id), "crowd_density", "warning", z.id, density=stat["density"], limit=z.max_density
                )
            if z.type == "queue" and z.max_wait_s and (stat["est_wait_s"] or 0) > lim(("wait", z.id), z.max_wait_s):
                raise_(
                    ("wait", z.id), "long_wait", "warning", z.id, est_wait_s=stat["est_wait_s"], limit_s=z.max_wait_s
                )
            if z.type == "restricted":
                for gid, p in present.items():
                    raise_(("restricted", z.id, gid), "restricted_entry", "critical", z.id, p.pos)
            if z.max_dwell_s:
                for gid, p in present.items():
                    d = ts - ins[gid][0]
                    if d > z.max_dwell_s:
                        raise_(("loiter", z.id, gid), "loitering", "info", z.id, p.pos, seconds=d)
        if sample_now:
            self._last_sample = ts

        # unattended bags: the bag is stationary AND its owner (nearest person while it was being
        # carried) has been farther than `owner_radius_m` -- or gone -- for N seconds.
        live_bags = set()
        by_gid = {p.gid: p for p in persons}
        for b in bags:
            live_bags.add(b.gid)
            st = self.bags.get(b.gid)
            if st is not None:
                st["last_seen"] = ts
            moved = st is None or np.linalg.norm(b.pos - st["anchor"]) > a["stationary_radius_m"]
            if st is None:
                st = self.bags[b.gid] = {
                    "anchor": b.pos.copy(),
                    "anchor_ts": ts,
                    "alone_since": None,
                    "owner": None,
                    "last_seen": ts,
                }
            if not moved:  # smooth the anchor so early noisy samples don't cause false "moved" events
                st["anchor"] = 0.9 * st["anchor"] + 0.1 * b.pos
            if moved:
                st["anchor"], st["anchor_ts"] = b.pos.copy(), ts
                near = [(np.linalg.norm(p.pos - b.pos), p.gid) for p in persons]
                near = [x for x in near if x[0] <= 1.5]
                st["owner"] = min(near)[1] if near else None
            carriers = [(np.linalg.norm(p.pos - b.pos), p.gid) for p in persons]
            carriers = [x for x in carriers if x[0] <= a["carry_radius_m"]]
            owner = by_gid.get(st["owner"])
            st["carry_since"] = (st.get("carry_since") or ts) if carriers else None
            # someone physically with the bag (sustained, to ignore passers-by) -> (re)bind ownership
            if carriers and (moved or ts - st["carry_since"] >= a["carry_min_s"]):
                st["owner"] = min(carriers)[1]
                attended = True
            elif owner is not None:
                attended = np.linalg.norm(owner.pos - b.pos) <= a["owner_radius_m"]
            elif st["owner"] is None:  # owner unknown: fall back to "any person nearby"
                attended = any(np.linalg.norm(p.pos - b.pos) <= a["owner_radius_m"] for p in persons)
            else:
                attended = False  # owner identity left the scene
            st["alone_since"] = None if attended else (st["alone_since"] or ts)
            if ("bag", b.gid) in self.active_alerts and not moved:
                seen.add(("bag", b.gid))  # latched: stays active until the bag is moved or disappears
            elif st["alone_since"] is not None:
                dur = min(ts - st["alone_since"], ts - st["anchor_ts"])
                if dur >= a["unattended_after_s"]:
                    zid = next((z.id for z in self.zones if point_in_polygon(b.pos, z.polygon)), None)
                    raise_(
                        ("bag", b.gid),
                        "unattended_bag",
                        "critical",
                        zid,
                        b.pos,
                        seconds=dur,
                        radius=a["owner_radius_m"],
                    )
        for gid in list(self.bags):
            if gid in live_bags:
                continue
            if ts - self.bags[gid]["last_seen"] <= a["bag_dropout_s"]:
                if ("bag", gid) in self.active_alerts:  # brief dropout: keep the alert latched
                    seen.add(("bag", gid))
            else:
                del self.bags[gid]

        # fall detection from sustained "lying" bounding-box aspect
        down = set()
        for p in persons:
            if p.fallen:
                down.add(p.gid)
                since = self.fall_since.setdefault(p.gid, ts)
                if ts - since >= a["fall_after_s"]:
                    zid = next((z.id for z in self.zones if point_in_polygon(p.pos, z.polygon)), None)
                    raise_(("fall", p.gid), "person_down", "critical", zid, p.pos, seconds=ts - since)
        for gid in [g for g in self.fall_since if g not in down]:
            del self.fall_since[gid]

        for key in [k for k in self.active_alerts if k not in seen]:
            del self.active_alerts[key]
        return new

    def heatmap(self) -> list[list[float]]:
        return np.round(self.heat, 2).tolist()
