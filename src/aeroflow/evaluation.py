"""Evaluate identity consistency of the fusion pipeline against simulator ground truth."""

from __future__ import annotations

from collections import Counter, defaultdict

from .config import Config
from .engine import Engine
from .sources import SimSource
from .store import Store


def evaluate(cfg: Config, seconds: float = 600.0) -> dict:
    src = SimSource(cfg)
    eng = Engine(cfg, src, Store(":memory:"))
    steps = int(seconds / src.dt)
    gt_to_gids: dict[int, Counter] = defaultdict(Counter)
    gid_to_gts: dict[int, Counter] = defaultdict(Counter)
    alerts: Counter = Counter()
    first_alert: dict[str, float] = {}
    t_start = None
    for _ in range(steps):
        new = eng.step()
        t_start = t_start or eng.ts
        for al in new:
            alerts[al["type"]] += 1
            first_alert.setdefault(al["type"], round(al["ts"] - t_start, 1))
        for gt, gid, _cls in eng.last_assignments:
            if gt is not None:
                gt_to_gids[gt][gid] += 1
                gid_to_gts[gid][gt] += 1

    def purity(d):
        tot = sum(sum(c.values()) for c in d.values())
        return sum(max(c.values()) for c in d.values()) / tot if tot else 0.0

    return {
        "sim_seconds": seconds,
        "ground_truth_objects": len(gt_to_gids),
        "global_ids_created": len(gid_to_gts),
        "id_consistency": round(purity(gt_to_gids), 4),  # does one real object keep one ID?
        "id_purity": round(purity(gid_to_gts), 4),  # does one ID stay one real object?
        "avg_ids_per_object": round(sum(len(c) for c in gt_to_gids.values()) / max(len(gt_to_gids), 1), 3),
        "alerts": dict(alerts),
        "first_alert_after_s": first_alert,
        "pipeline_fps": round(eng.fps, 1),
    }
