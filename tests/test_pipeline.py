"""End-to-end tests on the simulator: identity consistency and scripted incidents."""

from collections import Counter

from aeroflow.config import load_config
from aeroflow.engine import Engine
from aeroflow.sources import SimSource
from aeroflow.store import Store


def run(cfg, seconds):
    src = SimSource(cfg)
    eng = Engine(cfg, src, Store(":memory:"))
    alerts, per_gt = [], {}
    for _ in range(int(seconds / src.dt)):
        alerts += eng.step()
        for gt, gid, _ in eng.last_assignments:
            per_gt.setdefault(gt, Counter())[gid] += 1
    return eng, alerts, per_gt


def test_cross_camera_identity_is_consistent(cfg):
    _, _, per_gt = run(cfg, 300)
    total = sum(sum(c.values()) for c in per_gt.values())
    kept = sum(max(c.values()) for c in per_gt.values())
    assert kept / total > 0.85, "most observations should keep a single global id"


def test_one_alert_per_scripted_incident_and_no_false_positives():
    cfg = load_config("configs/default.yaml")
    _, alerts, _ = run(cfg, 520)
    types = Counter(a["type"] for a in alerts)
    assert types["unattended_bag"] == 1
    assert types["restricted_entry"] == 1
    assert types["person_down"] == 1


def test_no_incidents_when_nothing_scripted():
    cfg = load_config(
        "configs/default.yaml", {"sim": {"abandoned_bag_at_s": 1e9, "intruder_at_s": 1e9, "fall_at_s": 1e9}}
    )
    _, alerts, _ = run(cfg, 400)
    bad = {"unattended_bag", "restricted_entry", "person_down"}
    assert not [a for a in alerts if a["type"] in bad]


def test_queue_wait_estimate_is_sane(cfg):
    eng, _, _ = run(cfg, 420)
    q = eng.analytics.zone_stats["security_queue"]
    # service time is 6 s/person => ~10/min throughput; wait ~ count * 6 s
    assert q["count"] > 0
    assert 0.3 * q["count"] * 6 < q["est_wait_s"] < 3 * q["count"] * 6


def test_events_persisted(cfg):
    eng, alerts, _ = run(cfg, 520)
    assert len(eng.store.events()) == len(alerts)
    assert eng.store.series("security_queue")


def test_bag_alert_is_latched_and_not_duplicated():
    # regression: seed 99 used to re-raise the same bag alert ~350 s later because a noisy
    # first position sample made normal jitter look like the bag had "moved".
    cfg = load_config("configs/default.yaml", {"sim": {"seed": 99}})
    _, alerts, _ = run(cfg, 800)
    assert sum(a["type"] == "unattended_bag" for a in alerts) == 1


def test_process_accepts_external_detections(cfg):
    """`Engine.process` is the integration point for DeepStream / other detectors."""
    src = SimSource(cfg)
    eng = Engine(cfg, src, Store(":memory:"))
    for _ in range(30):
        ts, dets = src.read()
        eng.process(ts, dets)
    assert eng.state()["tracks"]
