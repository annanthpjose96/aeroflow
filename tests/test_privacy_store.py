import numpy as np

from aeroflow.privacy import blur_heads
from aeroflow.store import Store


def test_head_region_is_blurred_body_untouched():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    out = blur_heads(frame, [(50, 20, 110, 180)])
    head_before, head_after = frame[20:55, 50:110].std(), out[20:55, 50:110].std()
    assert head_after < head_before * 0.6
    assert np.array_equal(frame[100:180, 50:110], out[100:180, 50:110])
    assert not np.shares_memory(frame, out)


def test_retention_purge():
    s = Store(":memory:")
    s.add_snapshot(100, "z", 1, None, 0.1)
    s.add_snapshot(900, "z", 2, None, 0.1)
    s.add_event({"ts": 100, "type": "x", "severity": "info", "zone": "z", "summary": "", "payload": {}})
    assert s.purge(500) == 2
    assert [r["ts"] for r in s.series()] == [900]
