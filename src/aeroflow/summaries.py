"""Template-based, human-readable incident summaries (no external services)."""

from __future__ import annotations

import time


def _clock(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def summarize(alert: dict, zone_name: str | None = None) -> str:
    t = alert["type"]
    where = zone_name or alert.get("zone") or "the terminal"
    pos = alert.get("pos")
    at = f" at floor position ({pos[0]:.1f} m, {pos[1]:.1f} m)" if pos else ""
    when = _clock(alert["ts"])
    p = alert.get("payload", {})
    if t == "unattended_bag":
        return (
            f"[{when}] Unattended bag{at} in {where}: stationary for {p.get('seconds', 0):.0f} s "
            f"with no person within {p.get('radius', 0):.1f} m. Recommended: dispatch security to inspect."
        )
    if t == "person_down":
        return f"[{when}] Possible fall{at} in {where}: person has been lying down for {p.get('seconds', 0):.0f} s. Recommended: send first aid."
    if t == "restricted_entry":
        return f"[{when}] Person detected inside restricted area {where}{at}. Recommended: dispatch security."
    if t == "capacity_exceeded":
        return f"[{when}] {where} over capacity: {p.get('count')} people (limit {p.get('capacity')}). Recommended: redirect passengers."
    if t == "long_wait":
        return f"[{when}] Estimated wait at {where} is {p.get('est_wait_s', 0) / 60:.1f} min (limit {p.get('limit_s', 0) / 60:.1f} min)."
    if t == "open_lane":
        return (
            f"[{when}] Forecast: wait at {where} may reach {p.get('pred_wait_s', 0) / 60:.1f} min within "
            f"{p.get('horizon_s', 0) / 60:.0f} min. Recommended: open an additional lane."
        )
    if t == "crowd_density":
        return f"[{when}] High crowd density in {where}: {p.get('density', 0):.2f} people/m² (limit {p.get('limit')})."
    if t == "loitering":
        return f"[{when}] Person has stayed {p.get('seconds', 0):.0f} s in {where}{at}."
    return f"[{when}] {t} in {where}."
