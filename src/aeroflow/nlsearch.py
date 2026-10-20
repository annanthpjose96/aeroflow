"""Natural-language questions over stored analytics.

Default: a deterministic rule-based parser (no network, no keys).
Optional: Google Gemini *free tier* (set GEMINI_API_KEY) translates the question into the same
structured query; any failure silently falls back to the rule parser. Only the question text and
zone names are sent -- never video, frames or tracking data.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass

from .config import ZoneCfg
from .store import Store

ALERT_WORDS = {
    "unattended_bag": ["bag", "unattended", "abandoned", "luggage"],
    "person_down": ["fall", "fell", "collapse", "down"],
    "restricted_entry": ["restricted", "intruder", "staff only", "trespass"],
    "capacity_exceeded": ["capacity", "overcrowd"],
    "long_wait": ["long wait"],
    "open_lane": ["open lane", "extra lane"],
    "loitering": ["loiter"],
    "crowd_density": ["density"],
}


@dataclass
class Query:
    kind: str = "metric"  # metric | alerts
    metric: str = "count"  # count | wait | density
    agg: str = "latest"  # latest | max | min | avg
    zone_id: str | None = None
    since_s: float | None = None
    alert_type: str | None = None


def _match_zone(q: str, zones: list[ZoneCfg]) -> str | None:
    best, best_score = None, 0
    ql = q.lower()
    for z in zones:
        toks = [t for t in re.split(r"[\s_\-]+", f"{z.name} {z.id}".lower()) if t]
        score = sum(1 for t in set(toks) if re.search(rf"\b{re.escape(t)}\b", ql))
        if score > best_score:
            best, best_score = z.id, score
    return best


def parse_rules(question: str, zones: list[ZoneCfg]) -> Query:
    q = question.lower()
    out = Query(zone_id=_match_zone(q, zones))
    for typ, words in ALERT_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}", q) for w in words):
            out.alert_type = typ
            break
    if out.alert_type or re.search(r"\b(alerts?|alarms?|incidents?|happened|warnings?)\b", q):
        out.kind = "alerts"
    if "wait" in q:
        out.metric = "wait"
    elif "density" in q or "crowd" in q:
        out.metric = "density"
    if any(w in q for w in ("longest", "highest", "most", "peak", "max", "busiest", "worst", "longer")):
        out.agg = "max"
    elif any(w in q for w in ("lowest", "least", "min", "quietest", "shortest")):
        out.agg = "min"
    elif any(w in q for w in ("average", "avg", "mean", "typical")):
        out.agg = "avg"
    m = re.search(r"last\s+(\d+)\s*(second|sec|minute|min|hour|hr)", q)
    if m:
        n, u = int(m.group(1)), m.group(2)
        out.since_s = n * (3600 if u.startswith(("hour", "hr")) else 60 if u.startswith("min") else 1)
    elif "last hour" in q:
        out.since_s = 3600
    elif "last minute" in q:
        out.since_s = 60
    return out


def parse_gemini(question: str, zones: list[ZoneCfg], api_key: str, model: str) -> Query:
    prompt = (
        "Convert the airport analytics question into JSON with keys: kind (metric|alerts), "
        "metric (count|wait|density), agg (latest|max|min|avg), zone_id (one of "
        f"{[z.id for z in zones]} or null), since_s (seconds lookback or null), alert_type "
        f"(one of {list(ALERT_WORDS)} or null). Reply with JSON only.\nQuestion: {question}"
    )
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
        }
    ).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        text = json.load(r)["candidates"][0]["content"]["parts"][0]["text"]
    d = json.loads(text)
    valid = {k: d.get(k) for k in Query.__dataclass_fields__ if k in d}
    return Query(**valid)


def _fmt_time(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def answer(question: str, store: Store, zones: list[ZoneCfg]) -> dict:
    names = {z.id: z.name for z in zones}
    key = os.environ.get("GEMINI_API_KEY")
    parser = "rules"
    try:
        if not key:
            raise RuntimeError
        query = parse_gemini(question, zones, key, os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
        parser = "gemini"
    except Exception:  # noqa: BLE001 - any LLM/network failure must fall back to rules
        query = parse_rules(question, zones)

    ref = store.latest_ts()
    if ref is None:
        return {"answer": "No data collected yet.", "query": asdict(query), "parser": parser, "data": []}
    since = ref - query.since_s if query.since_s else None

    if query.kind == "alerts":
        evs = store.events(since=since, type_=query.alert_type, zone_id=query.zone_id, limit=20)
        if not evs:
            return {"answer": "No matching alerts found.", "query": asdict(query), "parser": parser, "data": []}
        lines = [e["summary"] for e in evs[:5]]
        return {
            "answer": f"{len(evs)} matching alert(s). Latest: " + " | ".join(lines),
            "query": asdict(query),
            "parser": parser,
            "data": evs,
        }

    col = {"count": "count", "wait": "est_wait_s", "density": "density"}[query.metric]
    rows = [r for r in store.series(query.zone_id, since) if r[col] is not None]
    if not rows:
        return {"answer": "No data for that question.", "query": asdict(query), "parser": parser, "data": []}
    where = names.get(query.zone_id, "all zones")
    unit = {"count": "people", "wait": "s", "density": "people/m²"}[query.metric]
    if query.agg in ("max", "min"):
        r = (max if query.agg == "max" else min)(rows, key=lambda r: r[col])
        txt = f"{query.agg.upper()} {query.metric} in {where}: {r[col]:.1f} {unit} at {_fmt_time(r['ts'])} ({r['zone_id']})."
        data = [r]
    elif query.agg == "avg":
        v = sum(r[col] for r in rows) / len(rows)
        txt, data = f"Average {query.metric} in {where}: {v:.1f} {unit} over {len(rows)} samples.", []
    else:
        last_ts = rows[-1]["ts"]
        data = [r for r in rows if r["ts"] == last_ts]
        txt = (
            "Latest "
            + query.metric
            + ": "
            + "; ".join(f"{names.get(r['zone_id'], r['zone_id'])} = {r[col]:.1f} {unit}" for r in data)
            + "."
        )
    return {"answer": txt, "query": asdict(query), "parser": parser, "data": data}
