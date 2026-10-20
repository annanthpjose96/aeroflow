from aeroflow.nlsearch import answer, parse_rules
from aeroflow.store import Store


def store_with_data():
    s = Store(":memory:")
    for i, c in enumerate([3, 9, 5]):
        s.add_snapshot(1000 + i * 60, "security_queue", c, c * 6.0, c / 72)
        s.add_snapshot(1000 + i * 60, "gate_5", 1, None, 0.01)
    s.add_event(
        {
            "ts": 1100,
            "type": "unattended_bag",
            "severity": "critical",
            "zone": "check_in",
            "summary": "bag found",
            "pos": [1, 2],
            "payload": {},
        }
    )
    return s


def test_parse_rules(cfg):
    q = parse_rules("When was the Security Queue wait longest in the last 30 minutes?", cfg.zones)
    assert (q.zone_id, q.metric, q.agg, q.since_s, q.kind) == ("security_queue", "wait", "max", 1800, "metric")


def test_parse_alert_question(cfg):
    q = parse_rules("any unattended bag in check-in hall?", cfg.zones)
    assert q.kind == "alerts" and q.alert_type == "unattended_bag" and q.zone_id == "check_in"


def test_answer_max(cfg, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = answer("when was the security queue longest", store_with_data(), cfg.zones)
    assert r["parser"] == "rules" and "9.0" in r["answer"]


def test_answer_alerts(cfg, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = answer("any unattended bag today", store_with_data(), cfg.zones)
    assert "1 matching alert" in r["answer"]


def test_empty_store(cfg, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert "No data" in answer("queue length", Store(":memory:"), cfg.zones)["answer"]


def test_bad_gemini_key_falls_back(cfg, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "invalid")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    r = answer("queue length now", store_with_data(), cfg.zones)
    assert r["parser"] == "rules"


def test_count_question_is_not_mistaken_for_alerts(cfg):
    # regression: "many " contains the substring "any "
    q = parse_rules("how many people are in the security queue", cfg.zones)
    assert q.kind == "metric" and q.metric == "count" and q.zone_id == "security_queue"
    assert parse_rules("is there a countdown timer", cfg.zones).alert_type is None
