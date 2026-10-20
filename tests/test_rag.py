import pytest
from fastapi.testclient import TestClient

from aeroflow.api import create_app
from aeroflow.config import load_config
from aeroflow.engine import Engine
from aeroflow.rag import SopIndex, find_sop_dir
from aeroflow.sources import SimSource
from aeroflow.store import Store

# Hand-written questions an operator might ask -> the SOP file that should answer them.
EVAL_SET = [
    ("what do we do if someone collapses", "person_down.md"),
    ("passenger fell on the floor and is not moving", "person_down.md"),
    ("when should we start CPR and fetch the defibrillator", "person_down.md"),
    ("abandoned suitcase and nobody claims it", "unattended_bag.md"),
    ("how big should the cordon be around a suspicious bag", "unattended_bag.md"),
    ("can we use radios near an unattended item", "unattended_bag.md"),
    ("a person without a pass entered the staff only area", "restricted_area.md"),
    ("how do we challenge someone in a restricted zone", "restricted_area.md"),
    ("security wait is too long should we open another lane", "queue_and_capacity.md"),
    ("redirect passengers to a less busy checkpoint", "queue_and_capacity.md"),
    ("zone is over capacity what now", "queue_and_capacity.md"),
    ("people are pushing and the crowd is too dense", "crowd_density.md"),
    ("keep emergency exits clear when overcrowded", "crowd_density.md"),
    ("someone has been standing in the same spot for a long time", "loitering.md"),
    ("welfare check for a person sleeping in the hall", "loitering.md"),
]


# Written AFTER the synonym map was tuned on EVAL_SET, so this is a fairer estimate of generalisation.
HELD_OUT = [
    ("a woman has fainted near the check-in desks", "person_down.md"),
    ("who do we call for a lost rucksack left on a bench", "unattended_bag.md"),
    ("man wandering behind the staff door", "restricted_area.md"),
    ("the security line is huge, what should we do", "queue_and_capacity.md"),
    ("terminal is packed and people are being squeezed", "crowd_density.md"),
    ("traveller has been idle in one place for hours", "loitering.md"),
    ("elderly man slipped and cannot get up", "person_down.md"),
    ("can we let people through a faster lane when flights are close to departing", "queue_and_capacity.md"),
]


@pytest.fixture(scope="module")
def index():
    return SopIndex("sops")


def test_chunks_have_metadata(index):
    assert len(index.chunks) >= 15
    bag = [c for c in index.chunks if c.source == "unattended_bag.md"]
    assert bag and all("unattended_bag" in c.alert_types for c in bag)


def test_retrieval_quality_on_labelled_questions(index):
    top1 = top3 = 0
    for q, expected in EVAL_SET:
        srcs = [h["source"] for h in index.search(q, k=3)]
        top1 += srcs[:1] == [expected]
        top3 += expected in srcs
    n = len(EVAL_SET)
    assert top1 / n >= 0.8, f"top-1 {top1}/{n}"
    assert top3 / n >= 0.93, f"top-3 {top3}/{n}"


def test_retrieval_on_held_out_questions(index):
    hits = sum(index.search(q, k=3)[0]["source"] == e for q, e in HELD_OUT if index.search(q, k=3))
    assert hits / len(HELD_OUT) >= 0.6, f"held-out top-1 {hits}/{len(HELD_OUT)}"


@pytest.mark.parametrize(
    "atype,expected",
    [
        ("unattended_bag", "unattended_bag.md"),
        ("person_down", "person_down.md"),
        ("restricted_entry", "restricted_area.md"),
        ("open_lane", "queue_and_capacity.md"),
        ("loitering", "loitering.md"),
    ],
)
def test_alert_maps_to_right_sop(index, atype, expected):
    rec = index.for_alert({"type": atype, "summary": f"{atype} in Check-in Hall"}, "Check-in Hall")
    assert rec["sources"][0]["source"] == expected
    assert rec["steps"] and "approved procedures" in rec["advisory"]


def test_ask_is_grounded_and_cites_sources(index, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = index.ask("what do we do if someone collapses")
    assert r["generator"] == "extractive" and r["sources"][0]["source"] == "person_down.md"
    assert "CPR" in r["answer"] or "first-aid" in r["answer"].lower()


def test_unrelated_question_returns_nothing(index, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = index.ask("what is the best pizza topping")
    assert r["sources"] == [] and r["generator"] == "none"


def test_llm_failure_falls_back_to_extractive(index, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "bad")
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert index.ask("unattended bag procedure")["generator"] == "extractive"


def test_engine_attaches_recommendations_to_alerts():
    cfg = load_config("configs/default.yaml")
    d = find_sop_dir(cfg.raw)
    eng = Engine(cfg, SimSource(cfg), Store(":memory:"), sop=SopIndex(d))
    alerts = []
    for _ in range(int(520 / cfg["sim"]["dt"])):
        alerts += eng.step()
    by_type = {a["type"]: a for a in alerts}
    assert by_type["unattended_bag"]["recommended"]["sources"][0]["source"] == "unattended_bag.md"
    assert by_type["person_down"]["recommended"]["sources"][0]["source"] == "person_down.md"
    stored = {e["type"]: e for e in eng.store.events()}
    assert stored["restricted_entry"]["payload"]["recommended"]["steps"]


def test_api_sop_endpoints():
    cfg = load_config("configs/default.yaml")
    eng = Engine(cfg, SimSource(cfg), Store(":memory:"), sop=SopIndex("sops"))
    c = TestClient(create_app(eng, run_engine=False))
    assert c.get("/api/sop/ask", params={"q": "unattended bag cordon"}).json()["sources"]
    assert c.get("/api/sop/search", params={"q": "person fell", "k": 2}).json()
    eng.sop = None
    assert c.get("/api/sop/ask", params={"q": "anything"}).status_code == 503
