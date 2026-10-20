"""Retrieval-augmented guidance from airport SOP documents.

Pipeline:  SOP markdown -> chunk by section -> index -> retrieve -> (optional) grounded answer

* Alerts: when an alert is raised, the most relevant SOP sections are retrieved and attached as
  `recommended` (steps + cited sources). This is decision support: nothing is ever executed
  automatically.
* Questions: `ask()` answers "what do we do if ...?" strictly from retrieved passages.

Backends (config `rag.backend`):
  bm25 (default)  pure-Python lexical ranking + alert-type metadata boost; no downloads, no keys.
  hybrid          BM25 + local sentence-transformers embeddings (pip install aeroflow[rag]);
                  downloads a small model on first use.
Generation is optional: with GEMINI_API_KEY (free tier) the answer is phrased by the model from the
retrieved passages only; otherwise the best passage is returned verbatim (extractive).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ADVISORY = "Decision support only - follow your airport's approved procedures and supervisor instructions."
SECTION_ORDER = [
    "immediate",
    "first response",
    "challenge",
    "crowd control",
    "passenger flow",
    "cordon",
    "follow",
    "escalation",
]
STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from", "how", "i", "if", "in", "is",
    "it", "of", "on", "or", "our", "should", "that", "the", "their", "then", "there", "this", "to", "was", "we",
    "what", "when", "where", "which", "who", "will", "with", "you", "your", "can",
})  # fmt: skip

# Airport-domain synonyms folded onto the vocabulary used in SOPs (applied to documents and queries).
SYNONYMS = {
    "fell": "fall",
    "fallen": "fall",
    "collapsed": "collapse",
    "faint": "collapse",
    "fainted": "collapse",
    "suitcase": "bag",
    "luggage": "bag",
    "baggage": "bag",
    "backpack": "bag",
    "item": "bag",
    "package": "bag",
    "abandoned": "unattended",
    "unclaimed": "unattended",
    "left": "unattended",
    "intruder": "restricted",
    "trespasser": "restricted",
    "unauthorised": "restricted",
    "unauthorized": "restricted",
    "line": "queue",
    "lane": "queue",
    "counter": "queue",
    "checkpoint": "queue",
    "crowded": "crowd",
    "overcrowded": "crowd",
    "overcrowding": "crowd",
    "packed": "crowd",
    "medic": "medical",
    "ambulance": "medical",
    "paramedic": "medical",
    "hurt": "injury",
    "injured": "injury",
    "wander": "loiter",
    "lingering": "loiter",
    "loitering": "loiter",
    "idle": "loiter",
}


def stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s", "e"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def tokenize(text: str) -> list[str]:
    words = [SYNONYMS.get(w, w) for w in re.findall(r"[a-z0-9]+", text.lower())]
    return [stem(w) for w in words if w not in STOP and len(w) > 1]


@dataclass
class Chunk:
    source: str
    title: str
    section: str
    text: str
    alert_types: list[str] = field(default_factory=list)

    @property
    def steps(self) -> list[str]:
        items = [
            re.sub(r"^(\d+[.)]|[-*])\s+", "", ln.strip())
            for ln in self.text.splitlines()
            if re.match(r"^\s*(\d+[.)]|[-*])\s+", ln)
        ]
        return items or [s.strip() for s in re.split(r"(?<=[.!?])\s+", self.text) if s.strip()][:3]


def load_chunks(sop_dir: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(sop_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta: dict = {}
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.DOTALL)
        body = raw
        if m:
            meta, body = yaml.safe_load(m.group(1)) or {}, m.group(2)
        title = meta.get("title", path.stem)
        types = list(meta.get("alert_types", []))
        for part in re.split(r"^## +", body, flags=re.MULTILINE)[1:]:
            head, _, text = part.partition("\n")
            if text.strip():
                chunks.append(Chunk(path.name, title, head.strip(), text.strip(), types))
    return chunks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.n = len(docs)
        self.avg = sum(len(d) for d in docs) / max(self.n, 1)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for i, d in enumerate(self.docs):
            s, dl = 0.0, len(d)
            for t in set(query):
                f = self.tf[i].get(t, 0)
                if f:
                    s += (
                        self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
                    )
            out.append(s)
        return out


class SopIndex:
    def __init__(self, sop_dir: str | Path, backend: str = "bm25", top_k: int = 3):
        self.chunks = load_chunks(sop_dir)
        if not self.chunks:
            raise ValueError(f"no SOP sections found in {sop_dir}")
        self.top_k, self.backend = top_k, backend
        # title and section name are repeated so they weigh more than body text
        self.bm25 = BM25([tokenize(f"{c.title} {c.title} {c.section} {c.text}") for c in self.chunks])
        self._emb = None
        if backend == "hybrid":
            from sentence_transformers import SentenceTransformer  # pip install aeroflow[rag]

            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._emb = self._model.encode(
                [f"{c.title}. {c.section}. {c.text}" for c in self.chunks], normalize_embeddings=True
            )

    def _scores(self, query: str) -> list[float]:
        s = self.bm25.scores(tokenize(query))
        if self._emb is not None:
            sims = self._emb @ self._model.encode([query], normalize_embeddings=True)[0]
            top = max(s) or 1.0
            s = [a / top + 0.5 * float(b) for a, b in zip(s, sims)]
        return s

    def search(self, query: str, k: int | None = None, alert_type: str | None = None) -> list[dict]:
        scores = self._scores(query)
        top = max(scores) or 1.0
        if alert_type:  # metadata boost: SOPs declared for this alert type outrank lexical matches
            scores = [s + (top if alert_type in c.alert_types else 0.0) for s, c in zip(scores, self.chunks)]
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)[: k or self.top_k]
        return [self._hit(self.chunks[i], scores[i]) for i in ranked if scores[i] > 0]

    @staticmethod
    def _hit(c: Chunk, score: float) -> dict:
        return {
            "source": c.source,
            "title": c.title,
            "section": c.section,
            "text": c.text,
            "steps": c.steps,
            "score": round(float(score), 3),
        }

    @staticmethod
    def _order(hits: list[dict]) -> list[dict]:
        def rank(h):
            sec = h["section"].lower()
            return next((i for i, key in enumerate(SECTION_ORDER) if key in sec), len(SECTION_ORDER))

        return sorted(hits, key=rank)

    def for_alert(self, alert: dict, zone_name: str | None = None) -> dict | None:
        """Recommended steps for an alert, with cited sources."""
        query = f"{alert['type'].replace('_', ' ')} {zone_name or ''} {alert.get('summary', '')}"
        hits = self.search(query, alert_type=alert["type"])
        if not hits:
            return None

        hits = self._order(hits)
        return {
            "steps": [s for h in hits for s in h["steps"]][:8],
            "sources": [{"source": h["source"], "title": h["title"], "section": h["section"]} for h in hits],
            "advisory": ADVISORY,
        }

    def ask(self, question: str, use_llm: bool = True) -> dict:
        """Answer a procedure question strictly from retrieved SOP passages."""
        hits = self.search(question, k=self.top_k)
        if not hits:
            return {
                "answer": "No relevant procedure found in the SOPs.",
                "sources": [],
                "generator": "none",
                "advisory": ADVISORY,
            }
        sources = [{"source": h["source"], "title": h["title"], "section": h["section"]} for h in hits]
        key = os.environ.get("GEMINI_API_KEY")
        if use_llm and key:
            try:
                return {
                    "answer": _gemini_grounded(question, hits, key),
                    "sources": sources,
                    "generator": "gemini",
                    "advisory": ADVISORY,
                }
            except Exception:  # noqa: BLE001 - any LLM/network failure falls back to extractive
                logging.getLogger("aeroflow.rag").warning("LLM answer failed; using extractive answer")
        # extractive answer: sections of the best-matching document, in operational order
        doc = [h for h in hits if h["source"] == hits[0]["source"]]
        steps = [st for h in self._order(doc) for st in h["steps"]][:6]
        text = f"{hits[0]['title']}: " + " ".join(f"{i}. {st}" for i, st in enumerate(steps, 1))
        return {
            "answer": text,
            "title": hits[0]["title"],
            "steps": steps,
            "sources": sources,
            "generator": "extractive",
            "advisory": ADVISORY,
        }


def _gemini_grounded(question: str, hits: list[dict], key: str) -> str:
    passages = "\n\n".join(f"[{h['source']} / {h['section']}]\n{h['text']}" for h in hits)
    prompt = (
        "You are an airport operations assistant. Answer ONLY using the passages below. "
        "Cite sources as [file / section]. If the passages do not contain the answer, say you cannot "
        f"find it in the SOPs.\n\nPassages:\n{passages}\n\nQuestion: {question}"
    )
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0}}).encode()
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)["candidates"][0]["content"]["parts"][0]["text"]


def find_sop_dir(cfg_raw: dict) -> Path | None:
    """Resolve rag.sop_dir against cwd, then the config file's directory and its parent."""
    rel = Path(cfg_raw["rag"]["sop_dir"])
    cands = [rel] if rel.is_absolute() else [Path.cwd() / rel]
    cp = cfg_raw.get("_config_path")
    if cp and not rel.is_absolute():
        cands += [Path(cp).resolve().parent / rel, Path(cp).resolve().parent.parent / rel]
    return next((c for c in cands if c.is_dir()), None)
