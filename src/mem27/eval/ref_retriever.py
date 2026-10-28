"""Incremental BM25 reference retriever (V, harness validation only; G's retrievers are the systems).

Implements the §0 Retriever protocol plus apply_update() so stream_eval() and the
freshness metrics can be tested end to end on CPU.
  version_filter="none"     rank everything visible at as_of_time
  version_filter="explicit" also hide notes whose valid_to <= as_of_time (what a
                            system learns from explicit supersessions only)
  recency_boost=b           score *= (1 + b * recency in [0,1]) -- a cheap freshness prior
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict

from mem27.eval.common import n_tokens, truncate_tokens
from mem27.vschema import EvidenceBundle, Note, RetrievalRequest

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = set("a an the and or of to in on at for with by from as is are was were be been i me my we our you your "
           "he she it its they them their this that what which who whom when where why how do does did now".split())


def tok(text: str) -> list[str]:
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOP]


class RefBM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75, version_filter: str = "none", recency_boost: float = 0.0):
        self.k1, self.b = k1, b
        self.version_filter = version_filter
        self.recency_boost = recency_boost
        self.notes: dict[str, Note] = {}
        self.tf: dict[str, Counter] = {}
        self.dl: dict[str, int] = {}
        self.post: dict[str, set] = defaultdict(set)
        self.len_sum = 0
        self.t_min = self.t_max = None

    def build(self, notes, summaries=None) -> None:
        for n in notes:
            self._add(n)

    def _add(self, n: Note) -> None:
        c = Counter(tok(n.text))
        self.notes[n.note_id] = n
        self.tf[n.note_id] = c
        self.dl[n.note_id] = sum(c.values())
        self.len_sum += self.dl[n.note_id]
        for w in c:
            self.post[w].add(n.note_id)
        self.t_min = min(self.t_min or n.ts, n.ts)
        self.t_max = max(self.t_max or n.ts, n.ts)

    def insert(self, note: Note) -> dict:
        t0 = time.perf_counter()
        self._add(note)
        return {"bm25_ms": (time.perf_counter() - t0) * 1000, "embed_ms": 0.0, "ann_ms": 0.0,
                "summary_ms": 0.0, "summary_tokens": 0, "links_ms": 0.0}

    def apply_update(self, up: dict) -> dict:
        from mem27.data.updates import to_note
        note = to_note(up["note"])
        tr = self.insert(note)
        if up["op"] in ("supersede", "correct") and up.get("explicit") and up.get("supersedes_note_id"):
            old = self.notes.get(up["supersedes_note_id"])
            if old is not None and old.valid_to is None:
                old.valid_to = up["ts"]
        return tr

    def retrieve(self, req: RetrievalRequest) -> EvidenceBundle:
        t0 = time.perf_counter()
        q = tok(req.query)
        N = len(self.notes)
        avg = self.len_sum / max(1, N)
        asof = req.as_of_time
        scores: dict[str, float] = defaultdict(float)
        for w in set(q):
            docs = self.post.get(w, ())
            if not docs:
                continue
            idf = math.log(1 + (N - len(docs) + 0.5) / (len(docs) + 0.5))
            for d in docs:
                n = self.notes[d]
                if asof and n.ts > asof:
                    continue
                if self.version_filter == "explicit" and n.valid_to and (not asof or n.valid_to <= asof):
                    continue
                f = self.tf[d][w]
                dl = self.dl[d]
                scores[d] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avg))
        if self.recency_boost and scores:
            from datetime import datetime
            lo = datetime.fromisoformat(self.t_min).timestamp()
            hi = datetime.fromisoformat(asof or self.t_max).timestamp()
            for d in scores:
                r = (datetime.fromisoformat(self.notes[d].ts).timestamp() - lo) / max(1.0, hi - lo)
                scores[d] *= 1 + self.recency_boost * max(0.0, min(1.0, r))
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        t_rank = (time.perf_counter() - t0) * 1000
        ids, passages, sc, total = [], [], [], 0
        for d, s in ranked:
            if len(ids) >= req.max_results:
                break
            text = self.notes[d].text
            k = n_tokens(text)
            if total + k > req.budget_tokens:
                if req.budget_tokens - total >= 32:
                    text = truncate_tokens(text, req.budget_tokens - total)
                    ids.append(d), passages.append(text), sc.append(s)
                    total += n_tokens(text)
                break
            ids.append(d), passages.append(text), sc.append(s)
            total += k
        stale = [bool(self.notes[d].valid_to and (not asof or self.notes[d].valid_to <= asof)) for d in ids]
        lat = {"total": (time.perf_counter() - t0) * 1000, "stages": {"bm25": t_rank}}
        return EvidenceBundle(ids, passages, sc, total, lat, stale,
                              {"ranked_ids": [d for d, _ in ranked[:200]], "version_filter": self.version_filter})
