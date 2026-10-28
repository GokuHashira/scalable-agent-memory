"""Incremental dense and hybrid reference retrievers for the streaming harness (V, evidence only; not systems).

Same protocol and version handling as RefBM25 (mem27/eval/ref_retriever.py):
  RefDense   exact cosine (bge-base-en-v1.5, normalized) over every note visible at as_of_time
  RefHybrid  RRF (k = 60, depth 100) over RefBM25 and RefDense rankings
  version_filter="explicit" hides notes closed by an explicit supersession; recency_boost=b multiplies the
  final score by (1 + b * recency in [0, 1]) (for RRF the fused score).
Embeddings come from an EmbCache, so stream runs do not re-embed; `insert` looks the new note up in the cache
(prefill with `EmbCache.add(texts)` before streaming) and records the lookup time as `embed_ms`.
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np

from mem27.eval.common import n_tokens, truncate_tokens
from mem27.eval.embed import QUERY_PREFIX, EmbCache
from mem27.eval.ref_retriever import RefBM25
from mem27.vschema import EvidenceBundle, Note, RetrievalRequest

RRF_K, RRF_DEPTH = 60, 100


def _ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def pack(notes: dict[str, Note], ranked: list[tuple[str, float]], req: RetrievalRequest):
    ids, passages, sc, total = [], [], [], 0
    for d, s in ranked:
        if len(ids) >= req.max_results:
            break
        text = notes[d].text
        k = n_tokens(text)
        if total + k > req.budget_tokens:
            if req.budget_tokens - total >= 32:
                text = truncate_tokens(text, req.budget_tokens - total)
                ids.append(d), passages.append(text), sc.append(s)
                total += n_tokens(text)
            break
        ids.append(d), passages.append(text), sc.append(s)
        total += k
    return ids, passages, sc, total


class _Versioned:
    version_filter = "none"
    recency_boost = 0.0

    def _visible(self, n: Note, asof: str | None) -> bool:
        if asof and n.ts > asof:
            return False
        if self.version_filter == "explicit" and n.valid_to and (not asof or n.valid_to <= asof):
            return False
        return True

    def _close_explicit(self, up: dict) -> None:
        if up["op"] in ("supersede", "correct") and up.get("explicit") and up.get("supersedes_note_id"):
            old = self.notes.get(up["supersedes_note_id"])
            if old is not None and old.valid_to is None:
                old.valid_to = up["ts"]

    def _recency(self, scores: dict[str, float], asof: str | None) -> None:
        if not (self.recency_boost and scores):
            return
        lo, hi = _ts(self.t_min), _ts(asof or self.t_max)
        for d in scores:
            r = (_ts(self.notes[d].ts) - lo) / max(1.0, hi - lo)
            scores[d] *= 1 + self.recency_boost * max(0.0, min(1.0, r))


class RefDense(_Versioned):
    """Vectorized: embeddings, timestamps and valid_to live in numpy arrays; visibility is a mask."""

    def __init__(self, emb: EmbCache, version_filter: str = "none", recency_boost: float = 0.0):
        self.emb = emb
        self.version_filter, self.recency_boost = version_filter, recency_boost
        self.notes: dict[str, Note] = {}
        self.ids: list[str] = []
        self.row: dict[str, int] = {}
        self._vecs: list[np.ndarray] = []
        self._ts: list[float] = []
        self.valid_to = np.zeros(0)
        self.mat = np.zeros((0, 768), np.float32)
        self.ts = np.zeros(0)
        self._n_mat = 0
        self.t_min = self.t_max = None
        self._qcache: dict[str, np.ndarray] = {}

    def prefetch_queries(self, queries: list[str]) -> None:
        todo = sorted({q for q in queries if q not in self._qcache})
        if todo:
            for q, v in zip(todo, self.emb.encode([QUERY_PREFIX + q for q in todo])):
                self._qcache[q] = v.astype(np.float32)

    def _add(self, n: Note) -> float:
        t0 = time.perf_counter()
        v = self.emb.get([n.text])[0].astype(np.float32)
        ms = (time.perf_counter() - t0) * 1000
        self.row[n.note_id] = len(self.ids)
        self.notes[n.note_id] = n
        self.ids.append(n.note_id)
        self._vecs.append(v)
        self._ts.append(_ts(n.ts))
        self.valid_to = np.append(self.valid_to, _ts(n.valid_to) if n.valid_to else np.inf)
        self.t_min = min(self.t_min or n.ts, n.ts)
        self.t_max = max(self.t_max or n.ts, n.ts)
        return ms

    def _sync(self) -> None:
        if self._n_mat < len(self.ids):
            self.mat = np.concatenate([self.mat, np.stack(self._vecs[self._n_mat:])]) if self._n_mat else np.stack(self._vecs)
            self.ts = np.asarray(self._ts)
            self._n_mat = len(self.ids)

    def build(self, notes, summaries=None) -> None:
        for n in notes:
            self._add(n)

    def insert(self, note: Note) -> dict:
        ms = self._add(note)
        return {"bm25_ms": 0.0, "embed_ms": ms, "ann_ms": 0.0, "summary_ms": 0.0, "summary_tokens": 0, "links_ms": 0.0}

    def close(self, note_id: str, ts: str) -> None:
        old = self.notes.get(note_id)
        if old is not None and old.valid_to is None:
            old.valid_to = ts
        if note_id in self.row and not np.isfinite(self.valid_to[self.row[note_id]]):
            self.valid_to[self.row[note_id]] = _ts(ts)

    def apply_update(self, up: dict) -> dict:
        from mem27.data.updates import to_note
        tr = self.insert(to_note(up["note"]))
        if up["op"] in ("supersede", "correct") and up.get("explicit") and up.get("supersedes_note_id"):
            self.close(up["supersedes_note_id"], up["ts"])
        return tr

    def ranked(self, query: str, asof: str | None, top: int, recency: bool = True) -> list[tuple[str, float]]:
        self._sync()
        if query not in self._qcache:
            self.prefetch_queries([query])
        s = self.mat @ self._qcache[query]
        mask = np.ones(len(s), bool)
        if asof:
            a = _ts(asof)
            mask &= self.ts <= a
            if self.version_filter == "explicit":
                mask &= ~(self.valid_to <= a)
        elif self.version_filter == "explicit":
            mask &= ~np.isfinite(self.valid_to)
        if recency and self.recency_boost:
            lo, hi = _ts(self.t_min), _ts(asof or self.t_max)
            r = np.clip((self.ts - lo) / max(1.0, hi - lo), 0.0, 1.0)
            s = s * (1 + self.recency_boost * r)
        idx = np.flatnonzero(mask)
        if len(idx) > top:
            idx = idx[np.argpartition(-s[idx], top)[:top]]
        order = sorted(idx.tolist(), key=lambda i: (-float(s[i]), self.ids[i]))
        return [(self.ids[i], float(s[i])) for i in order]

    def retrieve(self, req: RetrievalRequest) -> EvidenceBundle:
        t0 = time.perf_counter()
        ranked = self.ranked(req.query, req.as_of_time, 200)
        t_rank = (time.perf_counter() - t0) * 1000
        ids, passages, scs, total = pack(self.notes, ranked, req)
        asof = req.as_of_time
        stale = [bool(self.notes[d].valid_to and (not asof or self.notes[d].valid_to <= asof)) for d in ids]
        return EvidenceBundle(ids, passages, scs, total, {"total": (time.perf_counter() - t0) * 1000,
                                                          "stages": {"dense": t_rank}}, stale,
                              {"ranked_ids": [d for d, _ in ranked[:200]], "version_filter": self.version_filter})


class RefHybrid(_Versioned):
    def __init__(self, emb: EmbCache, version_filter: str = "none", recency_boost: float = 0.0):
        self.version_filter, self.recency_boost = version_filter, recency_boost
        # components rank without the recency prior; the prior is applied once, to the fused score
        self.bm = RefBM25(version_filter=version_filter)
        self.de = RefDense(emb, version_filter=version_filter)
        self.notes = self.de.notes

    @property
    def t_min(self):
        return self.de.t_min

    @property
    def t_max(self):
        return self.de.t_max

    def build(self, notes, summaries=None) -> None:
        notes = list(notes)
        self.bm.build(notes)
        self.de.build(notes)

    def insert(self, note: Note) -> dict:
        a, b = self.bm.insert(note), self.de.insert(note)
        return {**a, "embed_ms": b["embed_ms"]}

    def apply_update(self, up: dict) -> dict:
        from mem27.data.updates import to_note
        note = to_note(up["note"])
        tr = self.insert(note)
        if up["op"] in ("supersede", "correct") and up.get("explicit") and up.get("supersedes_note_id"):
            old = self.bm.notes.get(up["supersedes_note_id"])
            if old is not None and old.valid_to is None:
                old.valid_to = up["ts"]
            self.de.close(up["supersedes_note_id"], up["ts"])  # Note objects are shared; also updates the array
        return tr

    def prefetch_queries(self, queries: list[str]) -> None:
        self.de.prefetch_queries(queries)

    def retrieve(self, req: RetrievalRequest) -> EvidenceBundle:
        t0 = time.perf_counter()
        big = RetrievalRequest(query=req.query, budget_tokens=10 ** 9, max_results=RRF_DEPTH,
                               as_of_time=req.as_of_time, plan=req.plan, question_id=req.question_id)
        rb = self.bm.retrieve(big).trace["ranked_ids"][:RRF_DEPTH]
        t_bm = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        rd = [d for d, _ in self.de.ranked(req.query, req.as_of_time, RRF_DEPTH, recency=False)]
        t_de = (time.perf_counter() - t1) * 1000
        fused: dict[str, float] = {}
        for r in (rb, rd):
            for i, d in enumerate(r):
                fused[d] = fused.get(d, 0.0) + 1.0 / (RRF_K + i + 1)
        self._recency(fused, req.as_of_time)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
        ids, passages, scs, total = pack(self.notes, ranked, req)
        asof = req.as_of_time
        stale = [bool(self.notes[d].valid_to and (not asof or self.notes[d].valid_to <= asof)) for d in ids]
        return EvidenceBundle(ids, passages, scs, total, {"total": (time.perf_counter() - t0) * 1000,
                                                          "stages": {"bm25": t_bm, "dense": t_de}}, stale,
                              {"ranked_ids": [d for d, _ in ranked[:200]], "version_filter": self.version_filter})
