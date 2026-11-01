"""Reference retrievers ref:p0..p5 (planner.md section 3.2).

These operate on a per-history HistoryIndex using *cached* passage embeddings and
*cached* query embeddings (produced by mem27.planner.embed), so no neural model is
loaded at retrieval time. Each retriever returns a RankedResult: ranked note indices
with scores + a trace, up to MAX_RESULTS candidates. Bundle assembly / token capping
happens in run_plan.

Cost ladder (self-register under ref:p0..p5):
  p0 RefBM25  p1 RefDense  p2 RefHybrid  p3 RefSummaryFirst
  p4 RefTraversal  p5 RefHybridTrav
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from mem27.planner.schema import Note, MAX_RESULTS, RRF_K, TRAVERSAL_H, TRAVERSAL_L
from mem27 import registry
from mem27.planner.refcommon import (
    bm25_tokenize, rrf_fuse, group_sessions, passage_text,
)


# --------------------------------------------------------------------------
@dataclass
class RankedResult:
    indices: list[int]              # ranked note indices (into index.notes)
    scores: list[float]
    trace: dict = field(default_factory=dict)


@dataclass
class HistoryIndex:
    history_id: str
    notes: list[Note]
    emb: np.ndarray | None          # (N,768) L2-normalized float32, or None
    _bm25: object = None
    _sess_order: list = None
    _sess_members: dict = None
    _sess_date: dict = None
    _neighbors: np.ndarray = None    # (N,k) semantic neighbor indices

    def __post_init__(self):
        self._sess_order, self._sess_members, self._sess_date = group_sessions(self.notes)

    @property
    def n(self):
        return len(self.notes)

    # ---- lazy BM25 ----
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            toks = [bm25_tokenize(passage_text(x.role, x.text)) for x in self.notes]
            self._bm25 = BM25Okapi(toks)
        return self._bm25

    # ---- semantic neighbor graph (top-k cosine, excluding self) ----
    def neighbors(self, k: int = 5) -> np.ndarray:
        if self._neighbors is None:
            if self.emb is None:
                self._neighbors = np.zeros((self.n, 0), dtype=np.int32)
            else:
                sim = self.emb @ self.emb.T          # (N,N) cosine (normalized)
                np.fill_diagonal(sim, -1.0)
                kk = min(k, max(0, self.n - 1))
                self._neighbors = np.argpartition(-sim, kk - 1, axis=1)[:, :kk] if kk > 0 \
                    else np.zeros((self.n, 0), dtype=np.int32)
        return self._neighbors


# ---- scoring primitives ---------------------------------------------------
def _bm25_ranking(index: HistoryIndex, qtext: str, topn: int) -> list[int]:
    scores = index.bm25().get_scores(bm25_tokenize(qtext))
    n = min(topn, len(scores))
    return list(np.argsort(-scores)[:n])


def _dense_scores(index: HistoryIndex, qvec: np.ndarray) -> np.ndarray:
    if index.emb is None or index.n == 0:
        return np.zeros(index.n, dtype=np.float32)
    return index.emb @ qvec                         # cosine (both normalized)


def _dense_ranking(index: HistoryIndex, qvec: np.ndarray, topn: int) -> list[int]:
    s = _dense_scores(index, qvec)
    n = min(topn, len(s))
    return list(np.argsort(-s)[:n])


def _hybrid_ranking(index: HistoryIndex, qtext: str, qvec: np.ndarray,
                    topn: int, pool: int = 100) -> tuple[list[int], dict[int, float]]:
    bm = _bm25_ranking(index, qtext, pool)
    dn = _dense_ranking(index, qvec, pool)
    fused = rrf_fuse([bm, dn], k=RRF_K)
    ranked = sorted(fused, key=lambda i: -fused[i])[:topn]
    return ranked, fused


# ---- the retrievers -------------------------------------------------------
class RefRetriever:
    name = "ref"
    plan_id = "ref:px"

    def retrieve(self, qtext: str, qvec: np.ndarray, index: HistoryIndex) -> RankedResult:
        raise NotImplementedError


class RefBM25(RefRetriever):
    name, plan_id = "RefBM25", "ref:p0"

    def retrieve(self, qtext, qvec, index):
        scores = index.bm25().get_scores(bm25_tokenize(qtext))
        order = list(np.argsort(-scores)[:MAX_RESULTS])
        return RankedResult(order, [float(scores[i]) for i in order], {"stage": "bm25"})


class RefDense(RefRetriever):
    name, plan_id = "RefDense", "ref:p1"

    def retrieve(self, qtext, qvec, index):
        s = _dense_scores(index, qvec)
        order = list(np.argsort(-s)[:MAX_RESULTS])
        return RankedResult(order, [float(s[i]) for i in order], {"stage": "dense"})


class RefHybrid(RefRetriever):
    name, plan_id = "RefHybrid", "ref:p2"

    def retrieve(self, qtext, qvec, index):
        ranked, fused = _hybrid_ranking(index, qtext, qvec, MAX_RESULTS)
        return RankedResult(ranked, [float(fused[i]) for i in ranked], {"stage": "hybrid_rrf"})


class RefSummaryFirst(RefRetriever):
    """Coarse-to-fine: rank sessions by an extractive 3-sentence summary
    (first + last + longest turn), then drill into turns of the top sessions."""
    name, plan_id = "RefSummaryFirst", "ref:p3"

    @staticmethod
    def _summary_indices(members: list[int], notes) -> list[int]:
        if not members:
            return []
        first, last = members[0], members[-1]
        longest = max(members, key=lambda i: len(notes[i].text or ""))
        # preserve order, dedup
        picks = []
        for i in (first, last, longest):
            if i not in picks:
                picks.append(i)
        return picks

    def retrieve(self, qtext, qvec, index):
        # 1. session-level scoring via summary embeddings (mean of summary turns)
        sess_score = {}
        sess_summary_idx = {}
        for sid in index._sess_order:
            members = index._sess_members[sid]
            sidx = self._summary_indices(members, index.notes)
            sess_summary_idx[sid] = sidx
            if index.emb is not None and sidx:
                sv = index.emb[sidx].mean(axis=0)
                nrm = np.linalg.norm(sv)
                sv = sv / nrm if nrm > 0 else sv
                sess_score[sid] = float(sv @ qvec)
            else:
                sess_score[sid] = 0.0
        ranked_sessions = sorted(index._sess_order, key=lambda s: -sess_score[s])
        # 2. collect turns from top sessions until we have a healthy candidate pool
        cand: list[int] = []
        used_sessions = []
        for sid in ranked_sessions:
            cand.extend(index._sess_members[sid])
            used_sessions.append(sid)
            if len(cand) >= 4 * MAX_RESULTS:
                break
        # 3. re-rank candidate turns by dense cosine to query
        if index.emb is not None and cand:
            s = index.emb[cand] @ qvec
            order = [cand[j] for j in np.argsort(-s)[:MAX_RESULTS]]
            sc = [float(index.emb[i] @ qvec) for i in order]
        else:
            order, sc = cand[:MAX_RESULTS], [0.0] * min(len(cand), MAX_RESULTS)
        return RankedResult(order, sc, {"stage": "summary_first",
                                        "sessions_opened": used_sessions[:10]})


class RefTraversal(RefRetriever):
    """Link-graph expansion. links = adjacent turns + top-5 semantic neighbours.
    anchors = hybrid top-5. Expand H=4 hops up to L=50, re-rank by cosine."""
    name, plan_id = "RefTraversal", "ref:p4"
    use_hybrid_rerank = False

    def _adjacency(self, index: HistoryIndex) -> list[set]:
        adj: list[set] = [set() for _ in range(index.n)]
        # adjacent turns within a session
        for sid, members in index._sess_members.items():
            for a, b in zip(members, members[1:]):
                adj[a].add(b)
                adj[b].add(a)
        # top-5 semantic neighbours
        nb = index.neighbors(5)
        for i in range(index.n):
            for j in nb[i]:
                adj[i].add(int(j))
                adj[int(j)].add(i)
        return adj

    def retrieve(self, qtext, qvec, index):
        if index.n == 0:
            return RankedResult([], [], {"stage": "traversal", "empty": True})
        ranked_hy, fused = _hybrid_ranking(index, qtext, qvec, 5)
        anchors = list(ranked_hy)
        adj = self._adjacency(index)
        visited = set(anchors)
        frontier = set(anchors)
        hops_used = 0
        for hop in range(TRAVERSAL_H):
            if len(visited) >= TRAVERSAL_L or not frontier:
                break
            nxt = set()
            for u in frontier:
                nxt |= adj[u]
            nxt -= visited
            # add by cosine priority, capped at L
            if index.emb is not None and nxt:
                nxt_l = sorted(nxt, key=lambda i: -float(index.emb[i] @ qvec))
            else:
                nxt_l = list(nxt)
            for u in nxt_l:
                if len(visited) >= TRAVERSAL_L:
                    break
                visited.add(u)
                nxt.discard(u)
            frontier = {u for u in nxt_l if u in visited} or set(nxt_l[:0])
            frontier = set(nxt_l[:min(len(nxt_l), 10)]) & visited
            hops_used = hop + 1
        cand = list(visited)
        if index.emb is not None and cand:
            cos = index.emb[cand] @ qvec
            if self.use_hybrid_rerank:
                dense_rank = [cand[j] for j in np.argsort(-cos)]
                hy_rank = [i for i in sorted(fused, key=lambda x: -fused[x]) if i in visited]
                fused2 = rrf_fuse([dense_rank, hy_rank], k=RRF_K)
                order = sorted(cand, key=lambda i: -fused2.get(i, 0))[:MAX_RESULTS]
                sc = [float(fused2.get(i, 0)) for i in order]
            else:
                order = [cand[j] for j in np.argsort(-cos)[:MAX_RESULTS]]
                sc = [float(index.emb[i] @ qvec) for i in order]
        else:
            order, sc = cand[:MAX_RESULTS], [0.0] * min(len(cand), MAX_RESULTS)
        return RankedResult(order, sc, {"stage": "traversal", "anchors": anchors,
                                        "hops_used": hops_used, "pool_size": len(cand)})


class RefHybridTrav(RefTraversal):
    """p5: traversal seeded by hybrid, final re-rank = RRF(hybrid, cosine). Fullest."""
    name, plan_id = "RefHybridTrav", "ref:p5"
    use_hybrid_rerank = True


# ---- registration ---------------------------------------------------------
_CLASSES = [RefBM25, RefDense, RefHybrid, RefSummaryFirst, RefTraversal, RefHybridTrav]
for _cls in _CLASSES:
    registry.register(_cls.plan_id, _cls())


# ---- index loading --------------------------------------------------------
def load_history_index(history_id: str, notes: list[Note], emb_dir: str) -> HistoryIndex:
    emb = None
    path = os.path.join(emb_dir, f"{history_id}.npy")
    if os.path.exists(path):
        emb = np.load(path).astype(np.float32)
        assert emb.shape[0] == len(notes), f"emb/notes mismatch {history_id}: {emb.shape[0]} vs {len(notes)}"
    return HistoryIndex(history_id=history_id, notes=notes, emb=emb)


def load_query_embeddings(emb_dir: str) -> dict[str, np.ndarray]:
    qp = os.path.join(emb_dir, "queries.npy")
    ip = os.path.join(emb_dir, "queries_ids.json")
    if not os.path.exists(qp):
        return {}
    mat = np.load(qp).astype(np.float32)
    ids = json.load(open(ip))
    return {qid: mat[i] for i, qid in enumerate(ids)}
