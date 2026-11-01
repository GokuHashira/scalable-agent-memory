"""A genuinely-independent reference plan family gi:p0..p5 (planner.md §5).

Unlike `g_plans.py` (thin aliases that prove the registry swap), this module is a
*second, algorithmically-distinct* implementation of the six plan roles. It is still a
stand-in for G's real classes, but using different internals on purpose: it lets us test
whether the routing-headroom finding is an artifact of K's exact `ref:p*` retrievers or
survives a different plan family (a robustness check).

Each retriever exposes the same contract as the ref retrievers:
    .retrieve(qtext, qvec, index) -> RankedResult(.indices, .scores, .trace)

Distinct choices vs ref:
  gi:p0  TF-IDF cosine (vs ref BM25)
  gi:p1  dense + pseudo-relevance feedback (query += top-1 doc, re-rank)
  gi:p2  linear-weighted hybrid 0.6*dense + 0.4*lexical (vs RRF)
  gi:p3  centroid-summary coarse-to-fine (session mean emb, vs first/last/longest)
  gi:p4  kNN-graph traversal with PageRank re-rank (vs cosine re-rank)
  gi:p5  MMR-diversified hybrid (vs hybrid+traversal)
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from mem27.planner.schema import MAX_RESULTS
from mem27.planner.ref_retrievers import RankedResult, HistoryIndex
from mem27.planner.refcommon import bm25_tokenize, passage_text


# ---- lexical: TF-IDF ------------------------------------------------------
def _tfidf_scores(index: HistoryIndex, qtext: str) -> np.ndarray:
    docs = [bm25_tokenize(passage_text(n.role, n.text)) for n in index.notes]
    N = len(docs) or 1
    df = defaultdict(int)
    for d in docs:
        for t in set(d):
            df[t] += 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}
    qtok = bm25_tokenize(qtext)
    qset = set(qtok)
    scores = np.zeros(N, dtype=np.float32)
    for i, d in enumerate(docs):
        if not d:
            continue
        tf = defaultdict(int)
        for t in d:
            tf[t] += 1
        dl = len(d)
        s = 0.0
        for t in qset:
            if t in tf:
                s += (tf[t] / dl) * idf.get(t, 0.0) ** 2
        scores[i] = s
    return scores


def _dense(index, qvec):
    if index.emb is None or index.n == 0:
        return np.zeros(index.n, dtype=np.float32)
    return index.emb @ qvec


def _norm01(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class _GI:
    name = "GI"
    plan_id = "gi:px"
    def retrieve(self, qtext, qvec, index): raise NotImplementedError


class GITfidf(_GI):
    name, plan_id = "GITfidf", "gi:p0"
    def retrieve(self, qtext, qvec, index):
        s = _tfidf_scores(index, qtext)
        order = list(np.argsort(-s)[:MAX_RESULTS])
        return RankedResult(order, [float(s[i]) for i in order], {"impl": "gi", "stage": "tfidf"})


class GIDensePRF(_GI):
    name, plan_id = "GIDensePRF", "gi:p1"
    def retrieve(self, qtext, qvec, index):
        s = _dense(index, qvec)
        if index.emb is not None and index.n:
            top1 = int(np.argmax(s))
            q2 = qvec + 0.5 * index.emb[top1]           # pseudo-relevance feedback
            q2 = q2 / (np.linalg.norm(q2) + 1e-9)
            s = _dense(index, q2)
        order = list(np.argsort(-s)[:MAX_RESULTS])
        return RankedResult(order, [float(s[i]) for i in order], {"impl": "gi", "stage": "dense_prf"})


class GILinearHybrid(_GI):
    name, plan_id = "GILinearHybrid", "gi:p2"
    def retrieve(self, qtext, qvec, index):
        d = _norm01(_dense(index, qvec))
        l = _norm01(_tfidf_scores(index, qtext))
        s = 0.6 * d + 0.4 * l
        order = list(np.argsort(-s)[:MAX_RESULTS])
        return RankedResult(order, [float(s[i]) for i in order], {"impl": "gi", "stage": "linear_hybrid"})


class GICentroidSummary(_GI):
    name, plan_id = "GICentroidSummary", "gi:p3"
    def retrieve(self, qtext, qvec, index):
        if index.emb is None or index.n == 0:
            return RankedResult([], [], {"impl": "gi", "stage": "centroid", "empty": True})
        sess_score = {}
        for sid, members in index._sess_members.items():
            c = index.emb[members].mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-9)
            sess_score[sid] = float(c @ qvec)
        ranked = sorted(index._sess_order, key=lambda s: -sess_score[s])
        cand = []
        for sid in ranked:
            cand.extend(index._sess_members[sid])
            if len(cand) >= 4 * MAX_RESULTS:
                break
        sc = index.emb[cand] @ qvec
        order = [cand[j] for j in np.argsort(-sc)[:MAX_RESULTS]]
        return RankedResult(order, [float(index.emb[i] @ qvec) for i in order],
                            {"impl": "gi", "stage": "centroid_summary"})


class GIPageRankTraversal(_GI):
    name, plan_id = "GIPageRankTraversal", "gi:p4"
    def retrieve(self, qtext, qvec, index):
        if index.emb is None or index.n == 0:
            return RankedResult([], [], {"impl": "gi", "stage": "pagerank", "empty": True})
        n = index.n
        nb = index.neighbors(5)
        # build sparse-ish adjacency (kNN graph), personalized PR toward query sim
        sim = _norm01(_dense(index, qvec))
        teleport = sim / (sim.sum() + 1e-9)
        pr = np.full(n, 1.0 / n, dtype=np.float32)
        # transition: uniform over each node's neighbors
        for _ in range(12):
            nxt = 0.15 * teleport
            contrib = np.zeros(n, dtype=np.float32)
            for i in range(n):
                deg = len(nb[i])
                if deg:
                    share = 0.85 * pr[i] / deg
                    for j in nb[i]:
                        contrib[int(j)] += share
                else:
                    nxt += 0.85 * pr[i] * teleport
            pr = nxt + contrib
            pr /= pr.sum() + 1e-9
        order = list(np.argsort(-pr)[:MAX_RESULTS])
        return RankedResult(order, [float(pr[i]) for i in order], {"impl": "gi", "stage": "pagerank"})


class GIMMRHybrid(_GI):
    name, plan_id = "GIMMRHybrid", "gi:p5"
    LAMBDA = 0.7
    def retrieve(self, qtext, qvec, index):
        if index.emb is None or index.n == 0:
            return RankedResult([], [], {"impl": "gi", "stage": "mmr", "empty": True})
        rel = 0.6 * _norm01(_dense(index, qvec)) + 0.4 * _norm01(_tfidf_scores(index, qtext))
        pool = list(np.argsort(-rel)[:100])
        selected, sel_scores = [], []
        emb = index.emb
        while pool and len(selected) < MAX_RESULTS:
            best, best_val = None, -1e9
            for c in pool:
                div = 0.0
                if selected:
                    div = max(float(emb[c] @ emb[s]) for s in selected)
                val = self.LAMBDA * rel[c] - (1 - self.LAMBDA) * div
                if val > best_val:
                    best_val, best = val, c
            selected.append(best)
            sel_scores.append(float(rel[best]))
            pool.remove(best)
        return RankedResult(selected, sel_scores, {"impl": "gi", "stage": "mmr_hybrid"})


GI_CLASSES = [GITfidf, GIDensePRF, GILinearHybrid, GICentroidSummary,
              GIPageRankTraversal, GIMMRHybrid]


def register_gi(as_ref: bool = False):
    """Register gi:p0..p5. If as_ref, ALSO alias onto ref:pK so oracle/run_plan
    (which iterate FIXED_PLANS) run on this independent family unchanged."""
    from mem27 import registry
    from mem27.planner import ref_retrievers  # ensure neighbors()/emb available
    ids = []
    for cls in GI_CLASSES:
        inst = cls()
        registry.register(cls.plan_id, inst)
        ids.append(cls.plan_id)
        if as_ref:
            registry.register("ref:" + cls.plan_id.split(":")[1], inst)
    return ids
