"""A3: a learned per-question retrieval router (planner.md section 3.x / A3).

Where a0..a2 pick a fixed plan by a constant, a rule table, or a confidence
cascade, a3 *learns* the choice from features available at inference time (no
peeking at gold). It approximates the routing-headroom oracle (mem27.oracle):
per question, choose which of the six fixed plans ref:p0..p5 to run.

Pieces:
  features(question, bundles_by_plan_for_qid, qvec, index) -> np.ndarray
      question_type one-hot (6), question length (words), #sessions, #notes,
      and cheap-plan retrieval signals read from the already-materialized
      ref:p0/p1/p2 bundles (BM25 top-1 + margin, dense top-1 + margin, hybrid
      margin). All computable without the answer.
  Scaler    standardize features (fit on dev).
  A3Router  numpy multinomial logistic regression (L2, gradient descent,
            seed 0). Predicts a fixed-plan id per question.

Training label = cheapest fixed plan (ladder order p0<...<p5) that succeeds at
turn-hit@5 on the DEV split; if none succeed, the overall best-dev plan. Train
on dev only (100 q), evaluate on test (400 q).
"""
from __future__ import annotations

import numpy as np

from mem27 import registry
from mem27.metrics import success as success_fn
from mem27.schema import QUESTION_TYPES

LADDER = registry.FIXED_PLANS  # cheapest -> most expensive


# ---- features -------------------------------------------------------------
def _top1_margin(scores) -> tuple[float, float]:
    """(top-1 score, top1 - top2 margin) from a ranked score list."""
    if not scores:
        return 0.0, 0.0
    top1 = float(scores[0])
    top2 = float(scores[1]) if len(scores) > 1 else 0.0
    return top1, top1 - top2


def _overlap_at_k(a, b, k) -> float:
    """Jaccard-style overlap of two ranked id lists over their top-k."""
    if not a or not b:
        return 0.0
    return len(set(a[:k]) & set(b[:k])) / k


FEATURE_NAMES = (
    [f"qtype={t}" for t in QUESTION_TYPES]
    + ["q_len_words", "n_sessions", "n_notes",
       "bm25_top1", "bm25_margin", "dense_top1", "dense_margin", "hybrid_margin",
       # retriever-agreement signals: when cheap retrievers agree, cheap plan
       # likely suffices; disagreement -> escalate.
       "overlap_bm25_dense@5", "overlap_dense_hybrid@5", "overlap_bm25_dense@20",
       "dense_top3_mean"]
)


def features(question, bundles_by_plan_for_qid, qvec, index) -> np.ndarray:
    """Numeric feature vector for one question (answer-agnostic).

    bundles_by_plan_for_qid: {plan_id: Bundle} for this question's qid.
    qvec: query embedding (unused directly; dense signals come from bundles).
    index: HistoryIndex for this question's history (for #sessions / #notes).
    """
    feat: list[float] = [1.0 if question.question_type == t else 0.0
                         for t in QUESTION_TYPES]
    feat.append(float(len((question.question or "").split())))
    n_sessions = len(index._sess_order) if index is not None else 0
    n_notes = index.n if index is not None else 0
    feat.append(float(n_sessions))
    feat.append(float(n_notes))

    def scores(plan):
        b = bundles_by_plan_for_qid.get(plan)
        return b.scores if b is not None else []

    def note_ids(plan):
        b = bundles_by_plan_for_qid.get(plan)
        return b.note_ids if b is not None else []

    bm25_top1, bm25_margin = _top1_margin(scores("ref:p0"))
    dense_scores = scores("ref:p1")
    dense_top1, dense_margin = _top1_margin(dense_scores)
    _, hybrid_margin = _top1_margin(scores("ref:p2"))
    feat += [bm25_top1, bm25_margin, dense_top1, dense_margin, hybrid_margin]

    p0, p1, p2 = note_ids("ref:p0"), note_ids("ref:p1"), note_ids("ref:p2")
    feat += [_overlap_at_k(p0, p1, 5), _overlap_at_k(p1, p2, 5),
             _overlap_at_k(p0, p1, 20)]
    feat.append(float(np.mean([float(x) for x in dense_scores[:3]]))
                if dense_scores else 0.0)
    return np.asarray(feat, dtype=np.float64)


# ---- standardizer ---------------------------------------------------------
class Scaler:
    """Zero-mean / unit-std per column; constant columns pass through."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "Scaler":
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_


# ---- labels ---------------------------------------------------------------
def make_labels(questions, plan_bundles, success_on="note", k=5):
    """Per-question label = cheapest plan that succeeds; else best-overall plan.

    plan_bundles: {plan_id: {qid: Bundle}} (as mem27.oracle.load_plan_bundles).
    Returns (labels: {qid: plan_id}, best_overall_plan).
    """
    plans = [p for p in LADDER if p in plan_bundles]
    qids = [q.question_id for q in questions]
    succ = {}
    for q in questions:
        succ[q.question_id] = {
            p: (success_fn(plan_bundles[p][q.question_id].note_ids, q, success_on, k)
                if q.question_id in plan_bundles[p] else 0)
            for p in plans
        }
    per_plan_mean = {p: sum(succ[qid][p] for qid in qids) / max(1, len(qids))
                     for p in plans}
    best_overall = max(plans, key=lambda p: per_plan_mean[p])
    labels = {}
    for q in questions:
        qid = q.question_id
        chosen = next((p for p in plans if succ[qid][p]), None)
        labels[qid] = chosen if chosen is not None else best_overall
    return labels, best_overall


# ---- multinomial logistic regression (numpy) ------------------------------
class A3Router:
    """L2-regularized softmax classifier trained by batch gradient descent.

    Classes are the six fixed plans (fixed order = LADDER), so predictions are
    always valid plan ids even if a class is unseen in training.
    """

    def __init__(self, classes=None, lr=0.5, l2=1e-3, n_iter=2000, seed=0):
        self.classes = list(classes) if classes is not None else list(LADDER)
        self.lr = lr
        self.l2 = l2
        self.n_iter = n_iter
        self.seed = seed
        self.scaler = Scaler()
        self.W = None  # (d+1, C) including bias row
        self._cls_index = {c: i for i, c in enumerate(self.classes)}

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def _design(self, Xs: np.ndarray) -> np.ndarray:
        return np.hstack([Xs, np.ones((Xs.shape[0], 1))])  # bias column

    def fit(self, X: np.ndarray, y_plans) -> "A3Router":
        np.random.seed(self.seed)
        self.scaler.fit(X)
        Xs = self._design(self.scaler.transform(X))
        n, d = Xs.shape
        C = len(self.classes)
        Y = np.zeros((n, C))
        for i, p in enumerate(y_plans):
            Y[i, self._cls_index[p]] = 1.0
        W = np.zeros((d, C))
        for _ in range(self.n_iter):
            P = self._softmax(Xs @ W)
            grad = Xs.T @ (P - Y) / n
            grad[:-1] += self.l2 * W[:-1]  # L2 on weights, not bias
            W -= self.lr * grad
        self.W = W
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self._design(self.scaler.transform(X))
        return self._softmax(Xs @ self.W)

    def predict(self, X: np.ndarray) -> list[str]:
        idx = self.predict_proba(X).argmax(axis=1)
        return [self.classes[i] for i in idx]
