"""LongMemEval_S leakage / distractor-tier audit (V; CPU only).

1. Session sources: every haystack session is gold evidence, a ShareGPT or UltraChat
   chat (real filler), or a simulated user session written for another question.
2. Question-agnostic detector: TF-IDF + logistic regression on session text only (no
   question), 5-fold GroupKFold by history. Reports AUC overall and per distractor
   source, and the session recall@k of ranking each haystack by this prior alone.
3. Prior fusion: does adding the question-agnostic prior to BM25 session ranking raise
   recall? (If yes, retrieval numbers on this benchmark partly reward style artifacts.)
4. Distractor tiers of retrieval errors: for each ranking run, the share of non-gold
   sessions in the top-k that are real filler vs simulated sessions.

  python scripts/audit_lme_leakage.py --data data/lme_s --runs runs/vref_bm25_lme_s --out runs/_tables/lme_leakage
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import write_json  # noqa: E402
from mem27.vschema import Question, iter_jsonl, read_jsonl, session_of  # noqa: E402


def source_of(orig: str) -> str:
    if orig.startswith("answer"):
        return "gold_named"
    if orig.startswith("sharegpt"):
        return "sharegpt"
    if orig.startswith("ultrachat"):
        return "ultrachat"
    return "simulated"


def session_ranking(note_ids: list[str]) -> list[str]:
    out = []
    for n in note_ids:
        s = session_of(n)
        if s not in out:
            out.append(s)
    return out


def recall_at(ranked: list[str], gold: set[str], k: int) -> float:
    return len(gold & set(ranked[:k])) / len(gold) if gold else float("nan")


def main(argv=None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lme_s")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default="runs/_tables/lme_leakage")
    a = ap.parse_args(argv)
    data = Path(a.data)
    qs = {q.question_id: q for q in read_jsonl(data / "questions.jsonl", Question)}
    src = {}
    for r in iter_jsonl(data.parent / "_audit" / data.name / "session_id_map.jsonl"):
        src[(r["history_id"], r["session_id"])] = source_of(r["orig_session_id"])
    rows = []
    for n in iter_jsonl(data / "notes.jsonl"):
        if n["turn_idx"] != -1:
            continue
        h, s = n["history_id"], n["session_id"]
        q = qs[h]
        rows.append({"h": h, "s": s, "text": n["text"], "gold": s in q.gold_session_ids,
                     "src": src[(h, s)], "abs": h.endswith("_abs"), "n_turns": n["metadata"].get("n_turns", 0)})
    out: dict = {"n_sessions": len(rows), "sources": dict(Counter(r["src"] for r in rows)),
                 "sources_per_history": {k: v / len(qs) for k, v in Counter(r["src"] for r in rows).items()},
                 "gold_by_source": dict(Counter(r["src"] for r in rows if r["gold"]))}
    print(json.dumps(out, indent=1))

    # ---- question-agnostic detector
    texts = [r["text"] for r in rows]
    y = np.array([r["gold"] for r in rows], int)
    groups = np.array([r["h"] for r in rows])
    prob = np.zeros(len(rows))
    vec_kw = dict(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
    for tr, te in GroupKFold(n_splits=5).split(texts, y, groups):
        vec = TfidfVectorizer(**vec_kw)
        Xtr = vec.fit_transform([texts[i] for i in tr])
        clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        prob[te] = clf.predict_proba(vec.transform([texts[i] for i in te]))[:, 1]
    srcs = np.array([r["src"] for r in rows])
    det = {"auc_all": float(roc_auc_score(y, prob))}
    for s in ("sharegpt", "ultrachat", "simulated"):
        m = (y == 1) | (srcs == s)
        det[f"auc_gold_vs_{s}"] = float(roc_auc_score(y[m], prob[m]))
    # length-only baseline (how much is just "evidence sessions look different in size")
    lens = np.array([len(t) for t in texts], float)
    det["auc_length_only"] = float(max(roc_auc_score(y, lens), roc_auc_score(y, -lens)))
    by_h = defaultdict(list)
    for i, r in enumerate(rows):
        by_h[r["h"]].append(i)
    rec = defaultdict(list)
    prior_rank: dict[str, list[str]] = {}
    for h, idx in by_h.items():
        if h.endswith("_abs"):
            continue
        order = [rows[i]["s"] for i in sorted(idx, key=lambda i: -prob[i])]
        prior_rank[h] = order
        gold = set(qs[h].gold_session_ids)
        for k in (1, 3, 5, 10):
            rec[k].append(recall_at(order, gold, k))
        nsess = len(idx)
        for k in (5, 10):
            rec[f"random@{k}"].append(min(1.0, k / nsess))
    det["prior_only_session_recall"] = {str(k): float(np.mean(v)) for k, v in rec.items()}
    out["detector"] = det
    print(json.dumps(det, indent=1))

    # ---- prior fusion + distractor tiers for each ranking run
    pr = {(rows[i]["h"], rows[i]["s"]): prob[i] for i in range(len(rows))}
    out["runs"] = {}
    for run in a.runs:
        run = Path(run)
        f = run / "rankings.jsonl" if (run / "rankings.jsonl").exists() else run / "bundles.jsonl"
        res = defaultdict(list)
        tiers = Counter()
        for b in iter_jsonl(f):
            h = b["question_id"]
            if h.endswith("_abs"):
                continue
            gold = set(qs[h].gold_session_ids)
            base = session_ranking(b["note_ids"])
            for k in (5, 10):
                res[f"base@{k}"].append(recall_at(base, gold, k))
            for s in base[:10]:
                if s not in gold:
                    tiers[src[(h, s)]] += 1
            # fuse: RRF of retrieval session rank and prior rank (prior over all sessions of the history)
            pri = prior_rank.get(h, [])
            for w in (0.5, 1.0):
                sc = defaultdict(float)
                for i, s in enumerate(base):
                    sc[s] += 1 / (60 + i + 1)
                for i, s in enumerate(pri):
                    sc[s] += w / (60 + i + 1)
                fused = sorted(sc, key=lambda s: -sc[s])
                for k in (5, 10):
                    res[f"fused_w{w}@{k}"].append(recall_at(fused, gold, k))
        tot = sum(tiers.values())
        out["runs"][run.name] = {"session_recall": {k: float(np.mean(v)) for k, v in res.items()},
                                 "nongold_top10_by_source": {k: v / tot for k, v in tiers.items()} if tot else {}}
        print(run.name, json.dumps(out["runs"][run.name]))
    write_json(f"{a.out}.json", out)
    # top features (interpretability)
    vec = TfidfVectorizer(**vec_kw)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(X, y)
    names = np.array(vec.get_feature_names_out())
    co = clf.coef_[0]
    feats = {"gold_like": names[np.argsort(-co)[:40]].tolist(), "nongold_like": names[np.argsort(co)[:40]].tolist()}
    write_json(f"{a.out}_features.json", feats)
    print("gold-like features:", ", ".join(feats["gold_like"][:25]))


if __name__ == "__main__":
    main()
