"""Robustness check: recompute the routing-headroom oracle on the INDEPENDENT
gi:p0..p5 family (planner.md §5 "flip to G's classes", done with a genuinely distinct
reference family rather than aliases). If the headroom survives a different plan family,
it is not an artifact of K's exact ref:p* retrievers.

  python -m mem27.planner.g_robustness --data data/lme_s_k --split test --k 5
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict, Counter

import numpy as np

from mem27.planner.schema import load_notes, load_questions
from mem27 import metrics as M
from mem27.planner import ref_retrievers
from mem27.planner.ref_retrievers import load_query_embeddings
from mem27.planner.g_plans_indep import register_gi, GI_CLASSES
from mem27.oracle import paired_bootstrap

PLANS = [c.plan_id for c in GI_CLASSES]  # gi:p0..p5 (cheap->expensive-ish)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lme_s_k")
    ap.add_argument("--emb", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--success-on", choices=["session", "note"], default="note")
    ap.add_argument("--out", default="runs/analysis/g_robustness.json")
    args = ap.parse_args()
    emb_dir = args.emb or os.path.join(args.data, "emb")
    register_gi()

    notes = load_notes(os.path.join(args.data, "notes.jsonl"))
    by_hist = OrderedDict()
    for n in notes:
        by_hist.setdefault(n.history_id, []).append(n)
    questions = [q for q in load_questions(os.path.join(args.data, "questions.jsonl"))
                 if (not args.split or q.split == args.split)]
    qemb = load_query_embeddings(emb_dir)

    succ = {}  # qid -> plan -> 0/1
    for q in questions:
        idx = ref_retrievers.load_history_index(q.question_id, by_hist[q.history_id], emb_dir)
        qvec = qemb.get(q.question_id, np.zeros(768, np.float32))
        row = {}
        from mem27 import registry
        for p in PLANS:
            r = registry.get(p).retrieve(q.question, qvec, idx)
            note_ids = [idx.notes[i].note_id for i in r.indices[:20]]
            row[p] = M.success(note_ids, q, args.success_on, args.k)
        succ[q.question_id] = row

    qids = [q.question_id for q in questions]
    n = len(qids)
    per_plan = {p: sum(succ[q][p] for q in qids) / n for p in PLANS}
    best = max(PLANS, key=lambda p: per_plan[p])
    oracle = {q: max(succ[q][p] for p in PLANS) for q in qids}
    any_of = sum(oracle.values()) / n
    delta = [oracle[q] - succ[q][best] for q in qids]
    point, lo, hi = paired_bootstrap(delta, B=10000, seed=0)
    labels = Counter(min((p for p in PLANS if succ[q][p]), default="none") for q in qids)

    out = {
        "family": "gi (independent reference family)", "split": args.split,
        "success_on": args.success_on, "k": args.k, "n": n,
        "per_plan_success": {p: round(per_plan[p], 4) for p in PLANS},
        "best_fixed_plan": best, "best_fixed_success": round(per_plan[best], 4),
        "any_of_plans_upper_bound": round(any_of, 4),
        "oracle_minus_best_fixed": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "oracle_label_distribution": dict(labels),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"=== gi independent family: {args.success_on}-hit@{args.k} (test n={n}) ===")
    for p in PLANS:
        star = "  <- best" if p == best else ""
        print(f"  {p}: {per_plan[p]:.4f}{star}")
    print(f"  any-of upper bound : {any_of:.4f}")
    print(f"  oracle - best      : {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  labels             : {dict(labels)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
