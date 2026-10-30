"""A5 — learned weighted fusion: the strongest simple attempt to capture the headroom.

A4-fuse used *uniform* RRF weights and fell below best-fixed. The obvious objection is
"did you optimize the fusion?". A5 learns per-plan RRF weights on dev (coordinate ascent
to maximize note-hit@5), then evaluates the frozen weights on test. If even a dev-tuned
linear fusion cannot beat the best single plan, the routing headroom is unexploitable by
*any* fixed linear combination of the plans — the strongest form of the H2 conclusion.

  python -m mem27.planner.a5_weighted_fusion --dataset data/lme_s_k \
      --runs 'runs/ref_p*_lme_s_k_seed0_warm' --k 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from mem27.schema import load_questions, MAX_RESULTS
from mem27.oracle import load_plan_bundles, paired_bootstrap
from mem27 import registry
from mem27.metrics import success


def weighted_rrf(rankings: dict, weights: dict, k0: int = 60, topn: int = MAX_RESULTS) -> list:
    """rankings: {plan: [note_id,...]}. Fused score = sum_p w_p / (k0 + rank_p)."""
    scores = {}
    for p, ranking in rankings.items():
        w = weights[p]
        for rank, nid in enumerate(ranking):
            scores[nid] = scores.get(nid, 0.0) + w / (k0 + rank)
    return sorted(scores, key=lambda n: scores[n], reverse=True)[:topn]


def fused_rate(qids, pb, plans, qmap, weights, k):
    hit = 0
    for q in qids:
        rk = {p: pb[q][p].note_ids for p in plans if p in pb[q]}
        fused = weighted_rrf(rk, weights)
        hit += success(fused, qmap[q], "note", k)
    return hit / len(qids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/lme_s_k")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default="runs/analysis/a5_weighted_fusion.json")
    args = ap.parse_args()

    run_dirs = []
    for pat in args.runs:
        run_dirs.extend(sorted(glob.glob(pat)))
    plan_bundles = load_plan_bundles(run_dirs)
    plans = [p for p in registry.FIXED_PLANS if p in plan_bundles]

    questions = load_questions(os.path.join(args.dataset, "questions.jsonl"))
    dev = [q for q in questions if q.split == "dev"]
    test = [q for q in questions if q.split == "test"]
    qmap = {q.question_id: q for q in questions}
    # per-question plan->bundle
    pb = {}
    for p in plans:
        for qid, b in plan_bundles[p].items():
            pb.setdefault(qid, {})[p] = b

    dev_ids = [q.question_id for q in dev if q.question_id in pb]
    test_ids = [q.question_id for q in test if q.question_id in pb]

    # --- coordinate ascent on dev weights ---
    weights = {p: 1.0 for p in plans}
    best_dev = fused_rate(dev_ids, pb, plans, qmap, weights, args.k)
    grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    for _ in range(4):  # sweeps
        improved = False
        for p in plans:
            base = weights[p]
            best_w, best_val = base, best_dev
            for w in grid:
                weights[p] = w
                val = fused_rate(dev_ids, pb, plans, qmap, weights, args.k)
                if val > best_val:
                    best_val, best_w = val, w
            weights[p] = best_w
            if best_val > best_dev:
                best_dev, improved = best_val, True
        if not improved:
            break

    # --- evaluate on test ---
    # per-plan + best fixed + oracle on test
    def succ(qid, p):
        return success(pb[qid][p].note_ids, qmap[qid], "note", args.k)
    per_plan = {p: sum(succ(q, p) for q in test_ids) / len(test_ids) for p in plans}
    best = max(per_plan, key=per_plan.get)
    oracle = {q: max(succ(q, p) for p in plans) for q in test_ids}
    a5 = {}
    for q in test_ids:
        rk = {p: pb[q][p].note_ids for p in plans if p in pb[q]}
        a5[q] = success(weighted_rrf(rk, weights), qmap[q], "note", args.k)
    a5_rate = sum(a5.values()) / len(test_ids)

    def ci(router):
        delta = [router[q] - succ(q, best) for q in test_ids]
        return paired_bootstrap(delta, B=args.bootstrap, seed=0)
    a5_pt, a5_lo, a5_hi = ci(a5)
    o_pt, o_lo, o_hi = ci(oracle)

    out = {
        "k": args.k, "n_test": len(test_ids), "n_dev": len(dev_ids),
        "learned_weights": {p: round(weights[p], 3) for p in plans},
        "dev_fused_hit": round(best_dev, 4),
        "best_fixed_plan": best, "best_fixed_test": round(per_plan[best], 4),
        "oracle_test": round(sum(oracle.values()) / len(test_ids), 4),
        "oracle_minus_best": round(o_pt, 4), "oracle_ci95": [round(o_lo, 4), round(o_hi, 4)],
        "a5_test": round(a5_rate, 4),
        "a5_minus_best": round(a5_pt, 4), "a5_ci95": [round(a5_lo, 4), round(a5_hi, 4)],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"=== A5 learned weighted fusion (note-hit@{args.k}, test n={len(test_ids)}) ===")
    print(f"  learned weights (dev): {out['learned_weights']}")
    print(f"  best fixed {best}: {per_plan[best]:.4f}")
    print(f"  oracle (any-of)   : {out['oracle_test']:.4f}  (+{o_pt:.4f} [{o_lo:.4f},{o_hi:.4f}])")
    print(f"  A5 weighted fusion: {a5_rate:.4f}  (vs best {a5_pt:+.4f} [{a5_lo:.4f},{a5_hi:.4f}])")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
