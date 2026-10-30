"""A4 — can the routing headroom actually be captured? (planner.md §1 "learned routing
is the next step"; A3 the learned per-question selector matched best-fixed, CI straddles 0).

A4 tries two *principled, non-oracle* strategies A3 did not, both using the plans' actual
retrieval outputs rather than predicting from the query alone:

  A4-fuse    : reciprocal-rank fusion (RRF, k0=60) of ALL fixed plans' note rankings into
               one ranking per question. Tests whether *fusing* beats *selecting* — i.e.
               whether the any-of-plans diversity is capturable without foresight.
  A4-conf    : per-question, SELECT the single plan with the largest top-1/top-2 score
               margin (a confidence proxy). Tests whether retrieval confidence predicts
               which plan is right.

Both are compared, with a paired bootstrap, to the best single fixed plan (the quantity
H2's headroom is measured against) and to the per-question oracle (the upper bound).

  python -m mem27.planner.a4_router --dataset data/lme_s_k \
      --runs runs/ref_p*_lme_s_k_seed0_warm --success-on note --k 5 --split test
"""
from __future__ import annotations

import argparse
import glob
import os

from mem27.schema import load_questions
from mem27.oracle import load_plan_bundles, paired_bootstrap
from mem27 import registry
from mem27.metrics import success


def rrf_fuse(rankings: list[list[str]], k0: int = 60) -> list[str]:
    """Reciprocal-rank fusion of several note-id rankings into one."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, nid in enumerate(ranking):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k0 + rank)
    return sorted(scores, key=lambda n: scores[n], reverse=True)


def conf_select(plan_to_bundle: dict) -> str:
    """Pick the plan whose bundle has the largest top-1 minus top-2 score margin."""
    best_plan, best_margin = None, -1e9
    for p, b in plan_to_bundle.items():
        sc = b.scores or []
        margin = (sc[0] - sc[1]) if len(sc) >= 2 else (sc[0] if sc else 0.0)
        if margin > best_margin:
            best_margin, best_plan = margin, p
    return best_plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/lme_s_k")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--success-on", choices=["session", "note"], default="note")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--split", default="test")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", default="runs/analysis/a4_router.json")
    args = ap.parse_args()

    run_dirs = []
    for pat in args.runs:
        run_dirs.extend(sorted(glob.glob(pat)))
    plan_bundles = load_plan_bundles(run_dirs)
    plans = [p for p in registry.FIXED_PLANS if p in plan_bundles]

    questions = load_questions(os.path.join(args.dataset, "questions.jsonl"))
    if args.split:
        questions = [q for q in questions if q.split == args.split]
    qids = [q.question_id for q in questions]
    qmap = {q.question_id: q for q in questions}

    # per-plan success + per-question plan->bundle
    succ = {q: {} for q in qids}
    pb = {q: {} for q in qids}
    for p in plans:
        for qid, b in plan_bundles[p].items():
            if qid in qmap:
                succ[qid][p] = success(b.note_ids, qmap[qid], args.success_on, args.k)
                pb[qid][p] = b

    n = len(qids)
    per_plan = {p: sum(succ[q][p] for q in qids) / n for p in plans}
    best = max(per_plan, key=per_plan.get)
    oracle = {q: max(succ[q][p] for p in plans) for q in qids}
    any_of = sum(oracle.values()) / n

    # A4-fuse
    fuse_succ = {}
    for q in qids:
        fused = rrf_fuse([pb[q][p].note_ids for p in plans if p in pb[q]])
        fuse_succ[q] = success(fused, qmap[q], args.success_on, args.k)
    fuse_rate = sum(fuse_succ.values()) / n

    # A4-conf
    conf_succ = {}
    for q in qids:
        chosen = conf_select(pb[q])
        conf_succ[q] = succ[q][chosen]
    conf_rate = sum(conf_succ.values()) / n

    def ci_vs_best(router_succ):
        delta = [router_succ[q] - succ[q][best] for q in qids]
        return paired_bootstrap(delta, B=args.bootstrap, seed=0)

    fuse_pt, fuse_lo, fuse_hi = ci_vs_best(fuse_succ)
    conf_pt, conf_lo, conf_hi = ci_vs_best(conf_succ)
    orc_pt, orc_lo, orc_hi = ci_vs_best(oracle)

    out = {
        "success_on": args.success_on, "k": args.k, "split": args.split, "n": n,
        "plans": plans, "per_plan_success": {p: round(per_plan[p], 4) for p in plans},
        "best_fixed_plan": best, "best_fixed_success": round(per_plan[best], 4),
        "oracle_any_of": round(any_of, 4),
        "a4_fuse_success": round(fuse_rate, 4),
        "a4_fuse_minus_best": round(fuse_pt, 4), "a4_fuse_ci95": [round(fuse_lo, 4), round(fuse_hi, 4)],
        "a4_conf_success": round(conf_rate, 4),
        "a4_conf_minus_best": round(conf_pt, 4), "a4_conf_ci95": [round(conf_lo, 4), round(conf_hi, 4)],
        "oracle_minus_best": round(orc_pt, 4), "oracle_ci95": [round(orc_lo, 4), round(orc_hi, 4)],
    }
    import json
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"=== A4 router: {args.success_on}-hit@{args.k} {args.split} (n={n}) ===")
    print(f"  best fixed {best:8s}     : {per_plan[best]:.4f}")
    print(f"  oracle (any-of)        : {any_of:.4f}  (oracle-best {orc_pt:+.4f} [{orc_lo:.4f},{orc_hi:.4f}])")
    print(f"  A4-fuse (RRF all plans): {fuse_rate:.4f}  (vs best {fuse_pt:+.4f} [{fuse_lo:.4f},{fuse_hi:.4f}])")
    print(f"  A4-conf (margin select): {conf_rate:.4f}  (vs best {conf_pt:+.4f} [{conf_lo:.4f},{conf_hi:.4f}])")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
