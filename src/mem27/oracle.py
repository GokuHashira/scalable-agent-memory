"""Routing-headroom oracle over fixed plans (planner.md section 1, 4).

Per question, the oracle picks the plan that succeeds (cheapest that succeeds; a
type-default silver label when none succeed). Reports:
  - per-plan mean success (best single fixed plan)
  - any-of-plans upper bound (= mean oracle success)
  - Oracle - best fixed, with 95% paired-bootstrap CI
  - oracle label distribution, silver fraction
  - per-type success table (where fixed plans disagree)

  python -m mem27.oracle --dataset data/lme_s_k \
      --runs runs/ref_p0_* runs/ref_p1_* ... --success-on session --split test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import defaultdict, Counter

from mem27.planner.schema import load_questions, load_bundles, load_answers, QUESTION_TYPES
from mem27 import registry
from mem27.metrics import success as success_fn

LADDER = registry.FIXED_PLANS  # cheapest -> most expensive
A1_TYPE_DEFAULT = {  # silver fallback by type (mirrors a1 rules)
    "single-session-user": "ref:p2", "single-session-assistant": "ref:p2",
    "single-session-preference": "ref:p3", "temporal-reasoning": "ref:p4",
    "knowledge-update": "ref:p5", "multi-session": "ref:p2",
}


def load_plan_bundles(run_dirs):
    """dir(s)/glob(s) -> {plan: {qid: bundle}}."""
    out = {}
    for pat in run_dirs:
        for d in sorted(glob.glob(pat)) or [pat]:
            bp = os.path.join(d, "bundles.jsonl")
            if not os.path.exists(bp):
                continue
            bundles = load_bundles(bp)
            if not bundles:
                continue
            plan = bundles[0].plan
            out[plan] = {b.question_id: b for b in bundles}
    return out


def paired_bootstrap(delta, B=10000, seed=0):
    """delta: per-question (oracle - bestfixed). Returns (mean, lo95, hi95)."""
    rng = random.Random(seed)
    n = len(delta)
    point = sum(delta) / n
    boots = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += delta[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * B)]
    hi = boots[int(0.975 * B)]
    return point, lo, hi


def load_correct_labels(labels_path):
    """V's answers.jsonl -> {(question_id, plan): 0/1} from judge.correct."""
    out = {}
    for a in load_answers(labels_path):
        out[(a.question_id, a.plan)] = int(bool((a.judge or {}).get("correct")))
    return out


def run(dataset, run_dirs, success_on, split, k, bootstrap=10000, seed=0,
        labels_path=None):
    questions = load_questions(os.path.join(dataset, "questions.jsonl"))
    if split:
        questions = [q for q in questions if q.split == split]
    qmap = {q.question_id: q for q in questions}
    plan_bundles = load_plan_bundles(run_dirs)
    plans = [p for p in LADDER if p in plan_bundles]
    assert plans, f"no fixed-plan runs found in {run_dirs}"

    # success signal: V's QA-correctness labels (if given) else evidence-level recall
    correct_map = load_correct_labels(labels_path) if labels_path else None

    # success matrix: qid -> plan -> 0/1
    succ = {}
    for q in questions:
        row = {}
        for p in plans:
            if correct_map is not None:
                # plan ids in answers.jsonl are the run plan ids (ref:pK)
                row[p] = correct_map.get((q.question_id, p), 0)
            else:
                b = plan_bundles[p].get(q.question_id)
                row[p] = success_fn(b.note_ids, q, success_on, k) if b else 0
        succ[q.question_id] = row

    qids = [q.question_id for q in questions]
    n = len(qids)
    per_plan_mean = {p: sum(succ[q][p] for q in qids) / n for p in plans}
    best_fixed = max(plans, key=lambda p: per_plan_mean[p])

    oracle_succ = {q: max(succ[q][p] for p in plans) for q in qids}
    any_of = sum(oracle_succ.values()) / n

    # oracle label (cheapest succeeding plan; else silver type-default)
    labels = {}
    silver = {}
    for q in questions:
        qid = q.question_id
        chosen = None
        for p in plans:  # LADDER order = cheapest first
            if succ[qid][p]:
                chosen = p
                break
        if chosen is None:
            chosen = A1_TYPE_DEFAULT.get(q.question_type, best_fixed)
            silver[qid] = True
        else:
            silver[qid] = False
        labels[qid] = chosen
    silver_fraction = sum(silver.values()) / n
    label_dist = Counter(labels.values())

    # delta + paired bootstrap
    delta = [oracle_succ[q] - succ[q][best_fixed] for q in qids]
    point, lo, hi = paired_bootstrap(delta, B=bootstrap, seed=seed)

    # per-type table
    per_type = {}
    by_type = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q.question_id)
    for t in QUESTION_TYPES:
        ids = by_type.get(t, [])
        if not ids:
            continue
        row = {p: round(sum(succ[i][p] for i in ids) / len(ids), 4) for p in plans}
        row["oracle"] = round(sum(oracle_succ[i] for i in ids) / len(ids), 4)
        row["n"] = len(ids)
        row["best_fixed_plan"] = max(plans, key=lambda p: row[p])
        per_type[t] = row

    result = {
        "dataset": dataset, "split": split or "all", "success_on": success_on, "k": k,
        "success_signal": "qa_correct" if correct_map is not None else "evidence",
        "labels_path": labels_path,
        "n_questions": n, "plans": plans,
        "per_plan_success": {p: round(per_plan_mean[p], 4) for p in plans},
        "best_fixed_plan": best_fixed,
        "best_fixed_success": round(per_plan_mean[best_fixed], 4),
        "any_of_plans_upper_bound": round(any_of, 4),
        "oracle_minus_best_fixed": round(point, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "oracle_label_distribution": {p: label_dist.get(p, 0) for p in plans},
        "silver_fraction": round(silver_fraction, 4),
        "per_type_success": per_type,
        "bootstrap_B": bootstrap,
    }
    return result


def _print(r):
    sig = r.get("success_signal", "evidence")
    sig_desc = f"signal=qa_correct (labels)" if sig == "qa_correct" \
        else f"signal=evidence success-on={r['success_on']} k={r['k']}"
    print(f"\n=== Oracle: {sig_desc} split={r['split']} "
          f"(n={r['n_questions']}) ===")
    for p in r["plans"]:
        star = "  <- best fixed" if p == r["best_fixed_plan"] else ""
        print(f"  {p}: {r['per_plan_success'][p]:.4f}{star}")
    print(f"  any-of-plans upper bound : {r['any_of_plans_upper_bound']:.4f}")
    print(f"  Oracle - best fixed      : {r['oracle_minus_best_fixed']:.4f} "
          f"  95% CI [{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]")
    print(f"  silver fraction          : {r['silver_fraction']:.4f}")
    print(f"  oracle labels            : {r['oracle_label_distribution']}")
    print("  per-type success:")
    hdr = "    type".ljust(30) + "".join(p.split(':')[1].rjust(8) for p in r["plans"]) \
        + "oracle".rjust(9) + "n".rjust(6)
    print(hdr)
    for t, row in r["per_type_success"].items():
        line = f"    {t}".ljust(30) + "".join(f"{row[p]:.3f}".rjust(8) for p in r["plans"])
        line += f"{row['oracle']:.3f}".rjust(9) + f"{row['n']}".rjust(6)
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/lme_s_k")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--success-on", choices=["session", "note"], default="session")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default=None,
                    help="V's answers.jsonl; success = judge.correct (replaces "
                         "evidence-level recall). --success-on/--k are then ignored.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    r = run(args.dataset, args.runs, args.success_on, args.split, args.k,
            args.bootstrap, args.seed, labels_path=args.labels)
    _print(r)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(r, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
