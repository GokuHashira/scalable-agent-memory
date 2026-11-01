"""Train A3 on dev, route test, and report vs best-fixed and the oracle.

Builds one a3 Bundle per test question by selecting the predicted fixed plan's
existing bundle (copying note_ids/scores/tokens, plan="a3", trace.routed_to).
Writes runs/a3_lme_s_k_seed0_warm/bundles.jsonl and runs/analysis/a3_eval.json.

  python -m mem27.planner.eval_a3
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections import Counter, defaultdict

from mem27 import registry
from mem27.metrics import success as success_fn
from mem27.oracle import load_plan_bundles, paired_bootstrap
from mem27.planner.a3_router import A3Router, FEATURE_NAMES, features, make_labels
from mem27.planner.ref_retrievers import HistoryIndex, load_query_embeddings
from mem27.planner.schema import (QUESTION_TYPES, Bundle, load_notes, load_questions,
                          write_jsonl)

LADDER = registry.FIXED_PLANS
RUN_DIRS = [f"runs/ref_p{i}_lme_s_k_seed0_warm" for i in range(6)]


def build_indexes(notes):
    """history_id -> lightweight HistoryIndex (emb=None; we only need counts)."""
    by_hist = defaultdict(list)
    for n in notes:
        by_hist[n.history_id].append(n)
    return {hid: HistoryIndex(history_id=hid, notes=ns, emb=None)
            for hid, ns in by_hist.items()}


def build_feature_matrix(questions, plan_bundles, qemb, indexes):
    import numpy as np
    X, qids = [], []
    for q in questions:
        bbp = {p: plan_bundles[p].get(q.question_id) for p in LADDER
               if q.question_id in plan_bundles.get(p, {})}
        qvec = qemb.get(q.question_id)
        idx = indexes.get(q.history_id)
        X.append(features(q, bbp, qvec, idx))
        qids.append(q.question_id)
    return np.vstack(X), qids


def _metrics(qids, qmap, note_ids_by_qid, success_on, k):
    return sum(success_fn(note_ids_by_qid[qid], qmap[qid], success_on, k)
               for qid in qids) / len(qids)


def run(dataset="data/lme_s_k", success_on="note", k=5,
        bootstrap=10000, seed=0, out="runs/analysis/a3_eval.json",
        bundles_out="runs/a3_lme_s_k_seed0_warm/bundles.jsonl"):
    emb_dir = os.path.join(dataset, "emb")
    questions = load_questions(os.path.join(dataset, "questions.jsonl"))
    dev = [q for q in questions if q.split == "dev"]
    test = [q for q in questions if q.split == "test"]
    qmap = {q.question_id: q for q in questions}

    plan_bundles = load_plan_bundles(RUN_DIRS)
    plans = [p for p in LADDER if p in plan_bundles]
    assert plans, f"no fixed-plan runs found in {RUN_DIRS}"

    notes = load_notes(os.path.join(dataset, "notes.jsonl"))
    indexes = build_indexes(notes)
    qemb = load_query_embeddings(emb_dir)

    # ---- train on dev ----
    Xdev, dev_qids = build_feature_matrix(dev, plan_bundles, qemb, indexes)
    labels, best_overall = make_labels(dev, plan_bundles, success_on, k)
    ydev = [labels[qid] for qid in dev_qids]
    router = A3Router(classes=plans, seed=seed).fit(Xdev, ydev)

    # ---- predict on test ----
    Xtest, test_qids = build_feature_matrix(test, plan_bundles, qemb, indexes)
    pred = dict(zip(test_qids, router.predict(Xtest)))

    # ---- a3 bundles (select the routed plan's bundle) ----
    a3_bundles = []
    note_ids_a3 = {}
    for qid in test_qids:
        routed = pred[qid]
        src = plan_bundles[routed][qid]
        b = Bundle(
            question_id=qid, plan="a3",
            note_ids=list(src.note_ids), passages=list(src.passages),
            scores=list(src.scores), total_tokens=src.total_tokens,
            latency_ms=dict(src.latency_ms), stale_flags=list(src.stale_flags),
            trace={"planner": "a3", "routed_to": routed},
        )
        a3_bundles.append(b)
        note_ids_a3[qid] = b.note_ids
    os.makedirs(os.path.dirname(bundles_out), exist_ok=True)
    n_written = write_jsonl(bundles_out, a3_bundles)

    # ---- metrics on test ----
    per_plan = {p: {} for p in plans}
    note_ids_plan = {p: {qid: plan_bundles[p][qid].note_ids for qid in test_qids}
                     for p in plans}
    metric_specs = [("turn_hit@5", "note", 5), ("turn_hit@20", "note", 20),
                    ("session_hit@5", "session", 5)]
    headline = ("turn_hit@5", success_on, k)

    def plan_metric(nids, son, kk):
        return _metrics(test_qids, qmap, nids, son, kk)

    results_a3 = {name: round(plan_metric(note_ids_a3, son, kk), 4)
                  for name, son, kk in metric_specs}
    results_fixed = {p: {name: round(plan_metric(note_ids_plan[p], son, kk), 4)
                         for name, son, kk in metric_specs} for p in plans}
    best_fixed = max(plans, key=lambda p: results_fixed[p]["turn_hit@5"])

    # oracle upper bound per metric (max over fixed plans per question)
    oracle = {}
    for name, son, kk in metric_specs:
        oracle[name] = round(sum(
            max(success_fn(plan_bundles[p][qid].note_ids, qmap[qid], son, kk)
                for p in plans) for qid in test_qids) / len(test_qids), 4)

    # ---- paired bootstrap CI: (a3 - best_fixed) turn-hit@5 ----
    delta = [success_fn(note_ids_a3[qid], qmap[qid], success_on, k)
             - success_fn(note_ids_plan[best_fixed][qid], qmap[qid], success_on, k)
             for qid in test_qids]
    point, lo, hi = paired_bootstrap(delta, B=bootstrap, seed=seed)

    routed_dist = Counter(pred[qid] for qid in test_qids)
    label_dist = Counter(ydev)

    # ---- per-type turn-hit@5: a3 vs best-fixed vs oracle ----
    by_type = defaultdict(list)
    for q in test:
        by_type[q.question_type].append(q.question_id)
    per_type = {}
    for t in QUESTION_TYPES:
        ids = by_type.get(t, [])
        if not ids:
            continue
        a3_v = sum(success_fn(note_ids_a3[i], qmap[i], success_on, k) for i in ids) / len(ids)
        bf_v = sum(success_fn(note_ids_plan[best_fixed][i], qmap[i], success_on, k) for i in ids) / len(ids)
        or_v = sum(max(success_fn(plan_bundles[p][i].note_ids, qmap[i], success_on, k)
                       for p in plans) for i in ids) / len(ids)
        per_type[t] = {"n": len(ids), "a3": round(a3_v, 4),
                       "best_fixed": round(bf_v, 4), "oracle": round(or_v, 4)}

    result = {
        "dataset": dataset, "split": "test", "n_test": len(test_qids),
        "n_dev": len(dev_qids), "success_on": success_on, "k": k,
        "plans": plans, "feature_names": FEATURE_NAMES,
        "a3": results_a3,
        "per_fixed_plan": results_fixed,
        "best_fixed_plan": best_fixed,
        "best_fixed": results_fixed[best_fixed],
        "oracle_upper_bound": oracle,
        "a3_minus_best_fixed_turn_hit@5": round(point, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "bootstrap_B": bootstrap,
        "routed_plan_distribution": {p: routed_dist.get(p, 0) for p in plans},
        "dev_label_distribution": {p: label_dist.get(p, 0) for p in plans},
        "best_overall_dev_plan": best_overall,
        "per_type_turn_hit@5": per_type,
        "bundles_path": bundles_out, "n_bundles": n_written,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    return result


def _print(r):
    bf = r["best_fixed_plan"]
    print(f"\n=== A3 router: success-on={r['success_on']} k={r['k']} split=test "
          f"(n={r['n_test']}, trained on {r['n_dev']} dev) ===")
    print("  metric        a3     best-fixed(%s)  oracle-UB" % bf)
    for name in ("turn_hit@5", "turn_hit@20", "session_hit@5"):
        print(f"  {name:<13} {r['a3'][name]:.4f}   {r['best_fixed'][name]:.4f}"
              f"        {r['oracle_upper_bound'][name]:.4f}")
    d = r["a3_minus_best_fixed_turn_hit@5"]
    print(f"\n  a3 - best-fixed turn-hit@5 : {d:+.4f}  "
          f"95% CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
    beats = "YES" if r["a3"]["turn_hit@5"] > r["best_fixed"]["turn_hit@5"] else "no"
    print(f"  beats best fixed plan?     : {beats}")
    print(f"  routed-plan distribution   : {r['routed_plan_distribution']}")
    print(f"  dev-label distribution     : {r['dev_label_distribution']} "
          f"(best-overall-dev={r['best_overall_dev_plan']})")
    print("  per-type turn-hit@5 (a3 / best-fixed / oracle):")
    for t, row in r["per_type_turn_hit@5"].items():
        print(f"    {t:<28} n={row['n']:<4} "
              f"{row['a3']:.3f} / {row['best_fixed']:.3f} / {row['oracle']:.3f}")
    print(f"\n  wrote {r['bundles_path']} ({r['n_bundles']} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/lme_s_k")
    ap.add_argument("--success-on", choices=["note", "session"], default="note")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/analysis/a3_eval.json")
    args = ap.parse_args()
    r = run(args.dataset, args.success_on, args.k, args.bootstrap, args.seed, args.out)
    _print(r)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
