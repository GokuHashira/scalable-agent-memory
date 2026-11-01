"""Clean single-process latency benchmark (warm-up + per-question timing).

The parallel run_plan latencies are contended; this measures p50/p95 per plan with
one worker. Writes {plan: {qid: ms}} and a p50/p95 summary.

  python -m mem27.planner.timing --data data/lme_s_k --n 150 --out runs/timing.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict

import numpy as np

from mem27.planner.schema import load_notes, load_questions
from mem27 import registry
from mem27.metrics import percentile
from mem27.planner import ref_retrievers
from mem27.planner.ref_retrievers import load_query_embeddings
from mem27.planner.planners import PLANNER_FACTORIES, PlanDecision
from mem27.planner.run_plan import _assemble


def time_plan(plan, qtext, qvec, index, qtype, planners):
    t0 = time.perf_counter()
    if plan in PLANNER_FACTORIES:
        obj = planners[plan]
        dec = obj.decide(qtext, qvec, index, question_type=qtype) if plan == "a1" \
            else obj.decide(qtext, qvec, index)
        _assemble(dec.final, index.notes)
        for cr in dec.cost_results:
            _assemble(cr, index.notes)
    else:
        r = registry.get(plan).retrieve(qtext, qvec, index)
        _assemble(r, index.notes)
    return (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lme_s_k")
    ap.add_argument("--emb", default=None)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--a0-plan", default="ref:p2")
    ap.add_argument("--a2-tau", type=float, default=0.1)
    ap.add_argument("--out", default="runs/timing.json")
    args = ap.parse_args()
    emb_dir = args.emb or os.path.join(args.data, "emb")
    registry.ensure_ref_retrievers()

    notes = load_notes(os.path.join(args.data, "notes.jsonl"))
    by_hist = OrderedDict()
    for n in notes:
        by_hist.setdefault(n.history_id, []).append(n)
    questions = load_questions(os.path.join(args.data, "questions.jsonl"))[:args.n]
    qemb = load_query_embeddings(emb_dir)
    planners = {"a0": PLANNER_FACTORIES["a0"](default_plan=args.a0_plan),
                "a1": PLANNER_FACTORIES["a1"](),
                "a2": PLANNER_FACTORIES["a2"](tau_margin=args.a2_tau)}
    plans = registry.FIXED_PLANS + ["a0", "a1", "a2"]

    # Fresh index per (plan, question) so each plan's latency is standalone/comparable
    # (bm25 index build is attributed to every bm25-using plan, not just the first).
    def fresh(q):
        return ref_retrievers.load_history_index(q.question_id, by_hist[q.history_id], emb_dir)

    # warm-up (jit numpy, tiktoken, imports)
    for q in questions[:5]:
        for p in plans:
            time_plan(p, q.question, qemb.get(q.question_id, np.zeros(768, np.float32)),
                      fresh(q), q.question_type, planners)

    lat = {p: {} for p in plans}
    for q in questions:
        qvec = qemb.get(q.question_id, np.zeros(768, np.float32))
        for p in plans:
            lat[p][q.question_id] = time_plan(p, q.question, qvec, fresh(q),
                                              q.question_type, planners)

    summary = {}
    for p in plans:
        vals = list(lat[p].values())
        summary[p] = {"p50": round(percentile(vals, 50), 2),
                      "p95": round(percentile(vals, 95), 2),
                      "mean": round(sum(vals) / len(vals), 2), "n": len(vals)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"per_qid": lat, "summary": summary, "n": len(questions),
               "a0_plan": args.a0_plan, "a2_tau": args.a2_tau},
              open(args.out, "w"))
    print(f"=== latency (ms, single-process, n={len(questions)}) ===")
    for p in plans:
        print(f"  {p:>8}: p50={summary[p]['p50']:>7.2f}  p95={summary[p]['p95']:>7.2f}  "
              f"mean={summary[p]['mean']:>7.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
