"""Tune a2 cascade tau_margin on the dev split; report test separately.

Grid tau_margin in {0.05,0.1,0.2,0.3} (planner.md section 3.5). Pick the tau that
maximizes dev session-success@k, tie-break on fewer mean tokens.

  python -m mem27.planner.cascade_tune --data data/lme_s_k --split dev
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, OrderedDict

import numpy as np

from mem27.schema import load_notes, load_questions
from mem27 import metrics as M
from mem27.planner import ref_retrievers
from mem27.planner.ref_retrievers import load_query_embeddings
from mem27.planner.planners import A2Cascade
from mem27.planner.run_plan import _assemble

GRID = [0.05, 0.1, 0.2, 0.3]


def eval_tau(tau, questions, by_hist, qemb, emb_dir, k, success_on):
    casc = A2Cascade(tau_margin=tau)
    succ, toks = [], []
    stop_hist, rungs_hist = Counter(), Counter()
    for q in questions:
        notes = by_hist[q.history_id]
        qvec = qemb.get(q.question_id, np.zeros(768, dtype=np.float32))
        idx = ref_retrievers.load_history_index(q.question_id, notes, emb_dir)
        dec = casc.decide(q.question, qvec, idx)
        note_ids, _, _, _ = _assemble(dec.final, notes)
        succ.append(M.success(note_ids, q, success_on, k))
        total = 0
        for cr in dec.cost_results:
            _, _, tk, _ = _assemble(cr, notes)
            total += tk
        toks.append(total)
        stop_hist[dec.trace["stop_reason"]] += 1
        rungs_hist[dec.trace["rungs_run"]] += 1
    n = len(succ)
    return {
        "tau": tau, "success": sum(succ) / n, "mean_tokens": sum(toks) / n,
        "stop_hist": dict(stop_hist), "rungs_hist": dict(rungs_hist), "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lme_s_k")
    ap.add_argument("--emb", default=None)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--success-on", choices=["session", "note"], default="session")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    emb_dir = args.emb or os.path.join(args.data, "emb")

    notes = load_notes(os.path.join(args.data, "notes.jsonl"))
    by_hist = OrderedDict()
    for n in notes:
        by_hist.setdefault(n.history_id, []).append(n)
    questions = [q for q in load_questions(os.path.join(args.data, "questions.jsonl"))
                 if (not args.split or q.split == args.split)]
    qemb = load_query_embeddings(emb_dir)

    results = [eval_tau(t, questions, by_hist, qemb, emb_dir, args.k, args.success_on)
               for t in GRID]
    # pick: max success, tie-break fewer tokens
    best = max(results, key=lambda r: (round(r["success"], 6), -r["mean_tokens"]))

    print(f"=== cascade tuning on {args.split} (n={results[0]['n']}, "
          f"success-on={args.success_on}@{args.k}) ===")
    print(f"  {'tau':>6} {'success':>9} {'mean_tok':>9}  stop_reasons")
    for r in results:
        mark = "  <- chosen" if r["tau"] == best["tau"] else ""
        print(f"  {r['tau']:>6} {r['success']:>9.4f} {r['mean_tokens']:>9.1f}  "
              f"{r['stop_hist']}{mark}")
    out = {"grid": results, "chosen_tau": best["tau"], "split": args.split,
           "success_on": args.success_on, "k": args.k}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}")
    print(f"CHOSEN_TAU={best['tau']}")
    return best["tau"]


if __name__ == "__main__":
    main()
