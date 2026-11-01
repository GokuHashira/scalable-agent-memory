# STAND-IN GENERATOR -- NOT V's real data. Replace the output with V's answers.jsonl.
"""Generate a STAND-IN answers.jsonl so the correct-based oracle (oracle.py
--labels) can be exercised end-to-end before V's real deliverable exists
(planner.md section 0: build your own stand-ins, don't wait on anyone).

For each fixed plan's bundle over the chosen split we synthesize one Answer row
(V's schema: {question_id, plan, answer, judge{correct, judge_model, raw}}).
``judge.correct`` is a seeded, deterministic function of retrieval quality:
correct tracks "a gold note landed in the bundle's top-5" with 90% fidelity
(10% of the time the label is flipped), so better retrieval -> more correct.

INTEGRATION (post-midterm): delete this and point oracle.py --labels at V's real
``answers.jsonl`` (and --dataset at V's data/lme_s). The file emitted here is a
stand-in ONLY; its docstring/first-line comment flag that.

    python -m mem27.planner.make_v_standin \
        --dataset data/lme_s_k --runs 'runs/ref_p*_lme_s_k_seed0_warm' \
        --split test --out runs/analysis/answers_standin.jsonl
"""
from __future__ import annotations

import argparse
import os
import random

from mem27.planner.schema import load_questions, Answer, write_jsonl
from mem27.oracle import load_plan_bundles
from mem27 import registry

TOP_K = 5          # "gold turn in top-5"
FIDELITY = 0.90    # P(label == top-5 hit indicator)


def build(dataset, run_dirs, split, seed=0):
    questions = load_questions(os.path.join(dataset, "questions.jsonl"))
    if split:
        questions = [q for q in questions if q.split == split]
    qmap = {q.question_id: q for q in questions}
    plan_bundles = load_plan_bundles(run_dirs)
    plans = [p for p in registry.FIXED_PLANS if p in plan_bundles]
    assert plans, f"no fixed-plan runs found in {run_dirs}"

    rng = random.Random(seed)
    rows = []
    for p in plans:
        for q in questions:  # deterministic order
            b = plan_bundles[p].get(q.question_id)
            gold = set(qmap[q.question_id].gold_note_ids)
            hit5 = bool(set((b.note_ids[:TOP_K] if b else [])) & gold)
            # correct == hit5 with FIDELITY, flipped otherwise
            correct = int(hit5 if rng.random() < FIDELITY else (not hit5))
            rows.append(Answer(
                question_id=q.question_id, plan=p,
                answer="[STAND-IN placeholder answer]",
                judge={"correct": correct, "judge_model": "standin", "raw": ""},
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/lme_s_k")
    ap.add_argument("--runs", nargs="+", default=["runs/ref_p*_lme_s_k_seed0_warm"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/analysis/answers_standin.jsonl")
    args = ap.parse_args()
    rows = build(args.dataset, args.runs, args.split, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = write_jsonl(args.out, rows)
    n_correct = sum(r.judge["correct"] for r in rows)
    print(f"wrote {n} stand-in answers -> {args.out} "
          f"({n_correct} correct, {n - n_correct} incorrect)")


if __name__ == "__main__":
    main()
