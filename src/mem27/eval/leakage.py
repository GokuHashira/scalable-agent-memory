"""Leakage checks (workstream V scope).

no_retrieval: accuracy of the reader with an empty bundle vs the expected chance accuracy
  (synthetic questions carry a per-question `chance`); a binomial test flags parametric answering.
summaries: for G's summaries.jsonl -- share of questions whose answer string appears in the summary
  of a NON-gold session of the same history (should be ~0; a high rate means summaries were written
  with knowledge of the questions), and, for context, share of gold-session summaries that keep it.

  python -m mem27.eval.leakage no-retrieval --run runs/none_synth_10k --data data/synth_10k --answers answers.concise.jsonl
  python -m mem27.eval.leakage summaries --data data/lme_s --summaries data/lme_s/summaries.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from mem27.eval.freshness import contains, synth_correct
from mem27.vschema import iter_jsonl


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Bin(n, p)."""
    if k <= 0:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def no_retrieval(run: Path, data: Path, answers: str) -> dict:
    qs = {q["question_id"]: q for q in iter_jsonl(data / "questions.jsonl")}
    rows = list(iter_jsonl(run / answers))
    n = len(rows)
    k_str = sum(synth_correct(r.get("answer", ""), qs[r["question_id"]]) for r in rows)
    k_judge = sum(bool((r.get("judge") or {}).get("correct")) for r in rows)
    chance = sum(qs[r["question_id"]].get("chance", 0.0) for r in rows) / max(1, n)
    by_t = defaultdict(lambda: [0, 0])
    for r in rows:
        q = qs[r["question_id"]]
        by_t[q["question_type"]][0] += synth_correct(r.get("answer", ""), q)
        by_t[q["question_type"]][1] += 1
    return {"n": n, "acc_string": k_str / n, "acc_judge": k_judge / n, "expected_chance": chance,
            "p_value_string_vs_chance": binom_sf(k_str, n, max(chance, 1e-6)),
            "by_type_string": {t: v[0] / v[1] for t, v in by_t.items()},
            "verdict": "leak-free" if binom_sf(k_str, n, max(chance, 1e-6)) > 0.01 or k_str / n < 0.05
            else "reader answers parametrically -- check entity names"}


def summaries(data: Path, summaries_path: Path) -> dict:
    qs = list(iter_jsonl(data / "questions.jsonl"))
    by_h = defaultdict(list)
    for s in iter_jsonl(summaries_path):
        by_h[s["history_id"]].append(s)
    leak, keep, n_gold, n = 0, 0, 0, 0
    examples = []
    for q in qs:
        if q["question_id"].endswith("_abs") or len(q["answer"]) < 3 or q["question_type"] == "single-session-preference":
            continue
        ss = by_h.get(q["history_id"], [])
        if not ss:
            continue
        n += 1
        gold = set(q["gold_session_ids"])
        hit_non = [s for s in ss if s["session_id"] not in gold and contains(s["text"], q["answer"])]
        if hit_non:
            leak += 1
            if len(examples) < 10:
                examples.append({"question_id": q["question_id"], "answer": q["answer"], "session_id": hit_non[0]["session_id"]})
        for s in ss:
            if s["session_id"] in gold:
                n_gold += 1
                keep += contains(s["text"], q["answer"])
    return {"n_questions": n, "answer_in_nongold_summary_rate": leak / max(1, n),
            "answer_kept_in_gold_summary_rate": keep / max(1, n_gold), "examples": examples,
            "note": "string match on short answers over-counts coincidental matches; inspect examples"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("no-retrieval")
    a1.add_argument("--run", required=True)
    a1.add_argument("--data", required=True)
    a1.add_argument("--answers", default="answers.jsonl")
    a2 = sub.add_parser("summaries")
    a2.add_argument("--data", required=True)
    a2.add_argument("--summaries", required=True)
    a = ap.parse_args(argv)
    res = no_retrieval(Path(a.run), Path(a.data), a.answers) if a.cmd == "no-retrieval" else \
        summaries(Path(a.data), Path(a.summaries))
    print(json.dumps(res, indent=2))
    out = Path(a.run if a.cmd == "no-retrieval" else a.data) / f"leakage_{a.cmd.replace('-', '_')}.json"
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
