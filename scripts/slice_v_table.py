"""Slice V accuracy table: per bundle, accuracy with a single-run percentile-bootstrap 95% CI (10K resamples, seed 0)
under the official judge and the second judge, mean bundle tokens, and the paired bootstrap delta vs the gold bundle.

  python scripts/slice_v_table.py --runs runs/gold_lme_s runs/random_lme_s runs/fullhist_lme_s_2k runs/fullhist_lme_s_16k \
      --ref runs/gold_lme_s --split dev --out runs/_tables/slice_v_dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.bootstrap import load_rows, paired_bootstrap, single_bootstrap  # noqa: E402
from mem27.vschema import iter_jsonl  # noqa: E402
from scripts.reproduce_table import to_md  # noqa: E402

LABEL = {"gold": "gold (answer turns only; reader upper bound)", "random": "random turns, 2,048 tok (floor)",
         "fullhist2k": "full history, most recent first, 2,048 tok", "fullhist16k": "full history, most recent first, 16,384 tok"}


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/lme_s")
    a = ap.parse_args(argv)
    ref = load_rows(a.ref)
    rows = [["bundle", "plan", "n", "accuracy [95% CI] (gpt-oss-20b)", "accuracy [95% CI] (Qwen3-32B)",
             "Δ vs gold [95% CI] (gpt-oss)", "mean tokens"]]
    js = {}
    drows = [["plan", "abstention correct", "answerable correct", "≥ 1 answer turn in bundle", "all answer turns in bundle",
              "acc. when ≥ 1 answer turn", "acc. when none"]]
    for run in a.runs:
        pq = {q: r for q, r in load_rows(run).items() if a.split == "all" or r["split"] == a.split}
        ids = sorted(pq)
        plan = next(iter(pq.values()))["plan"]
        off = single_bootstrap(np.array([pq[q]["correct"] for q in ids]))
        sec = single_bootstrap(np.array([pq[q]["correct[primary]"] for q in ids if pq[q].get("correct[primary]") is not None]))
        tok = float(np.mean([pq[q]["total_tokens"] for q in ids]))
        if Path(run).resolve() == Path(a.ref).resolve():
            d, dcell = None, "ref"
        else:
            both = [q for q in ids if q in ref]
            d = paired_bootstrap(np.array([pq[q]["correct"] for q in both]), np.array([ref[q]["correct"] for q in both]))
            dcell = f"{100 * d['delta']:+.1f} [{100 * d['ci_lo']:+.1f}, {100 * d['ci_hi']:+.1f}]"
        rows.append([LABEL.get(plan, plan), plan, str(off["n"]), f"{pct(off['mean'])} [{pct(off['ci_lo'])}, {pct(off['ci_hi'])}]",
                     f"{pct(sec['mean'])} [{pct(sec['ci_lo'])}, {pct(sec['ci_hi'])}]", dcell, f"{tok:,.0f}"])
        # decomposition: answerable vs abstention questions, and whether the bundle holds any answer turn
        qs = {q["question_id"]: q for q in iter_jsonl(Path(a.data) / "questions.jsonl")}
        B = {b["question_id"]: set(b["note_ids"]) for b in iter_jsonl(Path(run) / "bundles.jsonl")}
        ab = [q for q in ids if q.endswith("_abs")]
        an = [q for q in ids if not q.endswith("_abs")]
        lab = [q for q in an if qs[q]["gold_note_ids"]]
        has = [q for q in lab if B[q] & set(qs[q]["gold_note_ids"])]
        allg = [q for q in lab if set(qs[q]["gold_note_ids"]) <= B[q]]
        dec = {"n_abstention": len(ab), "abstention_correct": int(sum(pq[q]["correct"] for q in ab)),
               "n_answerable": len(an), "answerable_correct": int(sum(pq[q]["correct"] for q in an)),
               "answerable_acc": float(np.mean([pq[q]["correct"] for q in an])) if an else None,
               "n_labelled": len(lab), "any_answer_turn": len(has), "all_answer_turns": len(allg),
               "acc_given_any_answer_turn": float(np.mean([pq[q]["correct"] for q in has])) if has else None,
               "acc_given_no_answer_turn": float(np.mean([pq[q]["correct"] for q in lab if q not in has])) if len(has) < len(lab) else None}
        drows.append([plan, f"{dec['abstention_correct']} / {dec['n_abstention']}", f"{dec['answerable_correct']} / {dec['n_answerable']}",
                      f"{dec['any_answer_turn']} / {dec['n_labelled']}", f"{dec['all_answer_turns']} / {dec['n_labelled']}",
                      "–" if dec["acc_given_any_answer_turn"] is None else pct(dec["acc_given_any_answer_turn"]),
                      "–" if dec["acc_given_no_answer_turn"] is None else pct(dec["acc_given_no_answer_turn"])])
        js[plan] = {"run": Path(run).name, "split": a.split, "official": off, "second": sec, "vs_gold": d, "mean_tokens": tok,
                    "decomposition": dec}
    caption = (f"Accuracy on the LongMemEval_S {a.split} split. Reader: Qwen3-8B (GGUF Q8_0, llama.cpp, greedy, non-thinking) "
               "with the pinned chain-of-note prompt `configs/prompts/reader.txt`. Judges: LongMemEval judge prompts "
               "(`configs/prompts/judge.txt`); official label gpt-oss-20b (MXFP4, reasoning low), second judge Qwen3-32B "
               "(Q4_K_M, non-thinking). CIs: percentile bootstrap over questions, 10,000 resamples, seed 0; Δ is paired.")
    md = f"{caption}\n\n{to_md(rows)}"
    md2 = ("Where the accuracy comes from (same split; questions with labelled answer turns only for the coverage columns):\n\n"
           + to_md(drows))
    Path(a.out + "_decomp.md").write_text(md2)
    Path(a.out + ".md").write_text(md)
    Path(a.out + ".json").write_text(json.dumps(js, indent=1))
    print(md)


if __name__ == "__main__":
    main()
