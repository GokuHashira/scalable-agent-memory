"""Where answer turns are lost: ranked too low, or ranked high but cut by the token cap when packing.

For each run (rankings.jsonl + bundles.jsonl) and every answer turn (gold_note_ids) of the selected questions:
  in_bundle     packed into the bundle
  cap_cut       ranked within the run's `max_results` (20; budget-sweep dirs: their budget.json) but not packed: the token cap filled first
  rank_miss     ranked below `max_results`
plus notes per bundle. Abstention questions and questions without turn labels are skipped.

  python scripts/cap_cut_analysis.py --runs runs/vref_bm25_lme_s runs/vref_dense_lme_s runs/vref_hybrid_lme_s \
      --data data/lme_s --split test --out runs/_tables/cap_cut_test
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.vschema import iter_jsonl  # noqa: E402
from scripts.reproduce_table import to_md  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-results", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    qs = {q["question_id"]: q for q in iter_jsonl(Path(a.data) / "questions.jsonl")}
    rows = [["run", "notes / bundle (mean, median)", "answer turns", "in bundle", "cap-cut (ranked within max_results)",
             "ranked below max_results", "questions with a cap-cut answer turn"]]
    js = {}
    for run in map(Path, a.runs):
        B = {b["question_id"]: b["note_ids"] for b in iter_jsonl(run / "bundles.jsonl")}
        rk, mr = run / "rankings.jsonl", a.max_results
        if not rk.exists() and (run / "budget.json").exists():  # budget-sweep dirs: ranking lives in the source run
            bj = json.loads((run / "budget.json").read_text())
            rk, mr = run.parent / bj["source"] / "rankings.jsonl", bj.get("max_results") or mr
        R = {r["question_id"]: r["note_ids"] for r in iter_jsonl(rk)}
        notes, c, nq, qcut = [], {"in_bundle": 0, "cap_cut": 0, "rank_miss": 0}, 0, 0
        for qid, q in qs.items():
            if (a.split != "all" and q["split"] != a.split) or qid.endswith("_abs") or not q["gold_note_ids"]:
                continue
            if qid not in B or qid not in R:
                continue
            nq += 1
            b, top = set(B[qid]), set(R[qid][:mr])
            notes.append(len(B[qid]))
            cut = False
            for g in q["gold_note_ids"]:
                k = "in_bundle" if g in b else "cap_cut" if g in top else "rank_miss"
                c[k] += 1
                cut |= k == "cap_cut"
            qcut += cut
        n = sum(c.values())
        js[run.name] = {**c, "n_answer_turns": n, "n_questions": nq, "questions_with_cap_cut": qcut,
                        "notes_mean": st.mean(notes), "notes_median": st.median(notes)}
        rows.append([run.name, f"{st.mean(notes):.1f}, {st.median(notes):g}", str(n),
                     f"{c['in_bundle']} ({100 * c['in_bundle'] / n:.1f}%)", f"{c['cap_cut']} ({100 * c['cap_cut'] / n:.1f}%)",
                     f"{c['rank_miss']} ({100 * c['rank_miss'] / n:.1f}%)", f"{qcut}/{nq}"])
    md = to_md(rows)
    Path(a.out + ".md").write_text(md)
    Path(a.out + ".json").write_text(json.dumps(js, indent=1))
    print(md)


if __name__ == "__main__":
    main()
