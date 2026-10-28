"""Dev reader-prompt sweep re-scored per judge: is the pinned prompt choice robust to the judge?

Reads answers.<tag>.jsonl (judges[NAME].correct) for every run x tag and pivots accuracy on the dev split,
one table per judge, plus a per-type table pooled over runs. No LLM calls.

  python scripts/dev_sweep_by_judge.py --runs runs/gold_sess_lme_s runs/vref_bm25_lme_s runs/gold_turn_lme_s \
      runs/random_lme_s runs/none_lme_s --tags dev_plain dev_concise dev_con dev_plain_hint dev_concise_hint \
      dev_con_hint dev_con_rank --data data/lme_s --judges primary j2 --out runs/_tables/dev_prompt_sweep_by_judge
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from scripts.reproduce_table import SHORT, TYPES, to_md  # noqa: E402
from mem27.vschema import Question, iter_jsonl, read_jsonl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--judges", nargs="+", default=["primary", "j2"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    qs = {q.question_id: q for q in read_jsonl(Path(a.data) / "questions.jsonl", Question)}
    md, js = [], {}
    for judge in a.judges:
        rows = [["run"] + a.tags]
        pooled = defaultdict(lambda: defaultdict(list))  # tag -> type -> [0/1]
        cells = {}
        for run in a.runs:
            row = [Path(run).name]
            for tag in a.tags:
                f = Path(run) / f"answers.{tag}.jsonl"
                vals = []
                if f.exists():
                    for r in iter_jsonl(f):
                        q = qs.get(r["question_id"])
                        j = (r.get("judges") or {}).get(judge)
                        if q is None or q.split != a.split or j is None:
                            continue
                        vals.append(1.0 if j["correct"] else 0.0)
                        t = "abstention" if q.question_id.endswith("_abs") else q.question_type
                        pooled[tag][t].append(vals[-1])
                cells[(Path(run).name, tag)] = (sum(vals) / len(vals), len(vals)) if vals else None
                row.append(f"{100 * sum(vals) / len(vals):.1f}" if vals else "–")
            rows.append(row)
        md.append(f"### judge = {judge} ({a.split} split; n per cell = questions judged)\n\n{to_md(rows)}\n")
        trows = [["tag"] + [SHORT[t] for t in TYPES] + ["abst."]]
        for tag in a.tags:
            trows.append([tag] + [f"{100 * sum(v) / len(v):.1f}" if (v := pooled[tag].get(t)) else "–"
                                  for t in TYPES + ["abstention"]])
        md.append(f"Per type, pooled over runs (judge = {judge}):\n\n{to_md(trows)}\n")
        js[judge] = {f"{r}|{t}": c for (r, t), c in cells.items()}
    Path(a.out + ".md").write_text("\n".join(md))
    Path(a.out + ".json").write_text(json.dumps(js, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
