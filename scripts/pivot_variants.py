"""Pivot reader-variant results: rows = runs, columns = variant tags (metrics.<tag>.json).

  python scripts/pivot_variants.py --runs runs/gold_sess_lme_s runs/vref_bm25_lme_s --tags dev_concise dev_con \
      --split dev [--by-type] [--metric accuracy] [--out runs/_tables/dev_prompt_sweep]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from scripts.reproduce_table import SHORT, TYPES, to_md  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--metric", default="accuracy")
    ap.add_argument("--by-type", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = [["run"] + a.tags]
    by_type = [["tag"] + [SHORT[t] for t in TYPES] + ["abst."]]
    mean = {t: [] for t in a.tags}
    for run in a.runs:
        r = [Path(run).name]
        for t in a.tags:
            f = Path(run) / f"metrics.{t}.json"
            if not f.exists():
                r.append("–")
                continue
            m = json.loads(f.read_text())
            v = m["aggregate"].get(a.split, {}).get(a.metric)
            r.append("–" if v is None else f"{100 * v:.1f}")
            if v is not None:
                mean[t].append(v)
        rows.append(r)
    rows.append(["mean over runs"] + [f"{100 * sum(v) / len(v):.1f}" if v else "–" for v in mean.values()])
    md = to_md(rows)
    if a.by_type:
        for t in a.tags:
            agg = {k: [] for k in TYPES + ["abst"]}
            for run in a.runs:
                f = Path(run) / f"metrics.{t}.json"
                if not f.exists():
                    continue
                for q in json.loads(f.read_text())["per_question"]:
                    if q["split"] != a.split and a.split != "all" or "correct" not in q:
                        continue
                    agg[q["question_type"]].append(q["correct"])
                    if q["abstention"]:
                        agg["abst"].append(q["correct"])
            by_type.append([t] + [f"{100 * sum(v) / len(v):.1f}" if v else "–" for v in agg.values()])
        md += "\nPer type (pooled over runs):\n\n" + to_md(by_type)
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(f"{a.out}.md").write_text(md)


if __name__ == "__main__":
    main()
