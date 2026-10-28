"""Compact report tables from paired-bootstrap CSVs (mem27.eval.bootstrap --out X -> X.csv).

  ci     one row per run: A, B, delta [95% CI], p, verdict for one metric (group ALL by default)
  power  one row per run: n, discordant share, s_d, MDE80, n needed for +-2-pt equivalence (metric `correct`)
  types  run x question-type matrix of "delta [CI]" for one metric

  python scripts/report_tables.py ci --csv runs/_tables/bootstrap_vs_vref_hybrid.csv --metric correct \
      --out runs/_tables/report_ci_vs_hybrid.md
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from scripts.reproduce_table import SHORT, TYPES, to_md  # noqa: E402

RATE = ("correct", "sess_", "turn_", "ctx_", "ndcg", "mrr")


def f(metric: str, x: str | float, signed: bool = False) -> str:
    v = float(x)
    if metric.startswith(RATE):
        v *= 100
    return f"{v:+.1f}" if signed else f"{v:.1f}"


def rows_of(path: str, metric: str, group: str = "ALL", order: list[str] | None = None) -> list[dict]:
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r["metric"] == metric and r["group"] == group]
    if order:
        rank = {n: i for i, n in enumerate(order)}
        rows.sort(key=lambda r: rank.get(r["run"], len(rank)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["ci", "power", "types"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--metric", default="correct")
    ap.add_argument("--group", default="ALL")
    ap.add_argument("--order", nargs="*", default=None, help="run names in display order")
    ap.add_argument("--label", nargs="*", default=None, help="run=label pairs for display")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    label = dict(x.split("=", 1) for x in (a.label or []))
    m = a.metric
    if a.kind == "ci":
        out = [["run", "A", "ref", "Δ [95% CI]", "p", "verdict (±2 pt)"]]
        for r in rows_of(a.csv, m, a.group, a.order):
            out.append([label.get(r["run"], r["run"]), f(m, r["mean_a"]), f(m, r["mean_b"]),
                        f"{f(m, r['delta'], True)} [{f(m, r['ci_lo'], True)}, {f(m, r['ci_hi'], True)}]",
                        f"{float(r['p']):.3f}", r["verdict"]])
        title = f"{m}, group {a.group}, vs {rows_of(a.csv, m, a.group)[0]['ref'] if out[1:] else '?'}"
    elif a.kind == "power":
        out = [["run", "ref", "n", "discordant", "s_d", "MDE80 (pts)", "n for ±2-pt equivalence"]]
        for r in rows_of(a.csv, m, a.group, a.order):
            n = int(r["n"])
            out.append([label.get(r["run"], r["run"]), r["ref"], str(n), f"{100 * int(r['discordant']) / n:.1f}%",
                        f"{float(r['sd_delta']):.3f}", f"{100 * float(r['mde80']):.1f}",
                        f"{int(float(r['n_for_equiv'])):,}" if r.get("n_for_equiv") not in (None, "", "None") else "–"])
        title = f"power for {m}"
    else:
        with open(a.csv) as fh:
            allr = [r for r in csv.DictReader(fh) if r["metric"] == m]
        runs = a.order or sorted({r["run"] for r in allr})
        out = [["run"] + [SHORT[t] for t in TYPES] + ["ALL"]]
        cell = {(r["run"], r["group"]): r for r in allr}
        for run in runs:
            row = [label.get(run, run)]
            for g in TYPES + ["ALL"]:
                r = cell.get((run, g))
                row.append("–" if r is None else
                           f"{f(m, r['delta'], True)} [{f(m, r['ci_lo'], True)}, {f(m, r['ci_hi'], True)}]")
            out.append(row)
        title = f"{m} delta by type"
    md = to_md(out)
    Path(a.out).write_text(md)
    print(f"<!-- {title} -->\n{md}")


if __name__ == "__main__":
    main()
