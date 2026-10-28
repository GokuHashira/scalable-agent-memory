"""Regenerate report tables from runs/*/metrics*.json (EVAL-04). Every number in the report should
come from here.

  python scripts/reproduce_table.py main   --runs runs/gold_turn_lme_s ... --ref runs/vref_hybrid_lme_s --split test
  python scripts/reproduce_table.py types  --runs ... --split test
  python scripts/reproduce_table.py budget --runs runs/vref_hybrid_lme_s_b512 ... --split test
Writes markdown to stdout and to runs/_tables/<name>.md (and .csv).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.bootstrap import paired_bootstrap  # noqa: E402

TYPES = ["single-session-user", "single-session-assistant", "single-session-preference", "multi-session",
         "temporal-reasoning", "knowledge-update"]
SHORT = {"single-session-user": "SS-user", "single-session-assistant": "SS-asst", "single-session-preference": "SS-pref",
         "multi-session": "multi", "temporal-reasoning": "temporal", "knowledge-update": "KU"}


def load(run: str, name: str) -> dict:
    return json.loads((Path(run) / name).read_text())


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def ci_cell(rows_a, rows_b, metric, split):
    qa = {r["question_id"]: r for r in rows_a if split == "all" or r["split"] == split}
    qb = {r["question_id"]: r for r in rows_b if split == "all" or r["split"] == split}
    ids = sorted(q for q in qa.keys() & qb.keys() if qa[q].get(metric) is not None and qb[q].get(metric) is not None)
    if not ids:
        return "–", None
    res = paired_bootstrap(np.array([qa[q][metric] for q in ids]), np.array([qb[q][metric] for q in ids]))
    return f"{100 * res['delta']:+.1f} [{100 * res['ci_lo']:+.1f}, {100 * res['ci_hi']:+.1f}]", res


def table_main(a) -> list[list[str]]:
    ref = load(a.ref, a.metrics_name) if a.ref else None
    hdr = ["run", "n", "acc", "Δacc vs ref [95% CI]", "task-avg", "abst.", "sR@5", "sR@20", "tR@20", "prec",
           "acc (all gold)", "acc (gold miss)", "tokens", "p50 ms", "p95 ms"]
    out = [hdr]
    for run in a.runs:
        m = load(run, a.metrics_name)
        g = m["aggregate"].get(a.split, {})
        delta = "ref" if ref and Path(run).resolve() == Path(a.ref).resolve() else (
            ci_cell(m["per_question"], ref["per_question"], "correct", a.split)[0] if ref else "–")
        out.append([m["run_id"], str(g.get("n", 0)), pct(g.get("accuracy")), delta, pct(g.get("accuracy_task_avg")),
                    pct(g.get("accuracy_abstention")), pct(g.get("sess_recall@5")), pct(g.get("sess_recall@20")),
                    pct(g.get("turn_recall@20")), pct(g.get("ctx_precision")), pct(g.get("acc_given_all_gold")),
                    pct(g.get("acc_given_gold_missing")), f"{g.get('tokens_mean', 0):.0f}",
                    f"{g.get('latency_p50') or 0:.1f}", f"{g.get('latency_p95') or 0:.1f}"])
    return out


def table_types(a) -> list[list[str]]:
    hdr = ["run"] + [SHORT[t] for t in TYPES] + ["overall", "task-avg"]
    out = [hdr]
    for run in a.runs:
        m = load(run, a.metrics_name)
        pt = m["per_type"].get(a.split, {})
        g = m["aggregate"].get(a.split, {})
        out.append([m["run_id"]] + [pct(pt.get(t, {}).get(a.metric)) for t in TYPES] +
                   [pct(g.get(a.metric)), pct(g.get("accuracy_task_avg")) if a.metric == "accuracy" else "–"])
    return out


def table_budget(a) -> list[list[str]]:
    out = [["run", "cap", "tokens", "sR@20", "tR@20", "acc", "task-avg"]]
    for run in a.runs:
        m = load(run, a.metrics_name)
        g = m["aggregate"].get(a.split, {})
        cap = json.loads((Path(run) / "budget.json").read_text())["cap"] if (Path(run) / "budget.json").exists() else 2048
        out.append([m["run_id"], str(cap), f"{g.get('tokens_mean', 0):.0f}", pct(g.get("sess_recall@20")),
                    pct(g.get("turn_recall@20")), pct(g.get("accuracy")), pct(g.get("accuracy_task_avg"))])
    return out


def to_md(rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])] +
                     ["| " + " | ".join(r) + " |" for r in rows[1:]]) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("table", choices=["main", "types", "budget"])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--metric", default="accuracy")
    ap.add_argument("--metrics-name", default="metrics.json")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default="runs/_tables")
    a = ap.parse_args(argv)
    rows = {"main": table_main, "types": table_types, "budget": table_budget}[a.table](a)
    md = to_md(rows)
    name = a.name or f"{a.table}_{a.split}"
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.out_dir) / f"{name}.md").write_text(md)
    with open(Path(a.out_dir) / f"{name}.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(md)


if __name__ == "__main__":
    main()
