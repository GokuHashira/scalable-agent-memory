"""Retrieval metrics on the frozen 400-question test split, with bootstrap CIs and paired comparisons.

The split comes from the evaluation conversion (data/lme_s/questions.jsonl, stratified 100 dev / 400 test, seed 0)
and is matched to the retrieval runs by question id. Gold labels come from the questions file the runs were
built on. Existing run directories are only read; results go to a new markdown table and a JSON file.

  uv run python scripts/evaluate_test_split.py --runs 'runs/*-lme_s_raw-seed0-warm' \
      --questions data/lme_s_raw/questions.jsonl --split-from data/lme_s/questions.jsonl \
      --ref dense-lme_s_raw-seed0-warm --out docs/midterm/retrieval_test_split
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np

from mem27.eval.bootstrap import paired_bootstrap, single_bootstrap
from mem27.retrieval import ids
from mem27.retrieval.metrics import percentile, session_recall, turn_recall
from mem27.schema import Question, load_questions, read_jsonl

METRICS = ("session_recall@5", "session_recall@20", "turn_recall@20")


def load_split(path: Path) -> dict[str, str]:
    return {row["question_id"]: row["split"] for row in read_jsonl(path)}


def per_question(bundles: dict[str, dict[str, Any]], questions: list[Question]) -> dict[str, dict[str, float]]:
    """One row per scored question: recall values (NaN when the question has no gold of that kind)."""
    rows: dict[str, dict[str, float]] = {}
    for q in questions:
        ranked = bundles[q.question_id]["note_ids"]
        rows[q.question_id] = {
            "session_recall@5": session_recall(ranked, q, 5) if q.gold_session_ids else float("nan"),
            "session_recall@20": session_recall(ranked, q, 20) if q.gold_session_ids else float("nan"),
            "turn_recall@20": turn_recall(ranked, q, 20) if q.gold_note_ids else float("nan"),
        }
    return rows


def column(rows: dict[str, dict[str, float]], qids: list[str], metric: str) -> np.ndarray:
    return np.array([rows[q][metric] for q in qids], float)


def summarize(run_dirs: list[Path], questions: list[Question], ref: str) -> dict[str, Any]:
    by_type: dict[str, list[str]] = {}
    for q in questions:
        by_type.setdefault(q.question_type, []).append(q.question_id)
    all_qids = [q.question_id for q in questions]
    runs: dict[str, Any] = {}
    rows_by_run: dict[str, dict[str, dict[str, float]]] = {}
    for run_dir in run_dirs:
        bundles = {b["question_id"]: b for b in read_jsonl(run_dir / "bundles.jsonl")}
        missing = [q for q in all_qids if q not in bundles]
        if missing:
            raise SystemExit(f"{run_dir.name}: {len(missing)} test questions have no bundle")
        rows = per_question(bundles, questions)
        rows_by_run[run_dir.name] = rows
        present = [bundles[q] for q in all_qids]
        overall = {}
        for metric in METRICS:
            values = column(rows, all_qids, metric)
            overall[metric] = single_bootstrap(values[~np.isnan(values)])
        per_type = {}
        for qtype, qids in sorted(by_type.items()):
            per_type[qtype] = {}
            for metric in METRICS:
                values = column(rows, qids, metric)
                values = values[~np.isnan(values)]
                per_type[qtype][metric] = {"n": len(values), "mean": float(values.mean()) if len(values) else None}
        runs[run_dir.name] = {
            "overall": overall,
            "per_type": per_type,
            "tokens_mean": float(np.mean([b["total_tokens"] for b in present])),
            "latency_ms": {
                "p50": percentile((b["latency_ms"]["total"] for b in present), 50),
                "p95": percentile((b["latency_ms"]["total"] for b in present), 95),
            },
        }
    if ref not in rows_by_run:
        raise SystemExit(f"reference run {ref} not among {sorted(rows_by_run)}")
    paired: dict[str, dict[str, Any]] = {}
    for name, rows in rows_by_run.items():
        if name == ref:
            continue
        paired[name] = {}
        for metric in METRICS:
            a, b = column(rows, all_qids, metric), column(rows_by_run[ref], all_qids, metric)
            keep = ~np.isnan(a) & ~np.isnan(b)
            paired[name][metric] = paired_bootstrap(a[keep], b[keep])
    return {"runs": runs, "paired_vs_ref": paired, "ref": ref}


def method(run_name: str) -> str:
    return run_name.split("-", 1)[0]


def to_markdown(result: dict[str, Any], counts: dict[str, int], sources: dict[str, str]) -> str:
    runs, ref = result["runs"], result["ref"]
    out = [
        "# Retrieval on the 400-question test split\n",
        f"Questions: the test split of `{sources['split_from']}` (stratified by question type and abstention, seed 0), "
        f"matched by question id to the runs built on `{sources['questions']}`. Abstention questions have no gold "
        f"evidence and are excluded from recall.\n",
        f"Scored test questions: {counts['scored']} (of {counts['test']}; {counts['abstention']} abstention). "
        f"Session recall uses {counts['with_sessions']} questions with gold sessions; turn recall uses "
        f"{counts['with_turns']} with gold turns.\n",
        "Values are means with 95% bootstrap CIs (10,000 resamples of questions, seed 0).\n",
        "| method | S-R@5 | S-R@20 | T-R@20 | mean tokens | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]

    def cell(s: dict[str, Any]) -> str:
        return f"{s['mean']:.3f} [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"

    for name, r in sorted(runs.items(), key=lambda kv: -kv[1]["overall"]["session_recall@20"]["mean"]):
        o = r["overall"]
        out.append(
            f"| {method(name)} | {cell(o['session_recall@5'])} | {cell(o['session_recall@20'])} | "
            f"{cell(o['turn_recall@20'])} | {r['tokens_mean']:.0f} | {r['latency_ms']['p50']:.1f} | "
            f"{r['latency_ms']['p95']:.1f} |"
        )
    out += [
        f"\n## Paired differences against {method(ref)}\n",
        "Each cell is the mean per-question difference (method minus reference) with its 95% CI. Verdicts use the "
        "pre-registered ±0.02 margin: *equivalent* when the CI lies inside ±0.02, *better*/*worse* when it excludes 0, "
        "otherwise *inconclusive*.\n",
        "| method | ΔS-R@5 | ΔS-R@20 | ΔT-R@20 |",
        "|---|---|---|---|",
    ]
    for name, metrics in sorted(result["paired_vs_ref"].items()):
        cells = [f"{m['delta']:+.3f} [{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}] {m['verdict']}" for m in metrics.values()]
        out.append(f"| {method(name)} | " + " | ".join(cells) + " |")
    types = sorted({t for r in runs.values() for t in r["per_type"]})
    for metric in ("session_recall@20", "turn_recall@20"):
        n_by_type = {t: next(iter(runs.values()))["per_type"][t][metric]["n"] for t in types}
        out += [
            f"\n## Per question type: {metric}\n",
            "| method | " + " | ".join(f"{t} (n={n_by_type[t]})" for t in types) + " |",
            "|---" * (len(types) + 1) + "|",
        ]
        for name, r in sorted(runs.items()):
            vals = [r["per_type"][t][metric]["mean"] for t in types]
            out.append(f"| {method(name)} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in vals) + " |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--questions", required=True, help="questions file the runs were built on (gold labels)")
    parser.add_argument("--split-from", required=True, help="questions file holding the frozen dev/test split")
    parser.add_argument("--split", default="test")
    parser.add_argument("--ref", required=True, help="reference run directory name for paired comparisons")
    parser.add_argument("--out", required=True, help="output prefix; writes <out>.md and <out>.json")
    args = parser.parse_args(argv)

    run_dirs = sorted({Path(p) for pattern in args.runs for p in glob.glob(pattern) if Path(p).is_dir()})
    if not run_dirs:
        raise SystemExit(f"no run directories match {args.runs}")
    split = load_split(Path(args.split_from))
    all_questions = load_questions(Path(args.questions))
    unmatched = [q.question_id for q in all_questions if q.question_id not in split]
    if unmatched:
        raise SystemExit(f"{len(unmatched)} questions have no split in {args.split_from}, e.g. {unmatched[:3]}")
    in_split = [q for q in all_questions if split[q.question_id] == args.split]
    scored = [q for q in in_split if not ids.is_abstention(q.question_id)]
    counts = {
        "test": len(in_split),
        "abstention": len(in_split) - len(scored),
        "scored": len(scored),
        "with_sessions": sum(bool(q.gold_session_ids) for q in scored),
        "with_turns": sum(bool(q.gold_note_ids) for q in scored),
    }
    result = summarize(run_dirs, scored, args.ref)
    result["counts"] = counts
    sources = {"questions": args.questions, "split_from": args.split_from}
    result["sources"] = sources
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    out.with_suffix(".md").write_text(to_markdown(result, counts, sources))
    print(json.dumps(counts), f"-> {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
