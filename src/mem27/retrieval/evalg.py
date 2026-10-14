"""G's local evidence metrics (research R10). V's canonical metrics replace these later on the same bundle format.

Recall is evidence-level only (no reader or judge). Abstention questions have no gold evidence, so they are
reported separately; questions without gold turns are skipped for turn recall and counted, never silently dropped.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from mem27.retrieval import ids
from mem27.schema import Note, Question, load_questions, read_jsonl

SESSION_KS = (5, 20)
TURN_K = 20


def percentile(values: Iterable[float], q: float) -> float:
    data = list(values)
    return float(np.percentile(data, q)) if data else float("nan")


def session_recall(ranked_note_ids: list[str], question: Question, k: int) -> float:
    """Share of gold sessions with at least one note in the top k."""
    gold = {ids.session_key(question.history_id, s) for s in question.gold_session_ids}
    hit = {ids.session_key_of_note(nid) for nid in ranked_note_ids[:k]}
    return len(gold & hit) / len(gold)


def turn_recall(ranked_note_ids: list[str], question: Question, k: int) -> float:
    """Share of gold (has_answer) turns in the top k."""
    gold = set(question.gold_note_ids)
    return len(gold & set(ranked_note_ids[:k])) / len(gold)


def _recall_block(rows: list[tuple[Question, list[str]]]) -> dict[str, Any]:
    with_sessions = [(q, r) for q, r in rows if q.gold_session_ids]
    with_turns = [(q, r) for q, r in rows if q.gold_note_ids]
    block: dict[str, Any] = {"n": len(rows)}
    for k in SESSION_KS:
        block[f"session_recall@{k}"] = _mean(session_recall(r, q, k) for q, r in with_sessions)
    block[f"turn_recall@{TURN_K}"] = _mean(turn_recall(r, q, TURN_K) for q, r in with_turns)
    return block


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return float(np.mean(data)) if data else float("nan")


def _latency(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    totals = [b["latency_ms"]["total"] for b in bundles]
    stage_names = sorted({name for b in bundles for name in b["latency_ms"]["stages"]})
    stages = {
        name: {
            "p50": percentile((b["latency_ms"]["stages"].get(name, 0.0) for b in bundles), 50),
            "p95": percentile((b["latency_ms"]["stages"].get(name, 0.0) for b in bundles), 95),
        }
        for name in stage_names
    }
    return {"total": {"p50": percentile(totals, 50), "p95": percentile(totals, 95)}, "stages": stages}


def evaluate(bundles: list[dict[str, Any]], questions: list[Question]) -> dict[str, Any]:
    """Metrics for one run: overall, per question type, abstention slice, token and latency summaries."""
    by_qid = {b["question_id"]: b for b in bundles}
    answered = [q for q in questions if q.question_id in by_qid]
    scored = [q for q in answered if not ids.is_abstention(q.question_id)]
    abstention = [q for q in answered if ids.is_abstention(q.question_id)]
    rows = [(q, by_qid[q.question_id]["note_ids"]) for q in scored]
    per_type: dict[str, list[tuple[Question, list[str]]]] = {}
    for q, ranked in rows:
        per_type.setdefault(q.question_type, []).append((q, ranked))
    present = [by_qid[q.question_id] for q in answered]
    return {
        "overall": _recall_block(rows),
        "per_type": {qtype: _recall_block(group) for qtype, group in sorted(per_type.items())},
        "abstention": {
            "n": len(abstention),
            "mean_tokens": _mean(by_qid[q.question_id]["total_tokens"] for q in abstention),
        },
        "counts": {
            "questions": len(questions),
            "abstention": len(abstention),
            "skipped_session_recall": sum(1 for q in scored if not q.gold_session_ids),
            "skipped_turn_recall": sum(1 for q in scored if not q.gold_note_ids),
            "missing_bundles": len(questions) - len(answered),
        },
        "tokens": {"mean": _mean(b["total_tokens"] for b in present), "max": max(b["total_tokens"] for b in present)}
        if present
        else {"mean": float("nan"), "max": 0},
        "latency": _latency(present) if present else {},
    }


def random_baseline(questions: list[Question], notes: list[Note], seed: int, k: int = TURN_K) -> dict[str, float]:
    """Recall of a seeded uniform-random ranking of each history's turn notes (the SC-003 floor)."""
    rng = random.Random(seed)
    turns_by_history: dict[str, list[str]] = {}
    for note in sorted(notes, key=lambda n: n.note_id):
        if not note.is_session_parent:
            turns_by_history.setdefault(note.history_id, []).append(note.note_id)
    rows = []
    for q in questions:
        if ids.is_abstention(q.question_id) or q.history_id not in turns_by_history:
            continue
        pool = turns_by_history[q.history_id]
        rows.append((q, rng.sample(pool, min(k, len(pool)))))
    return {
        f"session_recall@{k}": _mean(session_recall(r, q, k) for q, r in rows if q.gold_session_ids),
        f"turn_recall@{k}": _mean(turn_recall(r, q, k) for q, r in rows if q.gold_note_ids),
    }


def _fmt(value: float) -> str:
    return "—" if value != value else f"{value:.3f}"  # NaN check without importing math


def to_markdown_table(metrics_by_run: dict[str, dict[str, Any]]) -> str:
    """Overall retrieval table; every row names the run directory it came from."""
    header = "| run | n | S-R@5 | S-R@20 | T-R@20 | mean tokens | p50 ms | p95 ms |\n|---|---|---|---|---|---|---|---|"
    lines = [header]
    for run_id, m in sorted(metrics_by_run.items()):
        o, lat = m["overall"], m.get("latency", {}).get("total", {})
        lines.append(
            f"| {run_id} | {o['n']} | {_fmt(o['session_recall@5'])} | {_fmt(o['session_recall@20'])} | "
            f"{_fmt(o['turn_recall@20'])} | {m['tokens']['mean']:.1f} | {lat.get('p50', float('nan')):.1f} | "
            f"{lat.get('p95', float('nan')):.1f} |"
        )
    return "\n".join(lines) + "\n"


def to_markdown_per_type(metrics_by_run: dict[str, dict[str, Any]], metric: str = "session_recall@20") -> str:
    """One row per run, one column per question type, for a single metric."""
    types = sorted({t for m in metrics_by_run.values() for t in m["per_type"]})
    lines = [f"| run ({metric}) | " + " | ".join(types) + " |", "|---" * (len(types) + 1) + "|"]
    for run_id, m in sorted(metrics_by_run.items()):
        cells = [_fmt(m["per_type"].get(t, {}).get(metric, float("nan"))) for t in types]
        lines.append(f"| {run_id} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_reports(run_dirs: list[Path], questions_path: Path, table_path: Path) -> dict[str, dict[str, Any]]:
    """Evaluates each run dir (writing its metrics_g.json) and writes the combined markdown tables."""
    questions = load_questions(questions_path)
    metrics_by_run: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(run_dirs):
        bundles_path = run_dir / "bundles.jsonl"
        if not bundles_path.exists():
            raise FileNotFoundError(f"{bundles_path} missing; is {run_dir} a finished run directory?")
        metrics = evaluate(list(read_jsonl(bundles_path)), questions)
        (run_dir / "metrics_g.json").write_text(json.dumps(metrics, indent=2))
        metrics_by_run[run_dir.name] = metrics
    sections = [
        "# Retrieval table (local evidence metrics, G)\n",
        f"Questions: `{questions_path}`. Each row is a run directory under `runs/`.\n",
        to_markdown_table(metrics_by_run),
        "\n## Per question type: session recall@20\n",
        to_markdown_per_type(metrics_by_run, "session_recall@20"),
        "\n## Per question type: turn recall@20\n",
        to_markdown_per_type(metrics_by_run, "turn_recall@20"),
    ]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(sections))
    return metrics_by_run
