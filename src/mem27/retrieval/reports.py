"""Midterm report helpers: index cost table, determinism comparison, and failure-trace candidates (FR-018/019)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mem27.retrieval import ids
from mem27.retrieval.metrics import session_recall
from mem27.schema import Question, read_jsonl


def compare_bundle_files(a: str | Path, b: str | Path) -> dict[str, Any]:
    """Questions present in both files whose note ids or scores differ; timing fields are ignored."""
    left = {row["question_id"]: row for row in read_jsonl(a)}
    right = {row["question_id"]: row for row in read_jsonl(b)}
    shared = sorted(set(left) & set(right))
    mismatched = [
        qid
        for qid in shared
        if left[qid]["note_ids"] != right[qid]["note_ids"] or left[qid]["scores"] != right[qid]["scores"]
    ]
    return {"compared": len(shared), "identical": len(shared) - len(mismatched), "mismatched": mismatched}


def index_table(profile_path: str | Path, targets: Sequence[int] = (100_000, 1_000_000)) -> str:
    """Build cost per component at measured scale, with a stated linear projection in turn-note count."""
    profile = json.loads(Path(profile_path).read_text())
    n_turns = profile["n_turn_notes"]
    proj_cols = " | ".join(f"build @{t:,} | disk @{t:,}" for t in targets)
    lines = [
        f"| component | histories | build s | disk MB | peak RSS MB | {proj_cols} |",
        "|---" * (5 + 2 * len(targets)) + "|",
    ]
    for name, tot in profile["totals"].items():
        build_s, disk_mb = tot["build_ms"] / 1000.0, tot["disk_bytes"] / 1e6
        projections = " | ".join(f"{build_s * t / n_turns:.1f} s | {disk_mb * t / n_turns:.0f} MB" for t in targets)
        measured = f"| {name} | {tot['histories']} | {build_s:.1f} | {disk_mb:.1f} | {tot['peak_rss_mb']:.0f}"
        lines.append(f"{measured} | {projections} |")
    note = (
        f"\nMeasured on {n_turns:,} turn notes. Projection is linear in turn-note count (per-history flat indexes, "
        "no cross-history effects); it does not model approximate (HNSW) or Lucene indexes at 1M scale. "
        "Peak RSS is the process high-water mark, not a per-component delta.\n"
    )
    return "\n".join(lines) + "\n" + note


def failure_candidates(
    runs: dict[str, list[dict[str, Any]]], questions: list[Question], k: int = 20
) -> list[dict[str, Any]]:
    """Questions where one plan finds every gold session in its top k and another finds none."""
    by_run = {name: {b["question_id"]: b for b in bundles} for name, bundles in runs.items()}
    candidates = []
    for q in questions:
        if ids.is_abstention(q.question_id) or not q.gold_session_ids:
            continue
        recall = {
            name: session_recall(rows[q.question_id]["note_ids"], q, k)
            for name, rows in by_run.items()
            if q.question_id in rows
        }
        found_by = sorted(name for name, r in recall.items() if r == 1.0)
        if not found_by:
            continue
        for name in sorted(n for n, r in recall.items() if r == 0.0):
            retrieved = by_run[name][q.question_id]["note_ids"][:k]
            candidates.append(
                {
                    "question_id": q.question_id,
                    "question_type": q.question_type,
                    "question": q.question,
                    "answer": q.answer,
                    "gold_sessions": q.gold_session_ids,
                    "missed_by": name,
                    "found_by": found_by,
                    "missed_top_sessions": sorted({ids.session_key_of_note(n).split(":", 1)[1] for n in retrieved}),
                }
            )
    return candidates
