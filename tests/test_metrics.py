"""Local evidence metrics on hand-computed examples (FR-015)."""

import pytest

from mem27.retrieval.metrics import (
    evaluate,
    percentile,
    random_baseline,
    session_recall,
    to_markdown_table,
    turn_recall,
)
from mem27.schema import Note, Question


def _q(qid, qtype, gold_sessions, gold_notes):
    return Question(qid, qid, qtype, "q?", "2023-01-01", "a", gold_sessions, gold_notes)


def _bundle(qid, note_ids, tokens=10, total=5.0, stages=None):
    return {
        "question_id": qid,
        "plan": "p0",
        "note_ids": note_ids,
        "total_tokens": tokens,
        "latency_ms": {"total": total, "stages": stages or {"bm25": total}},
    }


def test_session_and_turn_recall():
    q = _q("h", "multi-session", ["s1", "s2"], ["h:s1:3", "h:s2:0"])
    ranked = ["h:s1:0", "h:s3:1", "h:s2:0"]
    assert session_recall(ranked, q, k=1) == 0.5
    assert session_recall(ranked, q, k=5) == 1.0
    assert turn_recall(ranked, q, k=20) == 0.5


def test_percentile_linear():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 50) == pytest.approx(5.5)
    assert percentile(values, 95) == pytest.approx(9.55)


def test_evaluate_overall_per_type_and_exclusions():
    questions = [
        _q("a", "single-session-user", ["s1"], ["a:s1:0"]),
        _q("b", "single-session-user", ["s1"], []),  # no gold turns: turn recall skipped, counted
        _q("c", "multi-session", ["s1", "s2"], ["c:s1:0", "c:s2:0"]),
        _q("d_abs", "temporal-reasoning", [], []),  # abstention: excluded from recall
    ]
    bundles = [
        _bundle("a", ["a:s1:0"], tokens=10, total=1.0),
        _bundle("b", ["b:s2:0"], tokens=20, total=2.0),
        _bundle("c", ["c:s1:0", "c:s9:0"], tokens=30, total=3.0),
        _bundle("d_abs", ["d_abs:s1:0"], tokens=40, total=4.0),
    ]
    m = evaluate(bundles, questions)
    assert m["overall"]["n"] == 3
    assert m["overall"]["session_recall@20"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert m["overall"]["turn_recall@20"] == pytest.approx((1.0 + 0.5) / 2)
    assert m["counts"] == {
        "questions": 4,
        "abstention": 1,
        "skipped_session_recall": 0,
        "skipped_turn_recall": 1,
        "missing_bundles": 0,
    }
    assert m["per_type"]["single-session-user"]["n"] == 2
    assert m["per_type"]["multi-session"]["session_recall@5"] == pytest.approx(0.5)
    assert m["abstention"]["n"] == 1 and m["abstention"]["mean_tokens"] == 40
    assert m["tokens"]["mean"] == pytest.approx(25.0)
    assert m["latency"]["total"]["p50"] == pytest.approx(2.5)
    assert set(m["latency"]["stages"]) == {"bm25"}


def test_missing_bundle_is_counted_not_silent():
    questions = [_q("a", "single-session-user", ["s1"], ["a:s1:0"]), _q("b", "multi-session", ["s1"], [])]
    m = evaluate([_bundle("a", ["a:s1:0"])], questions)
    assert m["counts"]["missing_bundles"] == 1


def test_random_baseline_is_seeded():
    notes = [Note(f"h:s{s}:{t}", "h", f"s{s}", t, "user", "x", "2023-01-01") for s in range(5) for t in range(10)]
    questions = [_q("h", "single-session-user", ["s0"], ["h:s0:0"])]
    a = random_baseline(questions, notes, seed=0, k=20)
    b = random_baseline(questions, notes, seed=0, k=20)
    assert a == b
    assert 0.0 <= a["turn_recall@20"] <= 1.0


def test_markdown_table_cites_runs():
    questions = [_q("a", "single-session-user", ["s1"], ["a:s1:0"])]
    table = to_markdown_table({"p0-fx-seed0-warm": evaluate([_bundle("a", ["a:s1:0"])], questions)})
    assert "p0-fx-seed0-warm" in table and "| run |" in table


def test_write_reports(tmp_path):
    from mem27.retrieval.metrics import write_reports
    from mem27.schema import write_jsonl

    questions = [_q("a", "single-session-user", ["s1"], ["a:s1:0"])]
    qpath = tmp_path / "questions.jsonl"
    write_jsonl(qpath, questions)
    run = tmp_path / "runs" / "p0-fx-seed0-warm"
    write_jsonl(run / "bundles.jsonl", [_bundle("a", ["a:s1:0"])])
    table = tmp_path / "table.md"
    metrics = write_reports([run], qpath, table)
    assert (run / "metrics.json").exists()
    assert "p0-fx-seed0-warm" in table.read_text()
    assert metrics["p0-fx-seed0-warm"]["overall"]["turn_recall@20"] == 1.0
