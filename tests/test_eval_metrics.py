"""EVAL-02: hand-built bundles with known recall / precision."""
import json

import pytest

from mem27.eval.metrics import aggregate, compute, question_row
from mem27.vschema import Question, write_jsonl


def Q(qid="q1", gs=("A", "B"), gt=("q1:A:1", "q1:B:0"), t="multi-session", split="test"):
    return Question(qid, qid, t, "?", "2023-01-01T00:00:00", "x", list(gs), list(gt), split)


def B(ids, toks=100, plan="p"):
    return {"question_id": "q1", "plan": plan, "note_ids": ids, "passages": ["word " * 10] * len(ids),
            "scores": [1.0] * len(ids), "total_tokens": toks, "latency_ms": {"total": 5.0, "stages": {"bm25": 2.0}},
            "stale_flags": [False] * len(ids)}


def test_recall_precision_known():
    ids = ["q1:A:1", "q1:C:0", "q1:C:1", "q1:D:0", "q1:E:0", "q1:B:0"]  # B only at rank 6
    r = question_row(Q(), B(ids), None)
    assert r["sess_recall@5"] == 0.5 and r["sess_all@5"] == 0.0
    assert r["sess_recall@10"] == 1.0 and r["sess_all@10"] == 1.0
    assert r["turn_recall@5"] == 0.5 and r["turn_recall@10"] == 1.0
    assert r["ctx_precision"] == pytest.approx(2 / 6)
    assert r["mrr_sess"] == 1.0


def test_session_parent_covers_turns():
    r = question_row(Q(), B(["q1:B:-1", "q1:A:-1"]), None)
    assert r["turn_recall@5"] == 1.0 and r["sess_recall@5"] == 1.0


def test_abstention_excluded_from_recall():
    r = question_row(Q(qid="q1_abs", gt=()), B(["q1:A:1"]), None)
    assert "sess_recall@5" not in r and "ctx_precision" not in r


def test_accuracy_and_conditionals():
    rows = [question_row(Q(), B(["q1:A:1", "q1:B:0"]), {"judge": {"correct": True}}),
            question_row(Q(), B(["q1:A:1"]), {"judge": {"correct": False}}),
            question_row(Q(t="knowledge-update"), B([]), {"judge": {"correct": True}})]
    a = aggregate(rows)
    assert a["accuracy"] == pytest.approx(2 / 3)
    assert a["accuracy_task_avg"] == pytest.approx((0.5 + 1.0) / 2)
    assert a["acc_given_all_gold"] == 1.0 and a["acc_given_gold_missing"] == 0.5
    assert a["latency_p50"] == 5.0 and a["stages"]["bm25"]["p95"] == 2.0


def test_compute_roundtrip(tmp_path):
    data, run = tmp_path / "data", tmp_path / "runs" / "r1"
    write_jsonl(data / "questions.jsonl", [Q(split="dev")])
    write_jsonl(run / "bundles.jsonl", [B(["q1:A:1", "q1:B:0"])])
    write_jsonl(run / "answers.jsonl", [{"question_id": "q1", "answer": "x", "judge": {"correct": True, "judge_model": "j"}}])
    (run / "env.json").write_text(json.dumps({"host": "test"}))
    m = compute(run, data)
    assert m["aggregate"]["dev"]["accuracy"] == 1.0 and "test" not in m["aggregate"]
    assert m["env"]["host"] == "test" and m["judge_model"] == "j"
