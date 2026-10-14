"""LongMemEval_S → notes/questions conversion (FR-001–003)."""

import copy
import json

import pytest

from mem27.retrieval.data_g import ConversionError, convert_lme_s, convert_records
from mem27.schema import load_notes, load_questions

RAW = [
    {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "What car did I buy?",
        "answer": "a red bike",
        "question_date": "2023/05/30 (Tue) 10:00",
        "haystack_session_ids": ["sA", "sB"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21", "2023/05/21 (Sun) 09:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "hello"}],
            [
                {"role": "user", "content": "I bought a red bike", "has_answer": True},
                {"role": "assistant", "content": "nice"},
                {"role": "user", "content": "thanks"},
            ],
        ],
        "answer_session_ids": ["sB"],
    },
    {
        "question_id": "q2_abs",
        "question_type": "temporal-reasoning",
        "question": "When did I visit Mars?",
        "answer": 42,
        "question_date": "2023/06/01 (Thu) 08:00",
        "haystack_session_ids": ["sC"],
        "haystack_dates": ["2023/05/25 (Thu) 12:00"],
        "haystack_sessions": [[{"role": "user", "content": "nothing about space"}]],
        "answer_session_ids": [],
    },
]


def test_notes_per_turn_plus_parents():
    notes, _ = convert_records(copy.deepcopy(RAW))
    q1 = [n for n in notes if n.history_id == "q1"]
    assert len([n for n in q1 if not n.is_session_parent]) == 5
    parents = [n for n in q1 if n.is_session_parent]
    assert [p.note_id for p in parents] == ["q1:sA:-1", "q1:sB:-1"]
    assert parents[1].text.startswith("2023-05-21T09:00:00 I bought a red bike")


def test_ids_timestamps_and_summary_links():
    notes, _ = convert_records(copy.deepcopy(RAW))
    by_id = {n.note_id: n for n in notes}
    turn = by_id["q1:sB:0"]
    assert (turn.history_id, turn.session_id, turn.turn_idx, turn.role) == ("q1", "sB", 0, "user")
    assert turn.ts == "2023-05-21T09:00:00"
    assert turn.summary_id == "q1:sB:sum"
    assert [n.note_id for n in notes] == sorted(by_id)


def test_gold_ids_and_types():
    _, questions = convert_records(copy.deepcopy(RAW))
    q1, q2 = questions
    assert q1.gold_session_ids == ["sB"] and q1.gold_note_ids == ["q1:sB:0"]
    assert q1.question_type == "single-session-user"
    assert q1.question_date == "2023-05-30T10:00:00"
    assert q2.question_id.endswith("_abs") and q2.gold_session_ids == [] and q2.answer == "42"


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda r: r.pop("haystack_sessions"), "q1: missing field 'haystack_sessions'"),
        (lambda r: r["haystack_sessions"][0][0].pop("content"), "q1: session 'sA' turn 0: missing 'content'"),
        (lambda r: r["haystack_dates"].pop(), "q1: haystack_dates has 1 entries"),
        (lambda r: r.__setitem__("haystack_dates", ["yesterday", "today"]), "q1: unparseable date 'yesterday'"),
        (lambda r: r.__setitem__("answer_session_ids", ["sZ"]), "q1: answer session 'sZ' not in haystack"),
    ],
)
def test_malformed_records_fail_with_actionable_message(mutate, needle):
    raw = copy.deepcopy(RAW)
    mutate(raw[0])
    with pytest.raises(ConversionError, match=needle.replace("(", r"\(").replace(")", r"\)")):
        convert_records(raw)


def test_convert_file_roundtrip(tmp_path):
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(RAW))
    counts = convert_lme_s(raw_path, tmp_path / "out")
    assert counts == {"questions": 2, "notes": 9, "turn_notes": 6, "abstention": 1, "empty_gold_turns": 1}
    assert len(load_notes(tmp_path / "out" / "notes.jsonl")) == 9
    assert [q.question_id for q in load_questions(tmp_path / "out" / "questions.jsonl")] == ["q1", "q2_abs"]
