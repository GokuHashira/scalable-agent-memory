"""DATA-01 checks on the converted LongMemEval sets (skips if not converted yet)."""
from collections import Counter
from datetime import datetime
from pathlib import Path

import pytest

from mem27.data.longmemeval import QUESTION_TYPES, parse_lme_date, stratified_split
from mem27.vschema import Question, iter_jsonl, read_jsonl

ROOT = Path(__file__).resolve().parents[1] / "data"


def _load(name):
    d = ROOT / name
    if not (d / "questions.jsonl").exists():
        pytest.skip(f"{d} not converted")
    qs = read_jsonl(d / "questions.jsonl", Question)
    note_ids, sessions, bad_ts, leaks = set(), set(), 0, 0
    for n in iter_jsonl(d / "notes.jsonl"):
        note_ids.add(n["note_id"])
        sessions.add((n["history_id"], n["session_id"]))
        try:
            datetime.fromisoformat(n["ts"])
        except ValueError:
            bad_ts += 1
        if "answer" in n["session_id"] or "has_answer" in n["metadata"]:
            leaks += 1
    return qs, note_ids, sessions, bad_ts, leaks


@pytest.mark.parametrize("name", ["lme_s", "lme_oracle"])
def test_counts_types_split(name):
    qs, *_ = _load(name)
    assert len(qs) == 500
    assert set(Counter(q.question_type for q in qs)) == set(QUESTION_TYPES)
    assert Counter(q.split for q in qs) == {"dev": 100, "test": 400}
    assert len({q.question_id for q in qs}) == 500


@pytest.mark.parametrize("name", ["lme_s", "lme_oracle"])
def test_gold_ids_exist_and_ts_parse(name):
    qs, note_ids, sessions, bad_ts, leaks = _load(name)
    assert bad_ts == 0
    assert leaks == 0, "session ids / metadata leak gold"
    for q in qs:
        datetime.fromisoformat(q.question_date)
        for nid in q.gold_note_ids:
            assert nid in note_ids, nid
        for sid in q.gold_session_ids:
            assert (q.history_id, sid) in sessions, (q.question_id, sid)
    assert sum(bool(q.gold_session_ids) for q in qs) == 500


def test_split_shared_between_s_and_oracle():
    a, *_ = _load("lme_s")
    b, *_ = _load("lme_oracle")
    assert {q.question_id: q.split for q in a} == {q.question_id: q.split for q in b}
    # gold ids agree too: same anonymisation, same haystack session for the evidence
    assert {q.question_id: sorted(q.gold_session_ids) for q in a} == {q.question_id: sorted(q.gold_session_ids) for q in b}


def test_parse_date():
    assert parse_lme_date("2023/05/20 (Sat) 02:21") == "2023-05-20T02:21:00"


def test_stratified_split_unit():
    qs = [Question(f"q{i}{'_abs' if i % 10 == 0 else ''}", "h", QUESTION_TYPES[i % 6], "", "", "", [], [], "")
          for i in range(500)]
    s1, s2 = stratified_split(qs, 100, 0), stratified_split(qs, 100, 0)
    assert s1 == s2 and Counter(s1.values()) == {"dev": 100, "test": 400}
    for t in QUESTION_TYPES:  # ~20% of each type in dev
        n = sum(1 for q in qs if q.question_type == t)
        d = sum(1 for q in qs if q.question_type == t and s1[q.question_id] == "dev")
        assert abs(d - n / 5) <= 1.5
