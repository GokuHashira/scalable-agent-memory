"""LongMemEval_S → shared notes/questions (G's own copy; V's canonical adapter replaces it later).

This module is the barricade: every raw record is validated here and rejected with a message naming the question
and field, so code downstream can assume well-formed notes (constitution II: correctness over robustness).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from mem27.retrieval import ids
from mem27.schema import Note, Question, write_jsonl

LME_DATE_FORMAT = "%Y/%m/%d (%a) %H:%M"
REQUIRED_FIELDS = (
    "question_id",
    "question_type",
    "question",
    "answer",
    "question_date",
    "haystack_session_ids",
    "haystack_dates",
    "haystack_sessions",
    "answer_session_ids",
)


class ConversionError(ValueError):
    """A raw record is malformed; the message names the question id and the offending field."""


def _iso(qid: str, raw_date: Any) -> str:
    try:
        return datetime.strptime(str(raw_date), LME_DATE_FORMAT).isoformat()
    except ValueError as exc:
        raise ConversionError(f"{qid}: unparseable date {raw_date!r}; expected like '2023/05/20 (Sat) 02:21'") from exc


def _validate(record: dict[str, Any]) -> None:
    qid = record.get("question_id", "<no question_id>")
    for name in REQUIRED_FIELDS:
        if name not in record:
            raise ConversionError(f"{qid}: missing field {name!r}")
    n_ids, n_dates, n_sessions = (
        len(record["haystack_session_ids"]),
        len(record["haystack_dates"]),
        len(record["haystack_sessions"]),
    )
    if not n_ids == n_dates == n_sessions:
        raise ConversionError(
            f"{qid}: haystack_dates has {n_dates} entries, haystack_sessions {n_sessions}, "
            f"haystack_session_ids {n_ids}; they must align one per session"
        )
    if len(set(record["haystack_session_ids"])) != n_ids:
        raise ConversionError(f"{qid}: duplicate session ids in haystack_session_ids")
    for answer_session in record["answer_session_ids"]:
        if answer_session not in record["haystack_session_ids"]:
            raise ConversionError(f"{qid}: answer session {answer_session!r} not in haystack")
    for session_id, turns in zip(record["haystack_session_ids"], record["haystack_sessions"], strict=True):
        for t_idx, turn in enumerate(turns):
            for key in ("role", "content"):
                if key not in turn:
                    raise ConversionError(f"{qid}: session {session_id!r} turn {t_idx}: missing {key!r}")


def _session_notes(qid: str, session_id: str, iso_date: str, turns: list[dict], parent_tokens: int) -> list[Note]:
    sum_id = ids.summary_id(qid, session_id)
    turn_notes = [
        Note(
            note_id=ids.note_id(qid, session_id, t_idx),
            history_id=qid,
            session_id=session_id,
            turn_idx=t_idx,
            role=str(turn["role"]),
            text=str(turn["content"]),
            ts=iso_date,
            summary_id=sum_id,
        )
        for t_idx, turn in enumerate(turns)
    ]
    opener = next((n.text for n in turn_notes if n.role == "user"), turn_notes[0].text if turn_notes else "")
    parent = Note(
        note_id=ids.note_id(qid, session_id, ids.PARENT_TURN_IDX),
        history_id=qid,
        session_id=session_id,
        turn_idx=ids.PARENT_TURN_IDX,
        role="session",
        text=ids.parent_text(iso_date, opener, parent_tokens),
        ts=iso_date,
        summary_id=sum_id,
    )
    return [parent, *turn_notes]


def convert_record(record: dict[str, Any], parent_tokens: int = 64) -> tuple[list[Note], Question]:
    _validate(record)
    qid = str(record["question_id"])
    notes: list[Note] = []
    gold_note_ids: list[str] = []
    sessions = zip(record["haystack_session_ids"], record["haystack_dates"], record["haystack_sessions"], strict=True)
    for session_id, raw_date, turns in sessions:
        notes.extend(_session_notes(qid, session_id, _iso(qid, raw_date), turns, parent_tokens))
        gold_note_ids.extend(ids.note_id(qid, session_id, t) for t, turn in enumerate(turns) if turn.get("has_answer"))
    question = Question(
        question_id=qid,
        history_id=qid,
        question_type=str(record["question_type"]),
        question=str(record["question"]),
        question_date=_iso(qid, record["question_date"]),
        answer=str(record["answer"]),
        gold_session_ids=list(record["answer_session_ids"]),
        gold_note_ids=gold_note_ids,
        split="test",  # placeholder; the frozen dev/test split comes later (data-model.md)
    )
    return notes, question


def convert_records(records: list[dict[str, Any]], parent_tokens: int = 64) -> tuple[list[Note], list[Question]]:
    """All histories, notes sorted by id; question ids must be unique because each names its own history."""
    seen: set[str] = set()
    notes: list[Note] = []
    questions: list[Question] = []
    for record in records:
        rec_notes, question = convert_record(record, parent_tokens)
        if question.question_id in seen:
            raise ConversionError(f"{question.question_id}: duplicate question_id; histories must be unique")
        seen.add(question.question_id)
        notes.extend(rec_notes)
        questions.append(question)
    notes.sort(key=lambda n: n.note_id)
    return notes, questions


def convert_lme_s(raw_path: str | Path, out_dir: str | Path, parent_tokens: int = 64) -> dict[str, int]:
    """Writes notes.jsonl and questions.jsonl; returns counts for the run log."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise ConversionError(
            f"{raw_path} not found; download longmemeval_s_cleaned.json from HF xiaowu0162/longmemeval-cleaned"
        )
    records = json.loads(raw_path.read_text(encoding="utf-8"))
    notes, questions = convert_records(records, parent_tokens)
    out = Path(out_dir)
    write_jsonl(out / "notes.jsonl", notes)
    write_jsonl(out / "questions.jsonl", questions)
    return {
        "questions": len(questions),
        "notes": len(notes),
        "turn_notes": sum(1 for n in notes if not n.is_session_parent),
        "abstention": sum(1 for q in questions if ids.is_abstention(q.question_id)),
        "empty_gold_turns": sum(1 for q in questions if not q.gold_note_ids),
    }
