"""Seeded synthetic fixture: small histories with planted facts so retrieval quality is measurable offline.

Each question gets its own history (as in LongMemEval). A fact "my <attribute> is <value>" is planted in the gold
session(s); everything else is filler drawn from a shared vocabulary, so lexical and semantic retrieval both have a
signal to find and random retrieval does not.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from mem27.retrieval import ids
from mem27.schema import Note, Question, write_jsonl

QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)
SYLLABLES = ("ka", "lo", "mi", "ren", "tu", "va", "zor", "pe", "qui", "sha", "dro", "nel", "fi", "gar", "bo", "xu")


@dataclass(frozen=True)
class FixtureShape:
    questions: int = 50
    sessions_per_history: int = 10
    vocab_size: int = 300
    filler_words: tuple[int, int] = (8, 15)

    def turns_per_session(self, target_turn_notes: int) -> int:
        return max(2, target_turn_notes // (self.questions * self.sessions_per_history))


def _word(rng: random.Random) -> str:
    return "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 3)))


def _filler(rng: random.Random, vocab: list[str], shape: FixtureShape) -> str:
    return " ".join(rng.choice(vocab) for _ in range(rng.randint(*shape.filler_words)))


def generate(
    n_turn_notes: int = 5000, seed: int = 0, shape: FixtureShape | None = None
) -> tuple[list[Note], list[Question]]:
    """Returns (notes, questions); identical for identical arguments."""
    shape = shape or FixtureShape()
    rng = random.Random(seed)
    vocab = sorted({_word(rng) for _ in range(shape.vocab_size * 2)})[: shape.vocab_size]
    turns = shape.turns_per_session(n_turn_notes)
    base = datetime(2023, 1, 1, 9, 0)
    notes: list[Note] = []
    questions: list[Question] = []
    for q_idx in range(shape.questions):
        qtype = QUESTION_TYPES[q_idx % len(QUESTION_TYPES)]
        abstention = q_idx == shape.questions - 1
        history = f"fx{q_idx:03d}" + ("_abs" if abstention else "")
        attribute, value = f"{_word(rng)}{_word(rng)}", _word(rng)
        n_gold = 0 if abstention else (2 if qtype == "multi-session" else 1)
        gold_sessions = sorted(rng.sample(range(shape.sessions_per_history), n_gold))
        gold_session_ids, gold_note_ids = [], []
        for s_idx in range(shape.sessions_per_history):
            session = f"s{s_idx}"
            date = (base + timedelta(days=7 * s_idx + q_idx)).isoformat()
            planted_turn = rng.randrange(0, turns, 2) if s_idx in gold_sessions else None
            session_turns = []
            for t_idx in range(turns):
                role = "user" if t_idx % 2 == 0 else "assistant"
                text = _filler(rng, vocab, shape)
                if t_idx == planted_turn:
                    text = f"{text} by the way my {attribute} is {value}"
                    gold_note_ids.append(ids.note_id(history, session, t_idx))
                session_turns.append(
                    Note(
                        note_id=ids.note_id(history, session, t_idx),
                        history_id=history,
                        session_id=session,
                        turn_idx=t_idx,
                        role=role,
                        text=text,
                        ts=date,
                        summary_id=ids.summary_id(history, session),
                    )
                )
            if s_idx in gold_sessions:
                gold_session_ids.append(session)
            notes.append(
                Note(
                    note_id=ids.note_id(history, session, ids.PARENT_TURN_IDX),
                    history_id=history,
                    session_id=session,
                    turn_idx=ids.PARENT_TURN_IDX,
                    role="session",
                    text=ids.parent_text(date, session_turns[0].text),
                    ts=date,
                    summary_id=ids.summary_id(history, session),
                )
            )
            notes.extend(session_turns)
        asked = f"{_word(rng)}{_word(rng)}" if abstention else attribute
        questions.append(
            Question(
                question_id=history,
                history_id=history,
                question_type=qtype,
                question=f"what is my {asked}",
                question_date=(base + timedelta(days=60 + q_idx)).isoformat(),
                answer="unknown" if abstention else value,
                gold_session_ids=gold_session_ids,
                gold_note_ids=gold_note_ids,
                split="dev" if q_idx % 5 == 0 else "test",
            )
        )
    notes.sort(key=lambda n: n.note_id)
    return notes, questions


def write_fixture(out_dir: str | Path, n_turn_notes: int = 5000, seed: int = 0) -> dict[str, int]:
    notes, questions = generate(n_turn_notes, seed)
    out = Path(out_dir)
    return {
        "notes": write_jsonl(out / "notes.jsonl", notes),
        "questions": write_jsonl(out / "questions.jsonl", questions),
    }
