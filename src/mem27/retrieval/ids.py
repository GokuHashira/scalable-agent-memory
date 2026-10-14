"""Deterministic identifiers and session-parent text shared by the converter, fixture and metrics."""

from mem27.schema import truncate_text

PARENT_TURN_IDX = -1


def note_id(history_id: str, session_id: str, turn_idx: int) -> str:
    return f"{history_id}:{session_id}:{turn_idx}"


def summary_id(history_id: str, session_id: str) -> str:
    return f"{history_id}:{session_id}:sum"


def session_key(history_id: str, session_id: str) -> str:
    return f"{history_id}:{session_id}"


def session_key_of_note(note_id_value: str) -> str:
    """The `history:session` key of a note id; turn indices never contain ':'."""
    return note_id_value.rsplit(":", 1)[0]


def parent_text(session_date: str, first_user_text: str, max_tokens: int = 64) -> str:
    """Session-parent stand-in text: the session date and its opening user turn, bounded in tokens."""
    return truncate_text(f"{session_date} {first_user_text}".strip(), max_tokens)


def is_abstention(question_id: str) -> bool:
    """LongMemEval marks abstention questions with an `_abs` id suffix (contract has no metadata field)."""
    return question_id.endswith("_abs")
