"""Extractive session summaries for the midterm (research R7); LLM summaries replace them post-midterm.

A summary is the session's first turn, last turn and longest user turn (deduplicated, in turn order), so it is
cheap, deterministic and traceable back to its sources.
"""

from __future__ import annotations

from mem27.retrieval import ids
from mem27.schema import Note, Summary

SUMMARY_KIND = "extractive"


def _pick(turns: list[Note]) -> list[Note]:
    user_turns = [n for n in turns if n.role == "user"]
    picks = [turns[0], turns[-1]]
    if user_turns:
        picks.append(max(user_turns, key=lambda n: (len(n.text), -n.turn_idx)))
    unique = {n.note_id: n for n in picks}
    return sorted(unique.values(), key=lambda n: n.turn_idx)


def build_extractive_summaries(notes: list[Note], created_ts: str) -> list[Summary]:
    """One summary per session that has turns, ordered by summary id."""
    sessions: dict[tuple[str, str], list[Note]] = {}
    for note in notes:
        if not note.is_session_parent:
            sessions.setdefault((note.history_id, note.session_id), []).append(note)
    summaries = []
    for (history_id, session_id), turns in sessions.items():
        chosen = _pick(sorted(turns, key=lambda n: n.turn_idx))
        summaries.append(
            Summary(
                summary_id=ids.summary_id(history_id, session_id),
                history_id=history_id,
                session_id=session_id,
                text="\n".join(f"{n.role}: {n.text}" for n in chosen),
                source_note_ids=[n.note_id for n in chosen],
                created_ts=created_ts,
                stale=False,
            )
        )
    return sorted(summaries, key=lambda s: s.summary_id)


def as_pseudo_notes(summaries: list[Summary]) -> list[Note]:
    """Summaries wrapped as notes so the BM25/dense components can index them unchanged."""
    return [
        Note(
            note_id=s.summary_id,
            history_id=s.history_id,
            session_id=s.session_id,
            turn_idx=0,
            role="summary",
            text=s.text,
            ts=s.created_ts,
        )
        for s in summaries
    ]
