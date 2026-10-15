"""Extractive session summaries (FR-008, research R7)."""

from mem27.retrieval.summaries import build_extractive_summaries
from mem27.schema import Note


def _turn(session, idx, role, text):
    return Note(f"h:{session}:{idx}", "h", session, idx, role, text, "2023-01-01")


def test_one_summary_per_session_with_expected_text(notes_by_history):
    history = notes_by_history["fx000"]
    summaries = build_extractive_summaries(history, created_ts="2026-10-14T00:00:00")
    assert [s.summary_id for s in summaries] == [f"fx000:s{i}:sum" for i in range(10)]
    turns = sorted((n for n in history if n.session_id == "s0" and not n.is_session_parent), key=lambda n: n.turn_idx)
    longest_user = max((n for n in turns if n.role == "user"), key=lambda n: (len(n.text), -n.turn_idx))
    expected_ids = sorted(
        {turns[0].note_id, turns[-1].note_id, longest_user.note_id}, key=lambda i: int(i.split(":")[-1])
    )
    first = summaries[0]
    assert first.source_note_ids == expected_ids
    assert first.text.splitlines()[0].endswith(turns[0].text)
    assert first.stale is False and first.created_ts == "2026-10-14T00:00:00"
    assert all(nid.startswith("fx000:s0:") for nid in first.source_note_ids)


def test_dedup_and_single_turn_session():
    notes = [
        Note("h:a:-1", "h", "a", -1, "session", "parent", "2023-01-01"),
        _turn("a", 0, "user", "only turn"),
        _turn("b", 0, "user", "hello there friend"),
        _turn("b", 1, "assistant", "hi"),
    ]
    summaries = {s.session_id: s for s in build_extractive_summaries(notes, created_ts="t")}
    assert summaries["a"].source_note_ids == ["h:a:0"]
    assert summaries["a"].text == "user: only turn"
    assert summaries["b"].source_note_ids == ["h:b:0", "h:b:1"]  # first == longest user, deduplicated
