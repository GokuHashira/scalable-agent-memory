"""save → load gives identical retrieval results (FR-014)."""

import pytest

from mem27.retrieval.plans import make_plan
from mem27.schema import RetrievalRequest


@pytest.mark.parametrize("plan", ["p0", "p1"])
def test_save_load_roundtrip(plan, tmp_path, notes_by_history, questions, encoder, config):
    for q in questions[:10]:
        history = notes_by_history[q.history_id]
        built = make_plan(plan, encoder, config)
        built.build(history, None)
        built.save(tmp_path / q.history_id)
        loaded = make_plan(plan, encoder, config)
        loaded.load(tmp_path / q.history_id, history)
        req = RetrievalRequest(query=q.question, question_id=q.question_id)
        a, b = built.retrieve(req), loaded.retrieve(req)
        assert a.note_ids == b.note_ids
        assert a.scores == b.scores
