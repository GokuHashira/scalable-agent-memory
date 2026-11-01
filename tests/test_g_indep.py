"""Tests for the independent gi:p0..p5 reference family (planner §5 robustness)."""
import numpy as np

from mem27.planner.schema import Note
from mem27.planner.ref_retrievers import HistoryIndex
from mem27.planner.g_plans_indep import GI_CLASSES, register_gi


def _tiny_index():
    notes = [
        Note(note_id="q:s1:0", history_id="q", session_id="s1", turn_idx=0,
             role="user", text="I love hiking in the mountains every weekend"),
        Note(note_id="q:s1:1", history_id="q", session_id="s1", turn_idx=1,
             role="assistant", text="Mountains are wonderful for hiking"),
        Note(note_id="q:s2:0", history_id="q", session_id="s2", turn_idx=0,
             role="user", text="My favorite programming language is Python"),
        Note(note_id="q:s2:1", history_id="q", session_id="s2", turn_idx=1,
             role="assistant", text="Python is great for data science"),
    ]
    emb = np.array([[1, 0, 0.1], [0.9, 0.1, 0], [0, 1, 0], [0, 0.9, 0.1]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return HistoryIndex(history_id="q", notes=notes, emb=emb)


def test_all_gi_plans_return_valid_results():
    idx = _tiny_index()
    qvec = np.array([1, 0, 0], dtype=np.float32)
    for cls in GI_CLASSES:
        r = cls().retrieve("hiking mountains", qvec, idx)
        assert all(0 <= i < idx.n for i in r.indices)
        assert len(r.indices) == len(r.scores)
        assert r.trace.get("impl") == "gi"


def test_gi_lexical_and_dense_find_relevant():
    idx = _tiny_index()
    qvec = np.array([1, 0, 0], dtype=np.float32)
    # tf-idf on lexical query should rank a hiking turn first
    from mem27.planner.g_plans_indep import GITfidf, GIDensePRF
    r = GITfidf().retrieve("hiking mountains", qvec, idx)
    assert r.indices[0] in (0, 1)
    r2 = GIDensePRF().retrieve("hiking mountains", qvec, idx)
    assert r2.indices[0] in (0, 1)


def test_register_gi():
    from mem27 import registry
    ids = register_gi()
    assert ids == [f"gi:p{i}" for i in range(6)]
    assert registry.get("gi:p0") is not None
