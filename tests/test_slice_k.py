"""Smoke + correctness tests for Slice K."""
import json
import os

import numpy as np
import pytest

from mem27 import registry
from mem27.planner import schema
from mem27.planner.schema import Note, Question, Bundle, write_jsonl, load_notes, load_questions
from mem27 import metrics
from mem27.planner import refcommon

DATA = os.environ.get("K_DATA", "data/lme_s_k")


# ---- schema I/O ----
def test_schema_roundtrip(tmp_path):
    n = Note(note_id="q:s:0", history_id="q", session_id="s", turn_idx=0,
             role="user", text="hi", ts="2023/01/01")
    p = tmp_path / "n.jsonl"
    write_jsonl(str(p), [n])
    got = load_notes(str(p))
    assert got[0].note_id == "q:s:0" and got[0].role == "user"


def test_question_fields():
    q = Question(question_id="q", history_id="q", question_type="multi-session",
                 question="?", gold_session_ids=["s1"], gold_note_ids=["q:s1:0"])
    assert q.gold_session_ids == ["s1"]


# ---- metrics ----
def test_session_of():
    assert metrics.session_of("qid:answer_abc:3") == "answer_abc"


def test_session_recall():
    nids = ["q:s1:0", "q:s2:1", "q:s3:0"]
    assert metrics.session_recall_at_k(nids, ["s2"], 5) == 1.0
    assert metrics.session_recall_at_k(nids, ["s9"], 5) == 0.0
    assert metrics.session_recall_at_k(nids, ["s1", "s9"], 5) == 0.5


def test_note_recall_and_hit():
    nids = ["q:s1:0", "q:s1:1", "q:s2:0"]
    assert metrics.note_recall_at_k(nids, ["q:s1:1"], 5) == 1.0
    assert metrics.note_hit_at_k(nids, ["q:s9:9"], 5) == 0
    assert metrics.note_recall_at_k(nids, ["q:s1:0", "q:s2:0"], 2) == 0.5


def test_percentile():
    assert metrics.percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)


# ---- refcommon ----
def test_rrf_fuse():
    f = refcommon.rrf_fuse([[0, 1, 2], [2, 1, 0]], k=60)
    assert f[1] == pytest.approx(2 / 62)  # rank 1 in both lists


def test_token_count():
    assert refcommon.count_tokens("hello world") >= 1


# ---- registry ----
def test_ref_retrievers_registered():
    registry.ensure_ref_retrievers()
    for p in registry.FIXED_PLANS:
        assert p in registry.available()


# ---- dataset integrity (needs converted data) ----
@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "questions.jsonl")),
                    reason="dataset not built")
def test_dataset_500_six_types():
    qs = load_questions(os.path.join(DATA, "questions.jsonl"))
    assert len(qs) == 500
    assert set(q.question_type for q in qs) == set(schema.QUESTION_TYPES)
    assert sum(q.split == "dev" for q in qs) == 100
    assert sum(q.split == "test" for q in qs) == 400


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "notes.jsonl")),
                    reason="dataset not built")
def test_gold_ids_resolve():
    qs = load_questions(os.path.join(DATA, "questions.jsonl"))
    note_ids = set(n.note_id for n in load_notes(os.path.join(DATA, "notes.jsonl")))
    for q in qs[:50]:
        for gn in q.gold_note_ids:
            assert gn in note_ids


# ---- retriever behaviour on a tiny in-memory history ----
def test_retriever_ranks_relevant_note():
    registry.ensure_ref_retrievers()
    from mem27.planner.ref_retrievers import HistoryIndex
    notes = [
        Note(note_id="q:s1:0", history_id="q", session_id="s1", turn_idx=0,
             role="user", text="I love hiking in the mountains"),
        Note(note_id="q:s1:1", history_id="q", session_id="s1", turn_idx=1,
             role="assistant", text="Great, mountains are beautiful"),
        Note(note_id="q:s2:0", history_id="q", session_id="s2", turn_idx=0,
             role="user", text="My favorite programming language is Python"),
    ]
    # fake embeddings: note0 close to query, others orthogonal-ish
    emb = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    idx = HistoryIndex(history_id="q", notes=notes, emb=emb)
    qvec = np.array([1, 0, 0], dtype=np.float32)
    r = registry.get("ref:p1").retrieve("hiking mountains", qvec, idx)  # dense
    assert r.indices[0] == 0
    r2 = registry.get("ref:p0").retrieve("hiking mountains", qvec, idx)  # bm25
    assert r2.indices[0] == 0
