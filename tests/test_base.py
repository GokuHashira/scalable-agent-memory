"""BaseRetriever template, stage timing, budget packing, and the stub encoder."""

import numpy as np
import pytest

from mem27.retrieval.base import BaseRetriever, SearchResult, StageTimer, pack_to_budget
from mem27.retrieval.encoder import StubEncoder
from mem27.schema import RetrievalRequest, count_tokens


def test_pack_respects_budget_and_max_results():
    passages = ["alpha beta gamma"] * 30
    ids = [f"n{i:02d}" for i in range(30)]
    packed = pack_to_budget(ids, passages, [1.0] * 30, budget_tokens=10, max_results=20)
    assert packed.total_tokens <= 10
    assert packed.note_ids == ids[: len(packed.note_ids)]
    assert packed.stop_reason == "budget"
    packed = pack_to_budget(ids, passages, [1.0] * 30, budget_tokens=10_000, max_results=20)
    assert len(packed.note_ids) == 20
    assert packed.stop_reason == "max_results"


def test_pack_truncates_oversized_top_passage():
    long_text = "word " * 5000
    packed = pack_to_budget(["n0", "n1"], [long_text, "short"], [2.0, 1.0], budget_tokens=100, max_results=20)
    assert packed.note_ids == ["n0"]
    assert packed.truncated is True
    assert count_tokens(packed.passages[0]) <= 100
    assert packed.total_tokens <= 100


def test_pack_exhausted_and_empty():
    packed = pack_to_budget(["n0"], ["a b"], [1.0], budget_tokens=100, max_results=20)
    assert packed.stop_reason == "exhausted"
    empty = pack_to_budget([], [], [], budget_tokens=100, max_results=20)
    assert empty.note_ids == [] and empty.total_tokens == 0


def test_stage_timer_accumulates():
    timer = StageTimer()
    with timer.stage("a"):
        sum(range(1000))
    with timer.stage("a"):
        pass
    with timer.stage("b"):
        pass
    assert set(timer.stages) == {"a", "b"}
    assert all(v >= 0 for v in timer.stages.values())
    assert timer.total_ms() >= sum(timer.stages.values())


class _Echo(BaseRetriever):
    plan_id = "echo"

    def _build_index(self, notes, summaries):
        self._ids = sorted(n.note_id for n in notes if not n.is_session_parent)

    def _search(self, req, timer):
        with timer.stage("echo"):
            return SearchResult(ranked=[(i, 1.0) for i in self._ids])


def test_retrieve_before_build_raises():
    with pytest.raises(RuntimeError, match="index not built"):
        _Echo().retrieve(RetrievalRequest(query="x"))


def test_template_fills_bundle(notes_by_history):
    history = next(iter(notes_by_history.values()))
    retriever = _Echo()
    retriever.build(history, None)
    bundle = retriever.retrieve(RetrievalRequest(query="x", question_id="q1"))
    assert bundle.plan == "echo" and bundle.question_id == "q1"
    assert len(bundle.note_ids) <= 20 and bundle.total_tokens <= 2048
    assert len(bundle.passages) == len(bundle.scores) == len(bundle.stale_flags) == len(bundle.note_ids)
    assert {"echo", "pack"} <= set(bundle.latency_ms["stages"])
    assert bundle.latency_ms["total"] >= sum(bundle.latency_ms["stages"].values())
    assert bundle.trace["plan"] == "echo" and "stop_reason" in bundle.trace


def test_stub_encoder_deterministic_and_normalised():
    enc = StubEncoder()
    a = enc.encode_passages(["red apple pie", "blue sky"])
    b = enc.encode_passages(["red apple pie", "blue sky"])
    assert np.array_equal(a, b)
    assert a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    q = enc.encode_queries(["apple pie"])[0]
    assert q @ a[0] > q @ a[1]
