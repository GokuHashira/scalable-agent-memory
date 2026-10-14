"""Every plan on the fixture: budget, result cap, determinism, empty results, and a recall floor (FR-004–013, 019)."""

import pytest

from mem27.retrieval.evalg import evaluate, random_baseline
from mem27.retrieval.plans import make_plan
from mem27.schema import RetrievalRequest, to_dict

PLANS = ["p0", "p1", "p2"]


def _build(plan, history_notes, encoder, config):
    retriever = make_plan(plan, encoder, config)
    retriever.build(history_notes, None)
    return retriever


def _run(plan, notes_by_history, questions, encoder, config):
    bundles = []
    for q in questions:
        retriever = _build(plan, notes_by_history[q.history_id], encoder, config)
        bundle = retriever.retrieve(RetrievalRequest(query=q.question, question_id=q.question_id))
        bundles.append(bundle)
    return bundles


@pytest.fixture(scope="module", params=PLANS)
def plan_run(request, notes_by_history, questions, encoder, config):
    return request.param, _run(request.param, notes_by_history, questions, encoder, config)


def test_one_bundle_per_question_within_budget(plan_run, questions, notes_by_history):
    plan, bundles = plan_run
    assert [b.question_id for b in bundles] == [q.question_id for q in questions]
    for bundle in bundles:
        assert bundle.plan == plan
        assert bundle.total_tokens <= 2048
        assert len(bundle.note_ids) <= 20
        assert len(set(bundle.note_ids)) == len(bundle.note_ids)
        assert all(not nid.endswith(":-1") for nid in bundle.note_ids), "session parents are not evidence"
        assert bundle.latency_ms["total"] >= sum(bundle.latency_ms["stages"].values())
        assert bundle.trace["stop_reason"]


def test_deterministic(plan_run, notes_by_history, questions, encoder, config):
    plan, bundles = plan_run
    again = _run(plan, notes_by_history, questions[:10], encoder, config)
    for first, second in zip(bundles[:10], again, strict=True):
        assert first.note_ids == second.note_ids
        assert first.scores == second.scores


def test_recall_floor(plan_run, questions, notes):
    plan, bundles = plan_run
    metrics = evaluate([to_dict(b) for b in bundles], questions)
    floor = random_baseline(questions, notes, seed=0, k=20)
    assert metrics["overall"]["turn_recall@20"] >= 2 * floor["turn_recall@20"], (plan, metrics["overall"], floor)


def test_zero_match_lexical_query_is_empty_not_error(notes_by_history, encoder, config):
    history = next(iter(notes_by_history.values()))
    retriever = _build("p0", history, encoder, config)
    bundle = retriever.retrieve(RetrievalRequest(query="zzzqqqxxx", question_id="none"))
    assert bundle.note_ids == []
    assert bundle.trace["stop_reason"] == "no_match"
