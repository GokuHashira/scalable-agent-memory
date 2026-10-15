"""Every plan on the fixture: budget, result cap, determinism, empty results, and a recall floor (FR-004–013, 019)."""

import pytest

from mem27.retrieval.metrics import evaluate, random_baseline
from mem27.retrieval.plans import make_plan
from mem27.retrieval.summaries import build_extractive_summaries
from mem27.schema import RetrievalRequest, to_dict

PLANS = ["p0", "p1", "p2", "p3", "p4", "p5"]


def _build(plan, history_notes, encoder, config):
    retriever = make_plan(plan, encoder, config)
    summaries = build_extractive_summaries(history_notes, created_ts="t") if plan == "p3" else None
    retriever.build(history_notes, summaries)
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


def test_plan_specific_traces(notes_by_history, questions, encoder, config):
    q = questions[0]
    history = notes_by_history[q.history_id]
    req = RetrievalRequest(query=q.question, question_id=q.question_id)
    p3 = _build("p3", history, encoder, config).retrieve(req)
    assert p3.trace["summary_kind"] == "extractive"
    assert 0 < len(p3.trace["sessions"]) <= 5
    p5 = _build("p5", history, encoder, config).retrieve(req)
    assert p5.trace["hops_used"] <= 4
    assert p5.trace["candidates_visited"] <= 50
    assert p5.trace["stop_reason"] in {"cap", "frontier_empty", "hops"}


def test_onehop_candidates_are_anchors_or_neighbours(notes_by_history, questions, encoder, config):
    q = questions[1]
    history = notes_by_history[q.history_id]
    p4 = _build("p4", history, encoder, config)
    bundle = p4.retrieve(RetrievalRequest(query=q.question, question_id=q.question_id))
    anchors = bundle.trace["anchors"]
    allowed = set(anchors) | {lk.target_note_id for a in anchors for lk in p4.links.neighbours(a)}
    assert set(bundle.note_ids) <= allowed
    assert bundle.note_ids[: len(anchors)] == anchors[: len(bundle.note_ids)]
