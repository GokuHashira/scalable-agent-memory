"""Tests for A3, the learned per-question retrieval router.

Covers: (a) the router trains on dev and predicts valid fixed-plan ids;
(b) eval_a3 writes a 400-row a3 bundle file; (c) a3's test turn-hit@5 is not
much worse than the best single fixed plan (routing sanity bound).

These use the cached Slice-K data / ref-plan bundles already in the repo.
"""
import json
import os

import numpy as np
import pytest

from mem27 import registry
from mem27.metrics import success as success_fn
from mem27.oracle import load_plan_bundles
from mem27.planner import eval_a3
from mem27.planner.a3_router import A3Router, FEATURE_NAMES, make_labels
from mem27.planner.schema import load_questions, load_notes, load_bundles

DATA = os.environ.get("K_DATA", "data/lme_s_k")
RUN_DIRS = [f"runs/ref_p{i}_lme_s_k_seed0_warm" for i in range(6)]
_have_data = os.path.exists(os.path.join(DATA, "questions.jsonl")) and all(
    os.path.exists(os.path.join(d, "bundles.jsonl")) for d in RUN_DIRS)
pytestmark = pytest.mark.skipif(not _have_data, reason="Slice-K data / ref runs absent")


@pytest.fixture(scope="module")
def eval_result():
    """Train + route once; shared across tests (also writes the artifacts)."""
    return eval_a3.run(dataset=DATA, success_on="note", k=5, bootstrap=200, seed=0)


def test_router_predicts_valid_plans():
    """Router trains on dev and only ever predicts registered fixed-plan ids."""
    questions = load_questions(os.path.join(DATA, "questions.jsonl"))
    dev = [q for q in questions if q.split == "dev"]
    plan_bundles = load_plan_bundles(RUN_DIRS)
    notes = load_notes(os.path.join(DATA, "notes.jsonl"))
    indexes = eval_a3.build_indexes(notes)

    X, dev_qids = eval_a3.build_feature_matrix(dev, plan_bundles, {}, indexes)
    assert X.shape == (len(dev), len(FEATURE_NAMES))
    labels, _ = make_labels(dev, plan_bundles, "note", 5)
    y = [labels[qid] for qid in dev_qids]

    router = A3Router(classes=registry.FIXED_PLANS, seed=0).fit(X, y)
    preds = router.predict(X)
    assert len(preds) == len(dev)
    assert all(p in registry.FIXED_PLANS for p in preds)
    # deterministic given seed 0
    preds2 = A3Router(classes=registry.FIXED_PLANS, seed=0).fit(X, y).predict(X)
    assert preds == preds2


def test_a3_bundle_file_has_400_rows(eval_result):
    """eval_a3 writes one a3 bundle per test question (400), all routed."""
    path = eval_result["bundles_path"]
    assert os.path.exists(path)
    bundles = load_bundles(path)
    assert len(bundles) == 400
    assert all(b.plan == "a3" for b in bundles)
    assert all(b.trace.get("routed_to") in registry.FIXED_PLANS for b in bundles)
    assert all(len(b.note_ids) > 0 for b in bundles)


def test_a3_not_much_worse_than_best_fixed(eval_result):
    """Routing sanity: a3 turn-hit@5 >= best single fixed plan - 0.02."""
    a3 = eval_result["a3"]["turn_hit@5"]
    best_fixed = eval_result["best_fixed"]["turn_hit@5"]
    assert a3 >= best_fixed - 0.02, (a3, best_fixed)
    # and the oracle is a genuine upper bound above both
    assert eval_result["oracle_upper_bound"]["turn_hit@5"] >= best_fixed
