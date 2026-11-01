"""Post-midterm integration scaffolding tests (planner.md section 5).

Covers: (a) the FIXED_PLAN_IMPL="g" swap registers g:p0..p5 and they satisfy the
.retrieve interface; (b) the correct-based (V-labels) oracle; (c) the existing
evidence-level oracle still works.
"""
import numpy as np
import pytest

from mem27 import registry
from mem27.planner.schema import Note, Question, Bundle, Answer, write_jsonl


# ---- (a) FIXED_PLAN_IMPL="g" registry swap ----
@pytest.fixture
def g_impl():
    """Activate the 'g' impl, restoring the ref registry afterwards."""
    saved_impl = registry.FIXED_PLAN_IMPL
    registry.ensure_ref_retrievers()  # make sure originals exist to restore
    saved = {p: registry.get(p) for p in registry.FIXED_PLANS}
    registry.FIXED_PLAN_IMPL = "g"
    try:
        yield
    finally:
        registry.FIXED_PLAN_IMPL = saved_impl
        for p, inst in saved.items():          # un-alias ref:pK
            registry._REGISTRY[p] = inst
        for g in [f"g:p{i}" for i in range(6)]:  # drop g:pK
            registry._REGISTRY.pop(g, None)


def _tiny_index():
    from mem27.planner.ref_retrievers import HistoryIndex
    notes = [
        Note(note_id="q:s1:0", history_id="q", session_id="s1", turn_idx=0,
             role="user", text="I love hiking in the mountains"),
        Note(note_id="q:s1:1", history_id="q", session_id="s1", turn_idx=1,
             role="assistant", text="Great, mountains are beautiful"),
        Note(note_id="q:s2:0", history_id="q", session_id="s2", turn_idx=0,
             role="user", text="My favorite programming language is Python"),
    ]
    emb = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 0, 1]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return HistoryIndex(history_id="q", notes=notes, emb=emb)


def test_g_impl_registers_and_retrieves(g_impl):
    plans = registry.ensure_fixed_plans()
    assert plans == registry.FIXED_PLANS
    g_ids = [f"g:p{i}" for i in range(6)]
    for g in g_ids:
        assert g in registry.available()
    # ref:pK are aliased onto the same g instances (oracle/run_plan unchanged)
    for i in range(6):
        assert registry.get(f"ref:p{i}") is registry.get(f"g:p{i}")

    idx = _tiny_index()
    qvec = np.array([1, 0, 0], dtype=np.float32)
    for g in g_ids:
        r = registry.get(g).retrieve("hiking mountains", qvec, idx)
        assert hasattr(r, "indices") and hasattr(r, "scores") and hasattr(r, "trace")
        assert len(r.indices) == len(r.scores)
        assert r.trace.get("impl") == "g-standin"
    # delegated retrieval still ranks the relevant note first (dense g:p1)
    assert registry.get("g:p1").retrieve("hiking mountains", qvec, idx).indices[0] == 0


@pytest.fixture
def gi_impl():
    """Activate the genuinely-independent 'gi' impl, restoring ref afterwards."""
    saved_impl = registry.FIXED_PLAN_IMPL
    registry.ensure_ref_retrievers()
    saved = {p: registry.get(p) for p in registry.FIXED_PLANS}
    registry.FIXED_PLAN_IMPL = "gi"
    try:
        yield
    finally:
        registry.FIXED_PLAN_IMPL = saved_impl
        for p, inst in saved.items():
            registry._REGISTRY[p] = inst
        for g in [f"gi:p{i}" for i in range(6)]:
            registry._REGISTRY.pop(g, None)


def test_gi_impl_swap_registers_independent_classes(gi_impl):
    """planner §5 swap executed with the INDEPENDENT family: ref:pK alias onto gi:pK,
    and the gi classes are standalone (not ref delegates)."""
    plans = registry.ensure_fixed_plans()
    assert plans == registry.FIXED_PLANS
    for i in range(6):
        assert registry.get(f"ref:p{i}") is registry.get(f"gi:p{i}")
    # gi plans are genuinely independent implementations (impl tag "gi", not "g-standin")
    idx = _tiny_index()
    qvec = np.array([1, 0, 0], dtype=np.float32)
    for i in range(6):
        r = registry.get(f"ref:p{i}").retrieve("hiking mountains", qvec, idx)
        assert r.trace.get("impl") == "gi"
        assert len(r.indices) == len(r.scores)


# ---- (b)/(c) oracle on a small synthetic dataset ----
def _write_dataset(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    qs = [
        Question(question_id="qa", history_id="qa", question_type="multi-session",
                 question="?", gold_session_ids=["s1"],
                 gold_note_ids=["qa:s1:0"], split="test"),
        Question(question_id="qb", history_id="qb", question_type="temporal-reasoning",
                 question="?", gold_session_ids=["s2"],
                 gold_note_ids=["qb:s2:0"], split="test"),
    ]
    write_jsonl(str(ds / "questions.jsonl"), qs)
    return ds


def _write_runs(tmp_path):
    """Two plans; p0 finds qa's gold, p2 finds qb's gold (they disagree)."""
    runs = []
    specs = {
        "ref:p0": {"qa": ["qa:s1:0"], "qb": ["qb:s9:9"]},
        "ref:p2": {"qa": ["qa:s9:9"], "qb": ["qb:s2:0"]},
    }
    for plan, hits in specs.items():
        d = tmp_path / f"run_{plan.split(':')[1]}"
        d.mkdir()
        rows = [Bundle(question_id=qid, plan=plan, note_ids=nids)
                for qid, nids in hits.items()]
        write_jsonl(str(d / "bundles.jsonl"), rows)
        runs.append(str(d))
    return runs


def test_evidence_oracle_still_works(tmp_path):
    from mem27 import oracle
    ds = _write_dataset(tmp_path)
    runs = _write_runs(tmp_path)
    r = oracle.run(str(ds), runs, success_on="note", split="test", k=20,
                   bootstrap=200, seed=0)
    assert r["success_signal"] == "evidence"
    assert set(("oracle_minus_best_fixed", "ci95")).issubset(r)
    # each plan solves 1/2; oracle solves both -> delta = 0.5
    assert r["any_of_plans_upper_bound"] == 1.0
    assert r["oracle_minus_best_fixed"] == pytest.approx(0.5)


def test_correct_based_oracle(tmp_path):
    from mem27 import oracle
    ds = _write_dataset(tmp_path)
    runs = _write_runs(tmp_path)
    # V-style answers: p0 correct on qa, p2 correct on qb (disagreement)
    answers = [
        Answer(question_id="qa", plan="ref:p0", answer="x",
               judge={"correct": 1, "judge_model": "standin", "raw": ""}),
        Answer(question_id="qa", plan="ref:p2", answer="x",
               judge={"correct": 0, "judge_model": "standin", "raw": ""}),
        Answer(question_id="qb", plan="ref:p0", answer="x",
               judge={"correct": 0, "judge_model": "standin", "raw": ""}),
        Answer(question_id="qb", plan="ref:p2", answer="x",
               judge={"correct": 1, "judge_model": "standin", "raw": ""}),
    ]
    lp = tmp_path / "answers.jsonl"
    write_jsonl(str(lp), answers)
    r = oracle.run(str(ds), runs, success_on="note", split="test", k=20,
                   bootstrap=200, seed=0, labels_path=str(lp))
    assert "oracle_minus_best_fixed" in r and "ci95" in r
    assert r["success_signal"] == "qa_correct"
    assert r["any_of_plans_upper_bound"] == 1.0
    # best fixed answers 1/2 correct; oracle answers both -> delta = 0.5
    assert r["oracle_minus_best_fixed"] == pytest.approx(0.5)
    assert isinstance(r["ci95"], list) and len(r["ci95"]) == 2


# ---- (e) A4 router fusion / confidence-select primitives ----
def test_a4_rrf_fuse_and_conf_select():
    from mem27.planner.a4_router import rrf_fuse, conf_select
    # fusion: a note ranked high by two plans should rank above a note ranked once
    r = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])
    assert r[0] in ("a", "b") and set(r) == {"a", "b", "c", "d"}

    class _B:
        def __init__(self, scores): self.scores = scores
    pb = {"ref:p0": _B([0.9, 0.1]), "ref:p1": _B([0.5, 0.45])}
    # p0 has the larger top1-top2 margin -> selected
    assert conf_select(pb) == "ref:p0"


# ---- (f) A5 weighted RRF fusion ----
def test_a5_weighted_rrf():
    from mem27.planner.a5_weighted_fusion import weighted_rrf
    # zero weight on a plan removes its influence
    rk = {"ref:p0": ["a", "b"], "ref:p1": ["c", "d"]}
    r = weighted_rrf(rk, {"ref:p0": 1.0, "ref:p1": 0.0})
    assert r[0] == "a"  # only p0 contributes
    r2 = weighted_rrf(rk, {"ref:p0": 1.0, "ref:p1": 1.0})
    assert set(r2) == {"a", "b", "c", "d"}
