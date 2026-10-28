"""DATA-02: same seed -> identical corpus; gold ids resolve; distractor tiers present; leak-free."""
import hashlib
from collections import Counter

import pytest

from mem27.data.synth import SynthConfig, generate
from mem27.vschema import iter_jsonl


def _digest(d):
    h = hashlib.sha256()
    for f in ("notes.jsonl", "updates.jsonl", "questions.jsonl", "questions_stream.jsonl"):
        h.update((d / f).read_bytes())
    return h.hexdigest()


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    cfg = SynthConfig(name="t2k", n_notes=2000, n_questions=120, n_stream_questions=60)
    stats = generate(cfg, root)
    return root, cfg, stats


def test_deterministic(world, tmp_path):
    root, cfg, _ = world
    generate(SynthConfig(name="t2k", n_notes=2000, n_questions=120, n_stream_questions=60), tmp_path)
    assert _digest(root / "t2k") == _digest(tmp_path / "t2k")
    generate(SynthConfig(name="t2k", n_notes=2000, n_questions=120, n_stream_questions=60, seed=1), tmp_path / "s1")
    assert _digest(root / "t2k") != _digest(tmp_path / "s1" / "t2k")


def test_counts_and_gold_resolve(world):
    root, cfg, stats = world
    d = root / "t2k"
    final = {n["note_id"]: n for n in iter_jsonl(d / "notes_final.jsonl")}
    assert len(final) == stats["n_notes_total"] == cfg.n_notes
    base = list(iter_jsonl(d / "notes.jsonl"))
    ups = list(iter_jsonl(d / "updates.jsonl"))
    assert len(base) + len(ups) == len(final)
    for qf in ("questions.jsonl", "questions_stream.jsonl"):
        for q in iter_jsonl(d / qf):
            assert q["gold_note_ids"], q["question_id"]
            for g in q["gold_note_ids"]:
                assert g in final and final[g]["ts"] <= q["question_date"]
            for s in q["fact"].get("superseded_note_ids", []):
                assert s in final and s not in q["gold_note_ids"]


def test_distractor_tiers_present(world):
    root, _, _ = world
    kinds = Counter(r["kind"] for r in iter_jsonl(root / "_audit" / "t2k" / "note_facts.jsonl"))
    for k in ("filler", "event", "initial", "restate", "supersede", "correct"):
        assert kinds[k] > 0, k
    qs = list(iter_jsonl(root / "t2k" / "questions.jsonl"))
    assert any(q["fact"].get("twin") for q in qs), "near-duplicate (twin) tier missing"
    assert any(q["fact"]["superseded_note_ids"] for q in qs), "superseded tier missing"
    assert {q["question_type"] for q in qs} == {"exact_lookup", "semantic_recall", "latest_value",
                                                "point_in_time", "what_changed", "multi_hop"}


def test_leak_free(world):
    root, _, stats = world
    assert stats["chance_accuracy_static"] < 0.05
    for n in iter_jsonl(root / "t2k" / "notes.jsonl"):
        assert not n["metadata"], "no ground truth in note metadata"


def test_mix_shift(world):
    root, _, _ = world
    qs = list(iter_jsonl(root / "t2k" / "questions_stream.jsonl"))
    half = len(qs) // 2
    first = Counter(q["question_type"] for q in qs[:half])
    second = Counter(q["question_type"] for q in qs[half:])
    assert first["latest_value"] > first["point_in_time"] and second["point_in_time"] > second["latest_value"]
