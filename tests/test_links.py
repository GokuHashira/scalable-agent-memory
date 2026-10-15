"""Deterministic link building (FR-007, research R6)."""

import pytest

from mem27.retrieval.dense import DenseRetriever
from mem27.retrieval.links import LinkGraph, build_links, load_links, write_links
from mem27.schema import Note


@pytest.fixture(scope="module")
def history(notes_by_history):
    return notes_by_history["fx000"]


@pytest.fixture(scope="module")
def graph(history, encoder):
    dense = DenseRetriever(encoder)
    dense.build(history, None)
    return build_links(history, dense, semantic_k=5)


def _targets(graph, nid, kind):
    return [link.target_note_id for link in graph.neighbours(nid) if link.type == kind]


def test_no_self_links_and_same_history(graph, history):
    ids = {n.note_id for n in history}
    for nid in ids:
        for link in graph.neighbours(nid):
            assert link.target_note_id != nid
            assert link.target_note_id in ids


def test_semantic_at_most_k_with_cosine_weights(graph, history):
    for note in history:
        semantic = [lk for lk in graph.neighbours(note.note_id) if lk.type == "semantic"]
        if note.is_session_parent:
            assert semantic == []
            continue
        assert 0 < len(semantic) <= 5
        assert all(-1.0001 <= lk.weight <= 1.0001 for lk in semantic)


def test_temporal_links(graph):
    assert set(_targets(graph, "fx000:s0:3", "temporal")) == {"fx000:s0:2", "fx000:s0:4"}
    assert "fx000:s1:0" in _targets(graph, "fx000:s0:9", "temporal")  # last turn → next session's first
    assert "fx000:s0:9" in _targets(graph, "fx000:s1:0", "temporal")
    assert set(_targets(graph, "fx000:s1:-1", "temporal")) == {"fx000:s0:-1", "fx000:s2:-1"}


def test_sibling_links(graph):
    siblings = set(_targets(graph, "fx000:s0:3", "sibling"))
    assert siblings == {f"fx000:s0:{t}" for t in range(10) if t != 3} | {"fx000:s0:-1"}
    assert "fx000:s0:3" in _targets(graph, "fx000:s0:-1", "sibling")


def test_links_sorted_by_type_priority_then_target(graph):
    order = {"version": 0, "temporal": 1, "semantic": 2, "sibling": 3}
    links = graph.neighbours("fx000:s3:5")
    keys = [(order[lk.type], lk.target_note_id) for lk in links]
    assert keys == sorted(keys)


def test_version_links_follow_supersedes(encoder):
    old = Note("h:s0:0", "h", "s0", 0, "user", "my car is red", "2023-01-01")
    new = Note("h:s1:0", "h", "s1", 0, "user", "my car is blue now", "2023-02-01", supersedes_note_id="h:s0:0")
    dense = DenseRetriever(encoder)
    dense.build([old, new], None)
    graph = build_links([old, new], dense, semantic_k=5)
    assert _targets(graph, "h:s1:0", "version") == ["h:s0:0"]
    assert _targets(graph, "h:s0:0", "version") == ["h:s1:0"]


def test_missing_supersedes_target_raises(encoder):
    new = Note("h:s1:0", "h", "s1", 0, "user", "x", "2023-02-01", supersedes_note_id="h:s0:9")
    dense = DenseRetriever(encoder)
    dense.build([new], None)
    with pytest.raises(ValueError, match="supersedes missing note h:s0:9"):
        build_links([new], dense, semantic_k=5)


def test_rebuild_identical_and_roundtrip(graph, history, encoder, tmp_path):
    dense = DenseRetriever(encoder)
    dense.build(history, None)
    assert build_links(history, dense, semantic_k=5).as_dict() == graph.as_dict()
    write_links(tmp_path / "links.jsonl", graph)
    assert load_links(tmp_path / "links.jsonl").as_dict() == graph.as_dict()
    assert isinstance(graph, LinkGraph)
