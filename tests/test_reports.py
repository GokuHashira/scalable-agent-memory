"""Index cost table, bundle comparison (determinism), and failure-trace candidates."""

import json

from mem27.retrieval.reports import compare_bundle_files, failure_candidates, index_table
from mem27.schema import Question, write_jsonl


def _bundle(qid, ids, scores=None, total=1.0):
    return {
        "question_id": qid,
        "plan": "p2",
        "note_ids": ids,
        "passages": ["x"] * len(ids),
        "scores": scores or [1.0] * len(ids),
        "total_tokens": 10,
        "latency_ms": {"total": total, "stages": {}},
        "trace": {"stop_reason": "budget"},
    }


def test_compare_ignores_timing_but_not_ranking(tmp_path):
    a, b, c = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    write_jsonl(a, [_bundle("q1", ["h:s:1", "h:s:2"], total=1.0)])
    write_jsonl(b, [_bundle("q1", ["h:s:1", "h:s:2"], total=9.0)])
    write_jsonl(c, [_bundle("q1", ["h:s:2", "h:s:1"])])
    assert compare_bundle_files(a, b) == {"compared": 1, "identical": 1, "mismatched": []}
    assert compare_bundle_files(a, c)["mismatched"] == ["q1"]


def test_index_table_totals_and_projection(tmp_path):
    profile = {
        "n_turn_notes": 1000,
        "totals": {
            "bm25": {"histories": 2, "build_ms": 2000.0, "disk_bytes": 2_000_000, "peak_rss_mb": 100.0},
            "dense": {"histories": 2, "build_ms": 10000.0, "disk_bytes": 4_000_000, "peak_rss_mb": 200.0},
        },
    }
    path = tmp_path / "build_profile.json"
    path.write_text(json.dumps(profile))
    md = index_table(path, targets=(100_000, 1_000_000))
    assert "| bm25 | 2 | 2.0 | 2.0 | 100 |" in md
    assert "200.0 s" in md and "2000.0 s" in md  # bm25 build time projected linearly to 100K and 1M turns
    assert "linear" in md.lower()


def test_failure_candidates_find_plan_disagreements():
    qs = [
        Question("q1", "q1", "temporal-reasoning", "when?", "d", "a", ["s1"], ["q1:s1:0"]),
        Question("q2", "q2", "multi-session", "what?", "d", "a", ["s1", "s2"], ["q2:s1:0", "q2:s2:0"]),
    ]
    runs = {
        "bm25": [_bundle("q1", ["q1:s1:0"]), _bundle("q2", ["q2:s9:0"])],
        "dense": [_bundle("q1", ["q1:s7:0"]), _bundle("q2", ["q2:s1:0", "q2:s2:0"])],
    }
    found = failure_candidates(runs, qs, k=20)
    assert ("q1", "dense") in {(c["question_id"], c["missed_by"]) for c in found}
    assert any(c["question_id"] == "q2" and c["missed_by"] == "bm25" for c in found)
    first = next(c for c in found if c["question_id"] == "q1")
    assert first["found_by"] == ["bm25"] and first["question_type"] == "temporal-reasoning"
