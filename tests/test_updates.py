"""DATA-03: sequential order; explicitly superseded notes get valid_to; explicit flag set; stream harness."""
import pytest

from mem27.data.synth import SynthConfig, generate
from mem27.data.updates import NoteStore, iter_updates, stream_eval, to_note
from mem27.eval.freshness import fresh_row, synth_freshness
from mem27.eval.ref_retriever import RefBM25
from mem27.vschema import iter_jsonl


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("upd")
    generate(SynthConfig(name="u2k", n_notes=2000, n_questions=60, n_stream_questions=60, update_rate="high"), root)
    return root / "u2k"


def test_sequential_and_valid_to(world):
    store = NoteStore(to_note(n) for n in iter_jsonl(world / "notes.jsonl"))
    prev, n_exp = "", 0
    for u in iter_updates(world / "updates.jsonl"):
        assert u["ts"] >= prev
        prev = u["ts"]
        tr = store.apply_update(u)
        if u["op"] in ("supersede", "correct"):
            assert isinstance(u["explicit"], bool)
            if u["explicit"]:
                n_exp += 1
                assert u["supersedes_note_id"] and store.notes[u["supersedes_note_id"]].valid_to == u["ts"]
                assert tr["closed"] == u["supersedes_note_id"]
            else:
                assert u["supersedes_note_id"] is None
    assert n_exp > 0


def test_out_of_order_rejected(world):
    ups = list(iter_jsonl(world / "updates.jsonl"))
    store = NoteStore()
    store.apply_update(ups[5])
    with pytest.raises(ValueError):
        store.apply_update(ups[0])


def test_stream_eval_and_freshness(world, tmp_path):
    run = tmp_path / "run"
    info = stream_eval(RefBM25(version_filter="explicit"), world, run, plan="ref_bm25")
    assert info["n_questions"] == 60 and info["n_updates_fed"] > 0
    res = synth_freshness(run, world, "questions_stream.jsonl")
    assert res["groups"]["ALL"]["n"] == 60
    for b in iter_jsonl(run / "bundles.jsonl"):
        assert b["total_tokens"] <= 2048 and len(b["note_ids"]) <= 20


def test_fresh_row():
    r = fresh_row(["a", "old", "new"], {"new"}, {"old"})
    assert r["fresh@5"] == 0.0 and r["stale_in@5"] == 1.0
    r = fresh_row(["new", "x", "old"], {"new"}, {"old"})
    assert r["fresh@1"] == 1.0
    r = fresh_row(["x", "new"], {"new"}, {"old"})
    assert r["fresh@1"] == 0.0 and r["fresh@5"] == 1.0
