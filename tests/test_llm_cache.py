"""Best-effort LLM cache and reader resume (Oct 13 fix after the shared-cache corruption)."""
import json
import os
import time

from mem27.eval.llm import Cache
from mem27.eval.reader import _reusable_rows


def test_cache_roundtrip(tmp_path):
    c = Cache(tmp_path / "c.sqlite")
    c.put("k", {"text": "hi"})
    assert c.get("k") == {"text": "hi"}
    assert c.get("missing") is None
    assert not c.broken


def test_cache_corrupt_file_degrades_to_no_cache(tmp_path):
    p = tmp_path / "bad.sqlite"
    p.write_bytes(b"this is not a sqlite database" * 100)
    c = Cache(p)  # must not raise
    assert c.get("k") is None
    c.put("k", {"text": "x"})  # must not raise
    assert c.broken
    assert c.get("k") is None


def _row(qid, sha="S", err=None, model="m", order="chrono", msg_sha=None):
    rd = {"model": model, "prompt_sha": sha, "order": order, "think": False, "date_hints": False, "error": err}
    if msg_sha:
        rd["msg_sha"] = msg_sha
    return {"question_id": qid, "answer": "a", "response": "r", "reader": rd, "judge": None}


def test_reusable_rows_filters(tmp_path):
    (tmp_path / "bundles.jsonl").write_text("{}\n")
    old = time.time() - 100
    os.utime(tmp_path / "bundles.jsonl", (old, old))
    rows = [_row("ok"), _row("err", err="Timeout"), _row("othersha", sha="T"), _row("othermodel", model="x"),
            _row("rank", order="rank")]
    f = tmp_path / "answers.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    keep = _reusable_rows(f, "S", "chrono", False, False, "m")
    assert set(keep) == {"ok"}


def test_reusable_rows_refuses_when_bundles_newer(tmp_path):
    f = tmp_path / "answers.jsonl"
    f.write_text(json.dumps(_row("ok")) + "\n")
    old = time.time() - 100
    os.utime(f, (old, old))
    (tmp_path / "bundles.jsonl").write_text("{}\n")  # newer than the answers
    assert _reusable_rows(f, "S", "chrono", False, False, "m") == {}


def test_endpoint_directory_refresh(tmp_path):
    from mem27.eval.llm import ChatLLM
    d = tmp_path / "ep"
    d.mkdir()
    (d / "a.txt").write_text("http://host-a:1/v1\n")
    llm = ChatLLM(str(d), model="m", refresh_s=0.0)  # model given: no network call
    assert llm.urls == ["http://host-a:1/v1"]
    (d / "b.txt").write_text("http://host-b:2/v1\n")
    (d / "a.txt").unlink()
    with llm.lb_lock:
        llm._refresh()
    assert llm.urls == ["http://host-a:1/v1", "http://host-b:2/v1"]
    assert llm.dead == {0} and len(llm.clients) == 2 and llm.inflight == [0, 0]


def test_chat_cache_hit_path(tmp_path):
    """Exercises ChatLLM.chat() without a server: a cache hit must return before any network call
    (guards against attribute/method name clashes such as the Oct 14 `_key` shadowing bug)."""
    from mem27.eval.llm import ChatLLM
    llm = ChatLLM("http://unused:1/v1", model="m", cache=tmp_path / "c.sqlite")
    msgs = [{"role": "user", "content": "hi"}]
    params = {"temperature": 0.0, "max_tokens": 8, "seed": 0}
    llm.cache.put(llm._key(msgs, params), {"text": "cached", "model": "m"})
    out = llm.chat(msgs, max_tokens=8)
    assert out["text"] == "cached" and out["cached"] is True
