"""RefDense / RefHybrid (streaming reference retrievers) with a fake embedding cache (no model download)."""
import hashlib

import numpy as np

from mem27.eval.ref_dense import RefDense, RefHybrid
from mem27.vschema import Note, RetrievalRequest


class FakeEmb:
    """Bag-of-words hashing vectors: texts sharing words get high cosine."""

    def _vec(self, text):
        v = np.zeros(768, np.float32)
        for w in text.lower().replace("?", "").replace(".", "").split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 768] += 1
        return (v / (np.linalg.norm(v) or 1)).astype(np.float16)

    def get(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode(self, texts):
        return np.stack([self._vec(t.split(": ", 1)[-1]) for t in texts])


def note(i, text, ts):
    return Note(note_id=f"n{i}", history_id="h", session_id="s", turn_idx=i, role="user", text=text, ts=ts)


def test_dense_ranks_and_versions():
    r = RefDense(FakeEmb(), version_filter="explicit")
    r.build([note(1, "Zorb works at Acme Labs", "2024-01-01T00:00:00"),
             note(2, "Quux lives in Paris", "2024-01-02T00:00:00")])
    up = {"op": "supersede", "explicit": True, "supersedes_note_id": "n1", "ts": "2024-02-01T00:00:00",
          "note": {"note_id": "n3", "history_id": "h", "session_id": "s", "turn_idx": 3, "role": "user",
                   "text": "Zorb works at Beta Labs now", "ts": "2024-02-01T00:00:00"}}
    r.apply_update(up)
    b = r.retrieve(RetrievalRequest(query="Where does Zorb work?", as_of_time="2024-03-01T00:00:00"))
    assert b.note_ids[0] == "n3" and "n1" not in b.note_ids  # explicit supersession hides the old version
    b_old = r.retrieve(RetrievalRequest(query="Where does Zorb work?", as_of_time="2024-01-15T00:00:00"))
    assert "n3" not in b_old.note_ids and "n1" in b_old.note_ids  # as_of_time: the update is not visible yet


def test_hybrid_fuses_and_filters():
    r = RefHybrid(FakeEmb(), version_filter="none", recency_boost=0.5)
    r.build([note(1, "Zorb works at Acme Labs", "2024-01-01T00:00:00"),
             note(2, "Quux lives in Paris", "2024-01-02T00:00:00"),
             note(3, "Zorb works at Beta Labs", "2024-02-01T00:00:00")])
    b = r.retrieve(RetrievalRequest(query="Where does Zorb work?", as_of_time="2024-03-01T00:00:00"))
    assert set(b.note_ids[:2]) == {"n1", "n3"}
    assert b.note_ids[0] == "n3"  # same lexical/dense evidence; the recency prior breaks the tie toward the newer note
    assert b.total_tokens > 0 and len(b.note_ids) <= 20
