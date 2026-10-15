"""Dense retrieval with an exact inner-product index over normalised embeddings (research R4)."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from mem27.retrieval.base import BaseRetriever, Ranked, SearchResult, StageTimer, rank, turn_notes
from mem27.retrieval.encoder import Encoder
from mem27.schema import Note, RetrievalRequest, Summary


class DenseRetriever(BaseRetriever):
    """Also serves cosine similarity to other plans (summary_first, one_hop and bounded re-rank by it).

    Embeddings are therefore computed once and shared.
    """

    plan_id = "p1"

    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder
        self._ids: list[str] = []
        self._row: dict[str, int] = {}
        self._emb = np.zeros((0, 0), dtype=np.float32)
        self._index: faiss.IndexFlatIP | None = None

    def _set_embeddings(self, ids: list[str], emb: np.ndarray) -> None:
        # Round-trip through fp16 so an in-memory build equals a reload from the fp16 cache (determinism, FR-014).
        self._emb = np.ascontiguousarray(emb.astype(np.float16).astype(np.float32))
        self._ids = ids
        self._row = {nid: i for i, nid in enumerate(ids)}
        self._index = faiss.IndexFlatIP(self._emb.shape[1])
        self._index.add(self._emb)

    def _build_index(self, notes: list[Note], summaries: list[Summary] | None) -> None:
        corpus = turn_notes(notes)
        if not corpus:
            raise ValueError("DenseRetriever.build: no turn notes in this history; check the converted notes.jsonl")
        self._set_embeddings([n.note_id for n in corpus], self.encoder.encode_passages([n.text for n in corpus]))

    def query_vector(self, query: str) -> np.ndarray:
        return self.encoder.encode_queries([query])[0]

    def search_vector(self, qvec: np.ndarray, depth: int) -> Ranked:
        assert self._index is not None, "search_vector() called before build()/load()"
        k = min(depth, len(self._ids))
        if k == 0:
            return []
        scores, rows = self._index.search(qvec.reshape(1, -1).astype(np.float32), k)
        return rank({self._ids[r]: float(s) for s, r in zip(scores[0], rows[0], strict=True) if r >= 0})

    def similarity(self, qvec: np.ndarray, note_ids: list[str]) -> dict[str, float]:
        """Cosine to the query for notes in this index; ids not indexed (e.g. session parents) are skipped."""
        rows = [self._row[nid] for nid in note_ids if nid in self._row]
        sims = self._emb[rows] @ qvec.astype(np.float32) if rows else np.zeros(0)
        return {self._ids[r]: float(s) for r, s in zip(rows, sims, strict=True)}

    def neighbours(self, note_id: str, k: int) -> Ranked:
        """Top-k most similar indexed notes to `note_id`, excluding itself."""
        ranked = self.search_vector(self._emb[self._row[note_id]], k + 1)
        return [(nid, s) for nid, s in ranked if nid != note_id][:k]

    def _search(self, req: RetrievalRequest, timer: StageTimer) -> SearchResult:
        with timer.stage("encode_query"):
            qvec = self.query_vector(req.query)
        with timer.stage("dense"):
            ranked = self.search_vector(qvec, req.max_results)
        return SearchResult(ranked=ranked)

    def _save(self, index_dir: Path) -> None:
        np.save(index_dir / "emb.npy", self._emb.astype(np.float16))
        (index_dir / "ids.json").write_text(json.dumps({"ids": self._ids, "encoder": self.encoder.name}))

    def _load(self, index_dir: Path) -> None:
        emb_path, ids_path = index_dir / "emb.npy", index_dir / "ids.json"
        if not emb_path.exists() or not ids_path.exists():
            raise FileNotFoundError(f"{index_dir} lacks emb.npy/ids.json; run scripts/build_indexes.py --plans dense")
        meta = json.loads(ids_path.read_text())
        if meta["encoder"] != self.encoder.name:
            raise ValueError(
                f"{index_dir} was built with encoder {meta['encoder']!r} but {self.encoder.name!r} is loaded; "
                "rebuild the index or pass the matching --encoder"
            )
        self._set_embeddings(meta["ids"], np.load(emb_path).astype(np.float32))
