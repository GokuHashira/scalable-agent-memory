"""P2: reciprocal-rank fusion of P0 and P1 (research R5). Also the anchor source for P4/P5."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mem27.retrieval.base import BaseRetriever, Ranked, SearchResult, StageTimer, rank
from mem27.retrieval.p0_bm25 import BM25Retriever
from mem27.retrieval.p1_dense import DenseRetriever
from mem27.schema import Note, RetrievalRequest, Summary


def rrf(rankings: list[Ranked], k: int) -> Ranked:
    """score(d) = Σ 1 / (k + rank_i(d)), ranks 1-based; ties broken by note id."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, (nid, _) in enumerate(ranking, start=1):
            fused[nid] = fused.get(nid, 0.0) + 1.0 / (k + position)
    return rank(fused)


class HybridRetriever(BaseRetriever):
    plan_id = "p2"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60, depth: int = 50):
        super().__init__()
        self.bm25, self.dense = bm25, dense
        self.rrf_k, self.depth = rrf_k, depth

    def _build_index(self, notes: list[Note], summaries: list[Summary] | None) -> None:
        for component in (self.bm25, self.dense):
            if not component.is_built:
                component.build(notes, summaries)

    def fuse(self, query: str, timer: StageTimer) -> tuple[Ranked, np.ndarray]:
        """Full fused ranking (up to 2×depth notes) and the query vector, reused by plans that re-rank."""
        with timer.stage("bm25"):
            lexical = self.bm25.search(query, self.depth)
        with timer.stage("encode_query"):
            qvec = self.dense.query_vector(query)
        with timer.stage("dense"):
            semantic = self.dense.search_vector(qvec, self.depth)
        with timer.stage("fuse"):
            fused = rrf([lexical, semantic], self.rrf_k)
        return fused, qvec

    def _search(self, req: RetrievalRequest, timer: StageTimer) -> SearchResult:
        fused, _ = self.fuse(req.query, timer)
        return SearchResult(ranked=fused[: req.max_results])

    def _save(self, index_dir: Path) -> None:
        self.bm25.save(index_dir / "bm25")
        self.dense.save(index_dir / "dense")

    def _load(self, index_dir: Path) -> None:
        notes = list(self._notes.values())
        self.bm25.load(index_dir / "bm25", notes)
        self.dense.load(index_dir / "dense", notes)
