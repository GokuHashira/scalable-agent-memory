"""Shared machinery for P4/P5: expand P2 anchors along links, then re-rank by similarity (research R8).

Session parents can be traversed through but are never returned as evidence. Results keep the anchors first (in
hybrid order), followed by expanded candidates ordered by cosine to the query; every score is that cosine.
"""

from __future__ import annotations

from mem27.retrieval.base import BaseRetriever, SearchResult, StageTimer, rank
from mem27.retrieval.hybrid import HybridRetriever
from mem27.retrieval.links import LinkGraph, build_links
from mem27.schema import Note, RetrievalRequest, Summary


class AnchoredTraversal(BaseRetriever):
    """Shared anchor selection and re-ranking; subclasses decide how far to expand."""

    def __init__(self, hybrid: HybridRetriever, links: LinkGraph | None, anchors: int, semantic_k: int = 5):
        super().__init__()
        self.hybrid = hybrid
        self.links = links if links is not None else LinkGraph({})
        self.n_anchors = anchors
        self.semantic_k = semantic_k
        self._has_links = links is not None

    def _build_index(self, notes: list[Note], summaries: list[Summary] | None) -> None:
        if not self.hybrid.is_built:
            self.hybrid.build(notes, None)
        if not self._has_links:
            self.links = build_links(notes, self.hybrid.dense, self.semantic_k)
            self._has_links = True

    def _is_evidence(self, note_id: str) -> bool:
        note = self._notes.get(note_id)
        return note is not None and not note.is_session_parent

    def _expand(self, anchors: list[str]) -> tuple[list[str], dict]:
        raise NotImplementedError

    def _search(self, req: RetrievalRequest, timer: StageTimer) -> SearchResult:
        fused, qvec = self.hybrid.fuse(req.query, timer)
        anchors = [nid for nid, _ in fused[: self.n_anchors]]
        if not anchors:
            return SearchResult(ranked=[], stop_reason="no_match", trace={"anchors": []})
        with timer.stage("expand"):
            candidates, trace = self._expand(anchors)
        with timer.stage("rerank"):
            anchor_scores = self.hybrid.dense.similarity(qvec, anchors)
            ranked = [(a, anchor_scores.get(a, 0.0)) for a in anchors]
            ranked += rank(self.hybrid.dense.similarity(qvec, [c for c in candidates if self._is_evidence(c)]))
        return SearchResult(
            ranked=ranked, stop_reason=trace.pop("stop_reason", None), trace={"anchors": anchors, **trace}
        )
