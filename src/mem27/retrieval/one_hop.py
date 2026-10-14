"""P4: hybrid anchors expanded by exactly one hop over all link types."""

from __future__ import annotations

from mem27.retrieval.hybrid import HybridRetriever
from mem27.retrieval.links import LinkGraph
from mem27.retrieval.traversal import AnchoredTraversal


class OneHopRetriever(AnchoredTraversal):
    plan_id = "p4"

    def __init__(self, hybrid: HybridRetriever, links: LinkGraph | None = None, anchors: int = 5, semantic_k: int = 5):
        super().__init__(hybrid, links, anchors, semantic_k)

    def _expand(self, anchors: list[str]) -> tuple[list[str], dict]:
        seen = set(anchors)
        candidates = []
        for anchor in anchors:
            for link in self.links.neighbours(anchor):
                if link.target_note_id not in seen:
                    seen.add(link.target_note_id)
                    candidates.append(link.target_note_id)
        return candidates, {"hops_used": 1, "candidates_visited": len(candidates)}
