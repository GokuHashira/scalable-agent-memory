"""Bounded traversal: hybrid anchors expanded breadth-first with a hop budget and a candidate cap (mirrors All-Mem)."""

from __future__ import annotations

from mem27.retrieval.hybrid import HybridRetriever
from mem27.retrieval.links import LinkGraph
from mem27.retrieval.traversal import AnchoredTraversal


class BoundedTraversalRetriever(AnchoredTraversal):
    plan_id = "p5"

    def __init__(
        self,
        hybrid: HybridRetriever,
        links: LinkGraph | None = None,
        anchors: int = 5,
        hops: int = 4,
        cap: int = 50,
        semantic_k: int = 5,
    ):
        super().__init__(hybrid, links, anchors, semantic_k)
        self.hops, self.cap = hops, cap

    def _step(
        self, frontier: list[str], visited: set[str], candidates: list[str], budget: int
    ) -> tuple[list[str], bool]:
        """One BFS layer in link order; returns the next frontier and whether the candidate cap was hit."""
        next_frontier: list[str] = []
        for node in frontier:
            for link in self.links.neighbours(node):
                if link.target_note_id in visited:
                    continue
                visited.add(link.target_note_id)
                next_frontier.append(link.target_note_id)
                candidates.append(link.target_note_id)
                if len(candidates) >= budget:
                    return next_frontier, True
        return next_frontier, False

    def _expand(self, anchors: list[str]) -> tuple[list[str], dict]:
        visited, frontier, candidates = set(anchors), list(anchors), []
        hops_used, stop_reason = 0, "hops"
        for hop in range(1, self.hops + 1):
            if not frontier:
                stop_reason = "frontier_empty"
                break
            frontier, hit_cap = self._step(frontier, visited, candidates, self.cap)
            hops_used = hop
            if hit_cap:
                stop_reason = "cap"
                break
        return candidates, {"hops_used": hops_used, "candidates_visited": len(candidates), "stop_reason": stop_reason}
