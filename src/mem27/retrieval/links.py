"""Deterministic links between notes of one history: temporal, semantic, version, sibling (research R6).

No language model is involved, so link maintenance cost is measurable and rebuilds are bit-identical. Each note's
links are sorted by type priority (version > temporal > semantic > sibling), then target id; traversal relies on it.
"""

from __future__ import annotations

from pathlib import Path

from mem27.retrieval.dense import DenseRetriever
from mem27.schema import Link, Note, read_jsonl, write_jsonl

TYPE_PRIORITY = {"version": 0, "temporal": 1, "semantic": 2, "sibling": 3}


class LinkGraph:
    """Adjacency lists keyed by source note id."""

    def __init__(self, links_by_note: dict[str, list[Link]]):
        self._links = links_by_note

    def neighbours(self, note_id: str) -> list[Link]:
        return self._links.get(note_id, [])

    def as_dict(self) -> dict[str, list[dict]]:
        return {
            nid: [{"type": lk.type, "target_note_id": lk.target_note_id, "weight": lk.weight} for lk in links]
            for nid, links in sorted(self._links.items())
        }

    def __len__(self) -> int:
        return sum(len(links) for links in self._links.values())


class _Builder:
    def __init__(self) -> None:
        self.edges: dict[str, dict[tuple[str, str], float]] = {}

    def add(self, kind: str, source: str, target: str, weight: float = 1.0, both: bool = True) -> None:
        if source == target:
            return
        self.edges.setdefault(source, {}).setdefault((kind, target), weight)
        if both:
            self.edges.setdefault(target, {}).setdefault((kind, source), weight)

    def graph(self, note_ids: list[str]) -> LinkGraph:
        links = {}
        for nid in note_ids:
            items = sorted(self.edges.get(nid, {}).items(), key=lambda e: (TYPE_PRIORITY[e[0][0]], e[0][1]))
            links[nid] = [Link(type=kind, target_note_id=target, weight=w) for (kind, target), w in items]
        return LinkGraph(links)


def _sessions_in_time_order(notes: list[Note]) -> list[tuple[Note | None, list[Note]]]:
    """(parent, turns sorted by turn_idx) per session, ordered by session timestamp then id."""
    by_session: dict[str, list[Note]] = {}
    for note in notes:
        by_session.setdefault(note.session_id, []).append(note)
    ordered = []
    for session_id, members in by_session.items():
        parent = next((n for n in members if n.is_session_parent), None)
        turns = sorted((n for n in members if not n.is_session_parent), key=lambda n: n.turn_idx)
        ts = (parent or turns[0]).ts
        ordered.append((ts, session_id, parent, turns))
    ordered.sort(key=lambda item: (item[0], item[1]))
    return [(parent, turns) for _, _, parent, turns in ordered]


def _add_structural(builder: _Builder, sessions: list[tuple[Note | None, list[Note]]]) -> None:
    previous: tuple[Note | None, list[Note]] | None = None
    for parent, turns in sessions:
        for a, b in zip(turns, turns[1:], strict=False):
            builder.add("temporal", a.note_id, b.note_id)
        for i, a in enumerate(turns):
            for b in turns[i + 1 :]:
                builder.add("sibling", a.note_id, b.note_id)
            if parent is not None:
                builder.add("sibling", a.note_id, parent.note_id)
        if previous is not None:
            prev_parent, prev_turns = previous
            if prev_turns and turns:
                builder.add("temporal", prev_turns[-1].note_id, turns[0].note_id)
            if prev_parent is not None and parent is not None:
                builder.add("temporal", prev_parent.note_id, parent.note_id)
        previous = (parent, turns)


def build_links(notes: list[Note], dense: DenseRetriever, semantic_k: int = 5) -> LinkGraph:
    """Links for one history; `dense` must be built over the same history's turn notes."""
    by_id = {n.note_id: n for n in notes}
    builder = _Builder()
    _add_structural(builder, _sessions_in_time_order(notes))
    for note in notes:
        if note.supersedes_note_id is None:
            continue
        if note.supersedes_note_id not in by_id:
            raise ValueError(f"{note.note_id} supersedes missing note {note.supersedes_note_id}; fix the update stream")
        builder.add("version", note.note_id, note.supersedes_note_id)
    for note in notes:
        if note.is_session_parent:
            continue
        for target, cosine in dense.neighbours(note.note_id, semantic_k):
            builder.add("semantic", note.note_id, target, weight=cosine, both=False)
    return builder.graph(sorted(by_id))


def write_links(path: str | Path, graph: LinkGraph) -> int:
    return write_jsonl(path, ({"note_id": nid, "links": links} for nid, links in graph.as_dict().items()))


def load_links(path: str | Path) -> LinkGraph:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/build_indexes.py --plans links first")
    return LinkGraph({row["note_id"]: [Link.from_dict(lk) for lk in row["links"]] for row in read_jsonl(path)})
