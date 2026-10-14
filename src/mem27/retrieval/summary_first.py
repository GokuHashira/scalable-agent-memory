"""P3: find the best sessions by their summaries, then return those sessions' turns ranked by similarity (R7)."""

from __future__ import annotations

from pathlib import Path

from mem27.retrieval import ids
from mem27.retrieval.base import BaseRetriever, SearchResult, StageTimer, rank, turn_notes
from mem27.retrieval.dense import DenseRetriever
from mem27.retrieval.hybrid import HybridRetriever
from mem27.retrieval.summaries import SUMMARY_KIND, as_pseudo_notes
from mem27.schema import Note, RetrievalRequest, Summary, load_summaries, write_jsonl


class SummaryFirstRetriever(BaseRetriever):
    plan_id = "p3"

    def __init__(self, dense: DenseRetriever, summary_index: HybridRetriever, top_sessions: int = 5):
        super().__init__()
        self.dense = dense  # turn-level embeddings (shared with P1)
        self.summary_index = summary_index  # BM25 + dense over summary texts
        self.top_sessions = top_sessions
        self._summaries: list[Summary] = []
        self._turns_by_session: dict[str, list[str]] = {}

    def _index_turns(self, notes: list[Note]) -> None:
        self._turns_by_session = {}
        for note in turn_notes(notes):
            self._turns_by_session.setdefault(ids.session_key(note.history_id, note.session_id), []).append(
                note.note_id
            )

    def _build_index(self, notes: list[Note], summaries: list[Summary] | None) -> None:
        if not summaries:
            raise ValueError("SummaryFirstRetriever.build needs summaries; run build_indexes.py --plans p3 first")
        if not self.dense.is_built:
            self.dense.build(notes, None)
        self._summaries = sorted(summaries, key=lambda s: s.summary_id)
        if not self.summary_index.is_built:
            self.summary_index.build(as_pseudo_notes(self._summaries), None)
        self._index_turns(notes)

    def _search(self, req: RetrievalRequest, timer: StageTimer) -> SearchResult:
        ranked_summaries, qvec = self.summary_index.fuse(req.query, timer, prefix="summary_")
        sessions = [ids.session_key_of_note(sid) for sid, _ in ranked_summaries[: self.top_sessions]]
        with timer.stage("expand"):
            candidates = [nid for key in sessions for nid in self._turns_by_session.get(key, [])]
        with timer.stage("rank"):
            ranked = rank(self.dense.similarity(qvec, candidates))
        return SearchResult(ranked=ranked, trace={"summary_kind": SUMMARY_KIND, "sessions": sessions})

    def _save(self, index_dir: Path) -> None:
        write_jsonl(index_dir / "summaries.jsonl", self._summaries)
        self.summary_index.save(index_dir / "index")

    def _load(self, index_dir: Path) -> None:
        notes = list(self._notes.values())
        self._summaries = load_summaries(index_dir / "summaries.jsonl")
        self.summary_index.load(index_dir / "index", as_pseudo_notes(self._summaries))
        self._index_turns(notes)
