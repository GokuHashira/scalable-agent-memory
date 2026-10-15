"""Lexical BM25 over the turn notes of one history (research R3)."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from mem27.retrieval.base import BaseRetriever, Ranked, SearchResult, StageTimer, rank, turn_notes
from mem27.schema import Note, RetrievalRequest, Summary

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; no stemming or stopwords, so behaviour is easy to reason about."""
    return _TOKEN.findall(text.lower())


class BM25Retriever(BaseRetriever):
    plan_id = "p0"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__()
        self.k1, self.b = k1, b
        self._ids: list[str] = []
        self._bm25: BM25Okapi | None = None

    def _build_index(self, notes: list[Note], summaries: list[Summary] | None) -> None:
        corpus_notes = turn_notes(notes)
        if not corpus_notes:
            raise ValueError("BM25Retriever.build: no turn notes in this history; check the converted notes.jsonl")
        self._ids = [n.note_id for n in corpus_notes]
        self._bm25 = BM25Okapi([tokenize(n.text) for n in corpus_notes], k1=self.k1, b=self.b)

    def search(self, query: str, depth: int) -> Ranked:
        """Top `depth` notes with a positive score; an empty list means no query term matched."""
        assert self._bm25 is not None, "search() called before build()/load()"
        terms = tokenize(query)
        if not terms:
            return []
        scores = self._bm25.get_scores(terms)
        top = np.flatnonzero(scores > 0)
        return rank({self._ids[i]: float(scores[i]) for i in top})[:depth]

    def _search(self, req: RetrievalRequest, timer: StageTimer) -> SearchResult:
        with timer.stage("bm25"):
            ranked = self.search(req.query, req.max_results)
        return SearchResult(ranked=ranked, stop_reason=None if ranked else "no_match")

    def _save(self, index_dir: Path) -> None:
        with (index_dir / "bm25.pkl").open("wb") as out:
            pickle.dump({"ids": self._ids, "bm25": self._bm25, "k1": self.k1, "b": self.b}, out)

    def _load(self, index_dir: Path) -> None:
        path = index_dir / "bm25.pkl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/build_indexes.py --plans bm25 first")
        with path.open("rb") as src:
            state = pickle.load(src)
        self._ids, self._bm25, self.k1, self.b = state["ids"], state["bm25"], state["k1"], state["b"]
