"""BaseRetriever: the one place that times stages, enforces the token budget, and shapes evidence bundles.

Plans implement only `_build_index` and `_search`; everything a caller relies on (budget, latency, trace, the
index-built guard, persistence hooks) lives here so the six plans stay small and consistent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

from mem27.schema import EvidenceBundle, Note, RetrievalRequest, Summary, count_tokens, truncate_text

Ranked = list[tuple[str, float]]


def rank(scores: dict[str, float]) -> Ranked:
    """Highest score first; ties broken by note id so orderings are deterministic."""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class StageTimer:
    """Accumulates wall-clock milliseconds per named stage, plus the total since construction."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._start = perf_counter()

    @contextmanager
    def stage(self, name: str) -> Generator[None]:
        began = perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (perf_counter() - began) * 1000.0

    def total_ms(self) -> float:
        return (perf_counter() - self._start) * 1000.0


@dataclass
class Packed:
    note_ids: list[str]
    passages: list[str]
    scores: list[float]
    total_tokens: int
    truncated: bool
    stop_reason: str  # budget | max_results | exhausted


def pack_to_budget(
    note_ids: list[str], passages: list[str], scores: list[float], budget_tokens: int, max_results: int
) -> Packed:
    """Takes ranked evidence in order until the token budget or result cap is hit.

    Stops at the first passage that would overflow (no skipping ahead to smaller ones), except that an
    oversized top passage is truncated so a non-empty ranking never yields empty evidence.
    """
    packed = Packed([], [], [], 0, False, "exhausted")
    for nid, passage, score in zip(note_ids, passages, scores, strict=True):
        if len(packed.note_ids) >= max_results:
            packed.stop_reason = "max_results"
            break
        cost = count_tokens(passage)
        if packed.total_tokens + cost > budget_tokens:
            packed.stop_reason = "budget"
            if packed.note_ids:
                break
            passage = truncate_text(passage, budget_tokens)
            cost = count_tokens(passage)
            packed.truncated = True
        packed.note_ids.append(nid)
        packed.passages.append(passage)
        packed.scores.append(score)
        packed.total_tokens += cost
        if packed.truncated:
            break
    return packed


@dataclass
class SearchResult:
    ranked: Ranked
    stop_reason: str | None = None  # set when the search itself decides why it ended (e.g. no_match, cap)
    trace: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Template for every plan. Indexes cover one history; build or load before retrieving."""

    plan_id: ClassVar[str]

    def __init__(self) -> None:
        self._notes: dict[str, Note] = {}
        self._built = False

    @property
    def is_built(self) -> bool:
        return self._built

    def note(self, note_id: str) -> Note:
        return self._notes[note_id]

    def build(self, notes: Iterable[Note], summaries: Iterable[Summary] | None = None) -> None:
        note_list = list(notes)
        self._notes = {n.note_id: n for n in note_list}
        self._build_index(note_list, list(summaries) if summaries is not None else None)
        self._built = True

    def insert(self, note: Note) -> dict:
        """Contract method. Incremental maintenance is a later feature; for now the history index is rebuilt."""
        timer = StageTimer()
        with timer.stage("rebuild"):
            self.build([*self._notes.values(), note], None)
        return {"rebuild_ms": timer.stages["rebuild"]}

    def retrieve(self, req: RetrievalRequest) -> EvidenceBundle:
        self._require_built()
        timer = StageTimer()
        result = self._search(req, timer)
        with timer.stage("pack"):
            ids = [nid for nid, _ in result.ranked]
            packed = pack_to_budget(
                ids,
                [self._notes[nid].text for nid in ids],
                [score for _, score in result.ranked],
                req.budget_tokens,
                req.max_results,
            )
        stop_reason = result.stop_reason or ("no_match" if not ids else packed.stop_reason)
        return EvidenceBundle(
            note_ids=packed.note_ids,
            passages=packed.passages,
            scores=packed.scores,
            total_tokens=packed.total_tokens,
            latency_ms={"total": timer.total_ms(), "stages": dict(timer.stages)},
            stale_flags=[False] * len(packed.note_ids),
            trace={"plan": self.plan_id, "stop_reason": stop_reason, "truncated": packed.truncated, **result.trace},
            question_id=req.question_id,
            plan=self.plan_id,
        )

    def save(self, index_dir: str | Path) -> None:
        self._require_built()
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        self._save(Path(index_dir))

    def load(self, index_dir: str | Path, notes: Iterable[Note]) -> None:
        """Restores a saved index; `notes` supplies passage text so it is not duplicated on disk."""
        self._notes = {n.note_id: n for n in notes}
        self._load(Path(index_dir))
        self._built = True

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError(f"index not built: call build() or load() on {type(self).__name__} first")

    @abstractmethod
    def _build_index(self, notes: list[Note], summaries: list[Summary] | None) -> None: ...

    @abstractmethod
    def _search(self, req: RetrievalRequest, timer: StageTimer) -> SearchResult: ...

    def _save(self, index_dir: Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} has no persisted state")

    def _load(self, index_dir: Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} has no persisted state")


def turn_notes(notes: Iterable[Note]) -> list[Note]:
    """Turn-level notes only, sorted by id; session parents are hierarchy, not evidence (research R2)."""
    return sorted((n for n in notes if not n.is_session_parent), key=lambda n: n.note_id)
