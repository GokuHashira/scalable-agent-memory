# STAND-IN — replace with the shared schema.py verbatim when it lands in this repo.
# Mirrors the shared record contract field for field. Do not extend it unilaterally
# (constitution III); tests/test_contract.py pins the field names so a swap to the shared file surfaces any drift.
"""Shared records, retriever protocol, and token-budget helpers used by every workstream."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, Self, cast

TOKENIZER_NAME = "cl100k_base"
DEFAULT_BUDGET_TOKENS = 2048
DEFAULT_MAX_RESULTS = 20


class _FromDict:
    """Builds a dataclass from a JSON object, ignoring unknown keys so older files stay readable."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        names = {f.name for f in fields(cast(Any, cls))}
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class Link(_FromDict):
    type: str  # temporal | semantic | version | sibling
    target_note_id: str
    weight: float = 1.0


@dataclass
class Note(_FromDict):
    note_id: str
    history_id: str
    session_id: str
    turn_idx: int  # -1 marks a session-parent note
    role: str
    text: str
    ts: str  # ISO-8601 valid-from
    valid_to: str | None = None
    version: int = 1
    supersedes_note_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    entity_tags: list[str] = field(default_factory=list)
    summary_id: str | None = None
    links: list[Link] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Note:
        note = super().from_dict(data)
        note.links = [link if isinstance(link, Link) else Link.from_dict(link) for link in note.links]
        return note

    @property
    def is_session_parent(self) -> bool:
        return self.turn_idx < 0


@dataclass
class Summary(_FromDict):
    summary_id: str
    history_id: str
    session_id: str
    text: str
    source_note_ids: list[str]
    created_ts: str
    stale: bool = False


@dataclass
class Question(_FromDict):
    question_id: str
    history_id: str
    question_type: str
    question: str
    question_date: str
    answer: str
    gold_session_ids: list[str]
    gold_note_ids: list[str]
    split: str = "test"


@dataclass
class Update(_FromDict):
    update_id: str
    ts: str
    op: str  # add | correct | supersede
    note: dict[str, Any]
    supersedes_note_id: str | None = None
    explicit: bool = True


@dataclass
class RetrievalRequest(_FromDict):
    query: str
    budget_ms: int = 5000
    budget_tokens: int = DEFAULT_BUDGET_TOKENS
    max_results: int = DEFAULT_MAX_RESULTS
    as_of_time: str | None = None
    plan: str | None = None
    question_id: str | None = None


@dataclass
class EvidenceBundle(_FromDict):
    note_ids: list[str]
    passages: list[str]
    scores: list[float]
    total_tokens: int
    latency_ms: dict[str, Any]  # {"total": float, "stages": {name: float}}
    stale_flags: list[bool]
    trace: dict[str, Any]
    question_id: str | None = None
    plan: str | None = None


class Retriever(Protocol):
    def build(self, notes: Iterable[Note], summaries: Iterable[Summary] | None) -> None: ...

    def insert(self, note: Note) -> dict: ...

    def retrieve(self, req: RetrievalRequest) -> EvidenceBundle: ...


# --- serialisation ---------------------------------------------------------------------------------------------


def to_dict(record: Any) -> dict[str, Any]:
    return asdict(record)


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    """Writes one JSON object per line; returns the number of records written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            row = record if isinstance(record, dict) else to_dict(record)
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as src:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg}); fix or regenerate the file") from exc


def load_notes(path: str | Path) -> list[Note]:
    return [Note.from_dict(row) for row in read_jsonl(path)]


def load_questions(path: str | Path) -> list[Question]:
    return [Question.from_dict(row) for row in read_jsonl(path)]


def load_summaries(path: str | Path) -> list[Summary]:
    return [Summary.from_dict(row) for row in read_jsonl(path)]


# --- token budget ----------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_encoding():
    import tiktoken

    return tiktoken.get_encoding(TOKENIZER_NAME)


def count_tokens(text: str) -> int:
    return len(get_encoding().encode(text, disallowed_special=()))


def truncate_text(text: str, max_tokens: int) -> str:
    """Cuts text to at most max_tokens tokens (cl100k_base)."""
    tokens = get_encoding().encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return get_encoding().decode(tokens[:max_tokens])


def truncate_to_budget(passages: list[str], budget_tokens: int) -> list[str]:
    """Keeps passages in rank order until the next one would exceed the budget.

    A top passage that alone exceeds the budget is truncated rather than dropped, so a non-empty
    ranking never yields empty evidence.
    """
    kept: list[str] = []
    used = 0
    for passage in passages:
        cost = count_tokens(passage)
        if used + cost <= budget_tokens:
            kept.append(passage)
            used += cost
            continue
        if not kept:
            kept.append(truncate_text(passage, budget_tokens))
        break
    return kept
