"""Shared dataclasses + JSONL (de)serialisers (contract §0).

Owner: K. This is V's stand-in written to the §0 contract ("If late: local
dataclasses per §0"); replace with K's file when it lands. Field names and
types follow the contract exactly so the swap is a no-op.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol, TypeVar

T = TypeVar("T")


@dataclass
class Link:
    type: str  # temporal | semantic | version | sibling
    target_note_id: str
    weight: float = 1.0


@dataclass
class Note:
    note_id: str
    history_id: str
    session_id: str
    turn_idx: int  # -1 = session-parent note
    role: str  # user | assistant | session
    text: str
    ts: str  # ISO-8601 valid-from
    valid_to: str | None = None
    version: int = 1
    supersedes_note_id: str | None = None
    metadata: dict = field(default_factory=dict)
    entity_tags: list = field(default_factory=list)
    summary_id: str | None = None
    links: list = field(default_factory=list)  # list[Link] (dicts on disk)


@dataclass
class Summary:
    summary_id: str
    history_id: str
    session_id: str
    text: str
    source_note_ids: list
    created_ts: str
    stale: bool = False


@dataclass
class Question:
    question_id: str
    history_id: str
    question_type: str
    question: str
    question_date: str
    answer: str
    gold_session_ids: list
    gold_note_ids: list
    split: str  # dev | test


@dataclass
class Update:
    update_id: str
    ts: str
    op: str  # add | correct | supersede
    note: Note | dict
    supersedes_note_id: str | None
    explicit: bool


@dataclass
class RetrievalRequest:
    query: str
    budget_ms: float | None = None
    budget_tokens: int = 2048
    max_results: int = 20
    as_of_time: str | None = None
    plan: str | None = None
    question_id: str | None = None


@dataclass
class EvidenceBundle:
    note_ids: list
    passages: list
    scores: list
    total_tokens: int
    latency_ms: dict  # {"total": ms, "stages": {name: ms}}
    stale_flags: list
    trace: dict = field(default_factory=dict)


class Retriever(Protocol):
    def build(self, notes: Iterable[Note], summaries: Iterable[Summary] | None) -> None: ...
    def insert(self, note: Note) -> dict: ...
    def retrieve(self, req: RetrievalRequest) -> EvidenceBundle: ...


# ---------------------------------------------------------------- JSONL I/O

def to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    return obj


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(to_dict(r), ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: str | Path, cls: type[T] | None = None) -> list:
    rows = list(iter_jsonl(path))
    if cls is None:
        return rows
    names = {f.name for f in dataclasses.fields(cls)}
    return [cls(**{k: v for k, v in r.items() if k in names}) for r in rows]


def session_of(note_id: str) -> str:
    """note_id = f"{history_id}:{session_id}:{turn_idx}" -> session_id."""
    return note_id.rsplit(":", 2)[-2]


def turn_of(note_id: str) -> int:
    return int(note_id.rsplit(":", 1)[-1])
