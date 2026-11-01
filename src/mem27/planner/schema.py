"""Shared dataclasses + JSONL I/O for Slice K.

All three slices (G, V, K) read/write the same JSONL files so merging is cheap.
See planner.md section 0 for the field contracts.

Fixed defaults (planner.md section 0):
  TOKEN_CAP = 2048 (cl100k_base)
  MAX_RESULTS = 20
  EMBED_MODEL = BAAI/bge-base-en-v1.5
  RRF_K = 60
  TRAVERSAL H=4, L=50
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Iterator

# ---- Fixed defaults -------------------------------------------------------
TOKEN_CAP = 2048
TOKEN_ENCODING = "cl100k_base"
MAX_RESULTS = 20
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
RRF_K = 60
TRAVERSAL_H = 4
TRAVERSAL_L = 50

QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "knowledge-update",
    "multi-session",
]


# ---- Dataclasses ----------------------------------------------------------
@dataclass
class Note:
    note_id: str
    history_id: str
    session_id: str
    turn_idx: int
    role: str
    text: str
    ts: str | None = None
    valid_to: str | None = None
    version: int = 1
    supersedes_note_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    entity_tags: list[str] = field(default_factory=list)
    summary_id: str | None = None
    links: list[str] = field(default_factory=list)


@dataclass
class Question:
    question_id: str
    history_id: str
    question_type: str
    question: str
    question_date: str | None = None
    answer: str | None = None
    gold_session_ids: list[str] = field(default_factory=list)
    gold_note_ids: list[str] = field(default_factory=list)
    split: str | None = None


@dataclass
class Bundle:
    """One plan's retrieval output for one question."""
    question_id: str
    plan: str
    note_ids: list[str] = field(default_factory=list)
    passages: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    total_tokens: int = 0
    latency_ms: dict[str, Any] = field(default_factory=lambda: {"total": 0.0, "stages": {}})
    stale_flags: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    question_id: str
    plan: str
    answer: str | None = None
    judge: dict[str, Any] = field(default_factory=dict)  # {correct, judge_model, raw}


# ---- JSONL I/O ------------------------------------------------------------
def _json_default(o):
    # numpy scalars/arrays -> native python (traces may carry np.int64 etc.)
    if hasattr(o, "item") and not isinstance(o, (list, dict)):
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def write_jsonl(path: str, rows: Iterable[Any]) -> int:
    n = 0
    with open(path, "w") as f:
        for r in rows:
            if dataclasses.is_dataclass(r) and not isinstance(r, type):
                r = asdict(r)
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _from_dict(cls, d: dict):
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def load_notes(path: str) -> list[Note]:
    return [_from_dict(Note, d) for d in read_jsonl(path)]


def load_questions(path: str) -> list[Question]:
    return [_from_dict(Question, d) for d in read_jsonl(path)]


def load_bundles(path: str) -> list[Bundle]:
    return [_from_dict(Bundle, d) for d in read_jsonl(path)]


def load_answers(path: str) -> list[Answer]:
    return [_from_dict(Answer, d) for d in read_jsonl(path)]
