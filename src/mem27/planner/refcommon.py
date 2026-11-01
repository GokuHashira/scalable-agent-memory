"""Shared helpers: tokenization, token counting, RRF, grouping."""
from __future__ import annotations

import functools
import re
from collections import defaultdict

from mem27.planner.schema import TOKEN_ENCODING, RRF_K

_WORD = re.compile(r"[A-Za-z0-9']+")

# bge-base-en-v1.5 retrieval query instruction (passages get no instruction).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def bm25_tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@functools.lru_cache(maxsize=1)
def _encoder():
    import tiktoken
    return tiktoken.get_encoding(TOKEN_ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text))


def passage_text(role: str, text: str) -> str:
    """Canonical passage string used for bundles + token counting."""
    role = (role or "").strip()
    return f"{role}: {text}" if role else text


def rrf_fuse(rank_lists: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal-rank fusion over several ranked lists of item indices."""
    scores: dict[int, float] = defaultdict(float)
    for ranks in rank_lists:
        for r, idx in enumerate(ranks):
            scores[idx] += 1.0 / (k + r + 1)
    return scores


def group_sessions(notes) -> tuple[list[str], dict[str, list[int]], dict[str, str]]:
    """Return (session_order, sid->note_indices, sid->date)."""
    order: list[str] = []
    members: dict[str, list[int]] = defaultdict(list)
    date: dict[str, str] = {}
    for i, n in enumerate(notes):
        sid = n.session_id
        if sid not in members:
            order.append(sid)
        members[sid].append(i)
        date[sid] = n.ts
    return order, dict(members), date
