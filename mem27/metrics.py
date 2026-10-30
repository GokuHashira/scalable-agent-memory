"""Evidence-level metrics: gold-session and gold-turn recall / hit @k.

note_id format is "{qid}:{sid}:{turn_idx}" so session id = note_id.split(':')[1].
Bundles store ranked note_ids (top-MAX_RESULTS); recall@k uses the first k.
"""
from __future__ import annotations

from statistics import mean


def session_of(note_id: str) -> str:
    parts = note_id.split(":")
    return parts[1] if len(parts) >= 3 else note_id


def topk_sessions(note_ids: list[str], k: int) -> list[str]:
    seen = []
    for nid in note_ids[:k]:
        s = session_of(nid)
        if s not in seen:
            seen.append(s)
    return seen


def session_recall_at_k(note_ids, gold_sessions, k) -> float:
    gold = set(gold_sessions)
    if not gold:
        return float("nan")
    got = set(topk_sessions(note_ids, k)) & gold
    return len(got) / len(gold)


def session_hit_at_k(note_ids, gold_sessions, k) -> int:
    gold = set(gold_sessions)
    if not gold:
        return 0
    return int(bool(set(s for s in (session_of(n) for n in note_ids[:k])) & gold))


def note_recall_at_k(note_ids, gold_notes, k) -> float:
    gold = set(gold_notes)
    if not gold:
        return float("nan")
    got = set(note_ids[:k]) & gold
    return len(got) / len(gold)


def note_hit_at_k(note_ids, gold_notes, k) -> int:
    gold = set(gold_notes)
    if not gold:
        return 0
    return int(bool(set(note_ids[:k]) & gold))


def success(note_ids, question, success_on: str, k: int) -> int:
    """Binary success used for oracle labels. hit-style (>=1 gold in top-k)."""
    if success_on == "session":
        return session_hit_at_k(note_ids, question.gold_session_ids, k)
    elif success_on == "note":
        return note_hit_at_k(note_ids, question.gold_note_ids, k)
    raise ValueError(success_on)


def nanmean(xs) -> float:
    vals = [x for x in xs if x == x]  # drop NaN
    return mean(vals) if vals else float("nan")


def percentile(xs, p) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return float(xs[0])
    idx = p / 100 * (len(xs) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    frac = idx - lo
    return float(xs[lo] * (1 - frac) + xs[hi] * frac)
