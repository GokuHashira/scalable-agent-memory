"""Budgeted packing into an EvidenceBundle (token cap 2,048 cl100k, max_results 20)."""
from __future__ import annotations

from mem27.eval.common import MAX_RESULTS, TOKEN_CAP, n_tokens, truncate_tokens

MIN_TAIL = 32  # don't add a truncated passage shorter than this


def pack(note_ids: list[str], texts: dict[str, str], scores: list[float] | None = None,
         token_cap: int | None = TOKEN_CAP, max_results: int | None = MAX_RESULTS) -> dict:
    """Greedy in rank order; the passage that crosses the cap is truncated, then stop."""
    out_ids, passages, out_scores, total = [], [], [], 0
    scores = scores if scores is not None else [0.0] * len(note_ids)
    for nid, s in zip(note_ids, scores):
        if max_results is not None and len(out_ids) >= max_results:
            break
        t = texts[nid]
        k = n_tokens(t)
        if token_cap is not None and total + k > token_cap:
            room = token_cap - total
            if room >= MIN_TAIL:
                t = truncate_tokens(t, room)
                k = n_tokens(t)
                out_ids.append(nid)
                passages.append(t)
                out_scores.append(float(s))
                total += k
            break
        out_ids.append(nid)
        passages.append(t)
        out_scores.append(float(s))
        total += k
    return {"note_ids": out_ids, "passages": passages, "scores": out_scores, "total_tokens": total}
