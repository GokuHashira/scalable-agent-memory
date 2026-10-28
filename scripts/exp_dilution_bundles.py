"""E1: context dilution. The same gold evidence, plus D tokens of distractor sessions from the same history.

Evidence = every turn of every gold session (what the reader needs; same as runs/oracle_lme_s).
Distractors = whole sessions from the same haystack, by source tier (data/_audit session map):
  hard = simulated user sessions written for other questions (same style as the evidence)
  easy = real ShareGPT / UltraChat chats (style-separable filler; leakage audit AUC 0.96)
Position (the reader renders sessions chronologically, so position is set through dates):
  last  = distractors dated before the evidence  -> the evidence closes the context
  first = distractors dated after the evidence   -> the evidence opens the context
  Candidates are taken nearest-in-time first; if one side runs out, the other side fills in and the
  achieved share of distractor tokens before the evidence is recorded in trace.frac_before.
Questions: the 100 dev questions plus all 30 abstention questions (the 24 test ones form a hallucination
diagnostic only; nothing is selected on them).

  python scripts/exp_dilution_bundles.py --data data/lme_s
  -> runs/xdil_{base,hard_4k_last,hard_4k_first,hard_16k_last,hard_16k_first,easy_4k_last,easy_16k_last}_lme_s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import NoteIndex, collect_env, n_tokens, truncate_tokens, write_json  # noqa: E402
from mem27.vschema import Question, iter_jsonl, read_jsonl, session_of, write_jsonl  # noqa: E402

CONDITIONS = [  # (name, tier, D tokens, position)
    ("base", None, 0, None),
    ("hard_4k_last", "hard", 4096, "last"), ("hard_4k_first", "hard", 4096, "first"),
    ("hard_16k_last", "hard", 16384, "last"), ("hard_16k_first", "hard", 16384, "first"),
    ("easy_4k_last", "easy", 4096, "last"), ("easy_16k_last", "easy", 16384, "last"),
]


def tier_of(orig: str) -> str:
    if orig.startswith("answer"):
        return "gold_named"
    if orig.startswith(("sharegpt", "ultrachat")):
        return "easy"
    return "hard"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lme_s")
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args(argv)
    data = Path(a.data)
    qs = [q for q in read_jsonl(data / "questions.jsonl", Question)
          if q.split == "dev" or q.question_id.endswith("_abs")]
    hist = {q.history_id for q in qs}
    idx = NoteIndex(data / "notes.jsonl", keep_text=True, histories=hist)
    tier = {(r["history_id"], r["session_id"]): tier_of(r["orig_session_id"])
            for r in iter_jsonl(data.parent / "_audit" / data.name / "session_id_map.jsonl") if r["history_id"] in hist}
    env = collect_env(seed=0)
    out = {c[0]: [] for c in CONDITIONS}
    for q in qs:
        h = q.history_id
        turns = [n for n in idx.by_history[h] if idx.meta[n][2] >= 0]
        by_sess: dict[str, list[str]] = {}
        for n in turns:
            by_sess.setdefault(idx.meta[n][1], []).append(n)
        for s in by_sess:
            by_sess[s].sort(key=lambda n: idx.meta[n][2])
        sdate = {s: idx.meta[ts[0]][4] for s, ts in by_sess.items()}
        gold = [s for s in q.gold_session_ids if s in by_sess]
        ev = [n for s in sorted(gold, key=lambda s: sdate[s]) for n in by_sess[s]]
        ev_tok = sum(n_tokens(idx.text[n]) for n in ev)
        g_lo = min(sdate[s] for s in gold) if gold else ""
        g_hi = max(sdate[s] for s in gold) if gold else ""
        for name, tr, D, pos in CONDITIONS:
            ids, passages = list(ev), [idx.text[n] for n in ev]
            dtok, before = 0, 0
            if D:
                cands = [s for s in by_sess if s not in gold and tier.get((h, s)) == tr]
                pre = sorted([s for s in cands if sdate[s] < g_lo], key=lambda s: sdate[s], reverse=True)
                post = sorted([s for s in cands if sdate[s] > g_hi], key=lambda s: sdate[s])
                order = (pre + post) if pos == "last" else (post + pre)
                for s in order:
                    if dtok >= D:
                        break
                    for n in by_sess[s]:
                        t = idx.text[n]
                        k = n_tokens(t)
                        if dtok + k > D:
                            room = D - dtok
                            if room < 32:
                                dtok = D
                                break
                            t = truncate_tokens(t, room)
                            k = n_tokens(t)
                        ids.append(n)
                        passages.append(t)
                        dtok += k
                        if sdate[s] < g_lo:
                            before += k
                        if dtok >= D:
                            break
            b = {"question_id": q.question_id, "plan": f"xdil_{name}", "note_ids": ids, "passages": passages,
                 "scores": [0.0] * len(ids), "total_tokens": ev_tok + dtok,
                 "latency_ms": {"total": 0.0, "stages": {}}, "stale_flags": [False] * len(ids),
                 "trace": {"kind": "dilution", "tier": tr, "D": D, "position": pos, "evidence_tokens": ev_tok,
                           "distractor_tokens": dtok, "frac_before": (before / dtok) if dtok else None}}
            out[name].append(b)
    for name, tr, D, pos in CONDITIONS:
        run = Path(a.runs) / f"xdil_{name}_{data.name}"
        rows = out[name]
        write_jsonl(run / "bundles.jsonl", rows)
        write_json(run / "env.json", env)
        write_json(run / "run_config.json", {"experiment": "E1 dilution", "tier": tr, "D": D, "position": pos,
                                             "questions": "dev (100) + all abstention (30)"})
        fb = [r["trace"]["frac_before"] for r in rows if r["trace"]["frac_before"] is not None]
        print(f"{run}: n={len(rows)} mean_tokens={sum(r['total_tokens'] for r in rows) / len(rows):.0f} "
              f"max={max(r['total_tokens'] for r in rows)} mean_frac_before={sum(fb) / len(fb) if fb else float('nan'):.2f}")


if __name__ == "__main__":
    main()
