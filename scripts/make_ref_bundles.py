"""Reference bundles for the evaluator self-test (V, Oct 13).

  gold_turn   gold turns only (abstention / unlabeled: gold-session turns), no cap
  gold_sess   gold sessions packed under the shared budget (2,048 tok / 20 notes),
              gold turns first, then neighbours outward  -> budgeted reader ceiling
  oracle      every turn of every gold session, no cap (LongMemEval_Oracle-style ceiling)
  random      random turns from the same history under the shared budget (seed 0)
  none        empty bundle (no-retrieval: parametric knowledge + abstention behaviour)

  python scripts/make_ref_bundles.py --data data/lme_s --tag lme_s --kinds gold_turn gold_sess oracle random none
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import NoteIndex, collect_env, write_json  # noqa: E402
from mem27.eval.packing import pack  # noqa: E402
from mem27.vschema import Question, read_jsonl, session_of, write_jsonl  # noqa: E402


def session_turns(idx: NoteIndex, hid: str, sid: str) -> list[str]:
    ids = [n for n in idx.by_history[hid] if idx.session(n) == sid and idx.turn(n) >= 0]
    return sorted(ids, key=idx.turn)


def gold_centered(idx: NoteIndex, q: Question) -> list[str]:
    """Gold turns first, then turns at distance 1, 2, ... inside gold sessions (chronological sessions)."""
    sess = sorted(q.gold_session_ids, key=lambda s: idx.meta[f"{q.history_id}:{s}:-1"][4])
    turns = {s: session_turns(idx, q.history_id, s) for s in sess}
    gold = [g for g in q.gold_note_ids if g in idx.meta]
    if not gold:
        gold = [t[0] for t in turns.values() if t]
    order, seen = list(gold), set(gold)
    for dist in range(1, 200):
        added = False
        for g in gold:
            s, t = session_of(g), idx.turn(g)
            for nb in (t - dist, t + dist):
                nid = f"{q.history_id}:{s}:{nb}"
                if nid in idx.meta and nid not in seen:
                    order.append(nid)
                    seen.add(nid)
                    added = True
        if not added:
            break
    for s in sess:  # any remaining turns of gold sessions
        for nid in turns[s]:
            if nid not in seen:
                order.append(nid)
                seen.add(nid)
    return order


def build(kind: str, q: Question, idx: NoteIndex, rng: random.Random) -> dict:
    hid = q.history_id
    trace = {"kind": kind}
    if kind == "none":
        b = {"note_ids": [], "passages": [], "scores": [], "total_tokens": 0}
    elif kind == "gold_turn":
        ids = [g for g in q.gold_note_ids if g in idx.meta]
        if not ids:
            trace["fallback"] = "gold_session_turns"
            ids = [n for s in q.gold_session_ids for n in session_turns(idx, hid, s)]
        b = pack(ids, idx.text, token_cap=None, max_results=None)
    elif kind == "gold_sess":
        b = pack(gold_centered(idx, q), idx.text)
    elif kind == "oracle":
        ids = [n for s in sorted(q.gold_session_ids, key=lambda s: idx.meta[f"{hid}:{s}:-1"][4])
               for n in session_turns(idx, hid, s)]
        b = pack(ids, idx.text, token_cap=None, max_results=None)
    elif kind == "random":
        ids = [n for n in idx.by_history[hid] if idx.turn(n) >= 0]
        rng.shuffle(ids)
        b = pack(ids, idx.text)
    else:
        raise ValueError(kind)
    b.update({"question_id": q.question_id, "plan": kind, "latency_ms": {"total": 0.0, "stages": {}},
              "stale_flags": [False] * len(b["note_ids"]), "trace": trace})
    return b


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--kinds", nargs="+", default=["gold_turn", "gold_sess", "oracle", "random", "none"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    data = Path(a.data)
    qs = read_jsonl(data / "questions.jsonl", Question)
    nf = data / "notes_final.jsonl" if (data / "notes_final.jsonl").exists() else data / "notes.jsonl"
    idx = NoteIndex(nf, keep_text=True)
    env = collect_env(seed=a.seed)
    for kind in a.kinds:
        rng = random.Random(a.seed)
        rows = [build(kind, q, idx, rng) for q in qs]
        run = Path(a.runs) / f"{kind}_{a.tag}"
        write_jsonl(run / "bundles.jsonl", rows)
        write_json(run / "env.json", env)
        write_json(run / "run_config.json", {"kind": kind, "data": str(data), "seed": a.seed})
        toks = [r["total_tokens"] for r in rows]
        print(f"{run}: n={len(rows)} mean_tokens={sum(toks) / len(toks):.0f} max={max(toks)}")


if __name__ == "__main__":
    main()
