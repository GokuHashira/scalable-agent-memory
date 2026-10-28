"""Slice V retriever-free bundles (docs/plans/slice-V-data-eval.md §2.2): gold, random, fullhist2k, fullhist16k.

  gold        gold turns only (`gold_note_ids`; questions without labelled turns fall back to their gold-session
              turns), chronological order, no cap. Same notes as runs/gold_turn_lme_s (scripts/make_ref_bundles.py
              kind gold_turn); only the order and the plan name differ, and the reader renders bundles
              chronologically anyway, so reader prompts are identical.
  random      random turns from the same history packed under 2,048 tokens / 20 notes (seed 0). Identical to
              runs/random_lme_s (make_ref_bundles kind random); an existing file is verified, never overwritten.
  fullhist2k  the whole history, most recent turns first (session date desc, then turn index desc), packed
              until 2,048 cl100k tokens (the turn crossing the cap is truncated); no max_results.
  fullhist16k the same with a 16,384-token cap.

  python scripts/gold_bundles.py --data data/lme_s --tag lme_s
  -> runs/gold_lme_s, runs/random_lme_s (verified), runs/fullhist_lme_s_2k, runs/fullhist_lme_s_16k
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
from mem27.vschema import Question, iter_jsonl, read_jsonl, write_jsonl  # noqa: E402
from scripts.make_ref_bundles import build  # noqa: E402

KEEP = ("question_id", "plan", "note_ids", "passages", "total_tokens")


def chrono_key(idx: NoteIndex, nid: str):
    m = idx.meta[nid]  # (history_id, session_id, turn_idx, role, ts, date_raw, valid_to)
    return (m[4] or "", m[1], m[2])


def finish(b: dict, q: Question, plan: str, trace: dict) -> dict:
    b.update({"question_id": q.question_id, "plan": plan, "latency_ms": {"total": 0.0, "stages": {}},
              "stale_flags": [False] * len(b["note_ids"]), "trace": trace})
    return b


def gold(q: Question, idx: NoteIndex) -> dict:
    b = build("gold_turn", q, idx, random.Random(0))
    order = sorted(range(len(b["note_ids"])), key=lambda i: chrono_key(idx, b["note_ids"][i]))
    for k in ("note_ids", "passages", "scores"):
        b[k] = [b[k][i] for i in order]
    return finish(b, q, "gold", {"kind": "gold", **(b.get("trace") or {})})


def fullhist(q: Question, idx: NoteIndex, cap: int, plan: str) -> dict:
    turns = [n for n in idx.by_history[q.history_id] if idx.meta[n][2] >= 0]
    recent_first = sorted(turns, key=lambda n: chrono_key(idx, n), reverse=True)
    b = pack(recent_first, idx.text, token_cap=cap, max_results=None)
    return finish(b, q, plan, {"kind": plan, "cap": cap, "order": "most recent first", "history_turns": len(turns)})


def write_run(run: Path, rows: list[dict], env: dict, cfg: dict) -> None:
    write_jsonl(run / "bundles.jsonl", rows)
    write_json(run / "env.json", env)
    write_json(run / "run_config.json", cfg)
    toks = [r["total_tokens"] for r in rows]
    print(f"{run}: n={len(rows)} mean_tokens={sum(toks) / len(toks):.0f} max={max(toks)}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    data, runs = Path(a.data), Path(a.runs)
    qs = read_jsonl(data / "questions.jsonl", Question)
    idx = NoteIndex(data / "notes.jsonl", keep_text=True)
    env = collect_env(seed=a.seed)

    write_run(runs / f"gold_{a.tag}", [gold(q, idx) for q in qs], env,
              {"kind": "gold", "data": str(data), "source": "gold_note_ids (fallback: gold-session turns)"})

    rng = random.Random(a.seed)  # same sequence as scripts/make_ref_bundles.py kind random
    rand = [build("random", q, idx, rng) for q in qs]
    for r in rand:
        r["plan"] = "random"
    rdir = runs / f"random_{a.tag}"
    if (rdir / "bundles.jsonl").exists():
        old = {r["question_id"]: r for r in iter_jsonl(rdir / "bundles.jsonl")}
        same = all({k: old[r["question_id"]].get(k) for k in KEEP} == {k: r.get(k) for k in KEEP} for r in rand)
        print(f"{rdir}: existing bundles {'identical (kept)' if same else 'DIFFER (kept existing; new not written)'}")
    else:
        write_run(rdir, rand, env, {"kind": "random", "data": str(data), "seed": a.seed})

    for cap, plan, name in ((2048, "fullhist2k", f"fullhist_{a.tag}_2k"), (16384, "fullhist16k", f"fullhist_{a.tag}_16k")):
        write_run(runs / name, [fullhist(q, idx, cap, plan) for q in qs], env,
                  {"kind": plan, "data": str(data), "cap": cap, "order": "most recent first"})


if __name__ == "__main__":
    main()
