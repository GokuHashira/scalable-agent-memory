# INDEPENDENT V-STYLE BUILD -- NOT V's real data/lme_s (which does not exist in
# this workspace). This is a genuinely-independent second conversion of the SAME
# public LongMemEval_S benchmark (planner.md section 0: "build your own copy of the
# data"), used as a DATA-AXIS robustness check: does the routing-headroom result (H2)
# survive a different note-construction granularity? It differs from the reference
# build (src/mem27/planner/ref_data.py, one note per turn) by merging consecutive turns
# into fixed 2-turn WINDOW notes -- a coarser, independently-coded retrieval unit.
#
# INTEGRATION (post-midterm): replace the output with V's real data/lme_s. The
# schema is identical (planner.md section 0), so oracle/run_plan/embed are unchanged;
# only the file contents differ.
#
#   python -m mem27.planner.v_data --raw data/raw/longmemeval_s.json --out data/lme_s
"""Independent V-style data build: 2-turn window notes over LongMemEval_S."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from mem27.planner.schema import Note, Question, write_jsonl
from mem27.planner.ref_data import _is_gold_turn, stratified_split

WINDOW = 2  # turns per window note (the independent-construction knob)


def convert(raw: list[dict]) -> tuple[list[Note], list[Question]]:
    """One history per question; notes are WINDOW-turn windows within a session.
    A window note is gold iff it contains at least one has_answer turn."""
    notes: list[Note] = []
    questions: list[Question] = []
    for q in raw:
        qid = q["question_id"]
        hid = qid
        gold_sessions = list(q.get("answer_session_ids") or [])
        sids = q["haystack_session_ids"]
        dates = q["haystack_dates"]
        sessions = q["haystack_sessions"]
        gold_notes: list[str] = []
        for k, (sid, date, sess) in enumerate(zip(sids, dates, sessions)):
            # group consecutive turns into windows of size WINDOW
            for j, start in enumerate(range(0, len(sess), WINDOW)):
                window = sess[start:start + WINDOW]
                note_id = f"{qid}:{sid}:w{j}"
                is_gold = any(_is_gold_turn(t) for t in window)
                if is_gold:
                    gold_notes.append(note_id)
                # merge the window's turns into one passage, role-prefixed
                merged = " \n".join(
                    f"{t.get('role','')}: {t.get('content','')}".strip() for t in window
                )
                roles = "+".join(sorted({t.get("role", "") for t in window}))
                notes.append(Note(
                    note_id=note_id,
                    history_id=hid,
                    session_id=sid,
                    turn_idx=j,
                    role=roles or "window",
                    text=merged,
                    ts=date,
                    metadata={"session_idx": k, "is_gold_window": is_gold,
                              "window_turns": len(window)},
                ))
        questions.append(Question(
            question_id=qid,
            history_id=hid,
            question_type=q["question_type"],
            question=q["question"],
            question_date=q.get("question_date"),
            answer=q.get("answer"),
            gold_session_ids=gold_sessions,
            gold_note_ids=gold_notes,
            split=None,
        ))
    return notes, questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/longmemeval_s.json")
    ap.add_argument("--out", default="data/lme_s")
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.raw) as f:
        raw = json.load(f)
    notes, questions = convert(raw)
    stratified_split(questions, n_dev=args.n_dev, seed=args.seed)

    os.makedirs(args.out, exist_ok=True)
    # If a prior symlink stand-in is present, replace it with real files.
    for fn in ("notes.jsonl", "questions.jsonl"):
        p = os.path.join(args.out, fn)
        if os.path.islink(p):
            os.unlink(p)
    write_jsonl(os.path.join(args.out, "notes.jsonl"), notes)
    write_jsonl(os.path.join(args.out, "questions.jsonl"), questions)

    # validation
    note_ids = set(n.note_id for n in notes)
    unresolved = 0
    n_gold = 0
    for q in questions:
        n_gold += len(q.gold_note_ids)
        for gn in q.gold_note_ids:
            if gn not in note_ids:
                unresolved += 1
    split_c = Counter(q.split for q in questions)
    types = Counter(q.question_type for q in questions)
    print(f"V-style build -> {args.out}")
    print(f"  notes={len(notes)} (2-turn windows)  questions={len(questions)}")
    print(f"  split={dict(split_c)}  types={len(types)}")
    print(f"  gold_note_ids={n_gold}  unresolved={unresolved}")
    assert unresolved == 0, "every gold window id must resolve"
    assert len(questions) == 500


if __name__ == "__main__":
    main()
