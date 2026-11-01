"""Convert LongMemEval_S -> Slice-K JSONL (planner.md section 3.1).

  one history_id per question
  one note per turn: note_id = f"{qid}:{sid}:{i}", ts = haystack_dates[k]
  gold_session_ids = answer_session_ids
  gold_note_ids    = turns with has_answer
  stratified 100 dev / 400 test, seed 0

Usage:
  python -m mem27.planner.ref_data \
      --raw data/raw/longmemeval_s.json --out data/lme_s_k
"""
from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict

from mem27.planner.schema import Note, Question, write_jsonl, QUESTION_TYPES


def _is_gold_turn(turn: dict) -> bool:
    v = turn.get("has_answer", False)
    return v is True or (isinstance(v, str) and v.strip().lower() == "true")


def convert(raw: list[dict]) -> tuple[list[Note], list[Question]]:
    notes: list[Note] = []
    questions: list[Question] = []
    for q in raw:
        qid = q["question_id"]
        hid = qid  # one history per question
        gold_sessions = list(q.get("answer_session_ids") or [])
        sids = q["haystack_session_ids"]
        dates = q["haystack_dates"]
        sessions = q["haystack_sessions"]
        gold_notes: list[str] = []
        for k, (sid, date, sess) in enumerate(zip(sids, dates, sessions)):
            for i, turn in enumerate(sess):
                note_id = f"{qid}:{sid}:{i}"
                is_gold = _is_gold_turn(turn)
                if is_gold:
                    gold_notes.append(note_id)
                notes.append(Note(
                    note_id=note_id,
                    history_id=hid,
                    session_id=sid,
                    turn_idx=i,
                    role=turn.get("role", ""),
                    text=turn.get("content", ""),
                    ts=date,
                    metadata={"session_idx": k, "is_gold_turn": is_gold},
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
            split=None,  # assigned below
        ))
    return notes, questions


def stratified_split(questions: list[Question], n_dev: int = 100, seed: int = 0):
    """Proportional stratified dev/test by question_type; deterministic."""
    by_type: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q)
    total = len(questions)
    rng = random.Random(seed)
    # proportional target per type (floor), then distribute remainder by largest fractional part
    targets = {}
    fracs = {}
    for t, qs in by_type.items():
        exact = n_dev * len(qs) / total
        targets[t] = int(exact)
        fracs[t] = exact - int(exact)
    assigned = sum(targets.values())
    for t in sorted(fracs, key=lambda x: (-fracs[x], x)):
        if assigned >= n_dev:
            break
        targets[t] += 1
        assigned += 1
    for t, qs in by_type.items():
        order = qs[:]
        rng.shuffle(order)
        dev_ids = set(id(q) for q in order[:targets[t]])
        for q in qs:
            q.split = "dev" if id(q) in dev_ids else "test"
    return questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/longmemeval_s.json")
    ap.add_argument("--out", default="data/lme_s_k")
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import json
    raw = json.load(open(args.raw))
    notes, questions = convert(raw)
    stratified_split(questions, n_dev=args.n_dev, seed=args.seed)

    os.makedirs(args.out, exist_ok=True)
    nn = write_jsonl(os.path.join(args.out, "notes.jsonl"), notes)
    nq = write_jsonl(os.path.join(args.out, "questions.jsonl"), questions)

    # ---- validation / stats ----
    split_c = Counter(q.split for q in questions)
    type_c = Counter(q.question_type for q in questions)
    note_ids = set(n.note_id for n in notes)
    sess_by_hist = defaultdict(set)
    for n in notes:
        sess_by_hist[n.history_id].add(n.session_id)
    unresolved_sess = unresolved_note = n_gold_sess = n_gold_note = 0
    abs_no_gold = 0
    for q in questions:
        if not q.gold_session_ids and q.question_id.endswith("_abs"):
            abs_no_gold += 1
        n_gold_sess += len(q.gold_session_ids)
        n_gold_note += len(q.gold_note_ids)
        for gs in q.gold_session_ids:
            if gs not in sess_by_hist[q.history_id]:
                unresolved_sess += 1
        for gn in q.gold_note_ids:
            if gn not in note_ids:
                unresolved_note += 1

    print(f"notes={nn} questions={nq}")
    print(f"types={dict(type_c)}")
    print(f"split={dict(split_c)}")
    print(f"gold_session_ids total={n_gold_sess} unresolved={unresolved_sess}")
    print(f"gold_note_ids total={n_gold_note} unresolved={unresolved_note}")
    print(f"abstention(_abs) with no gold session={abs_no_gold}")
    assert nq == 500, f"expected 500 questions, got {nq}"
    assert set(type_c) == set(QUESTION_TYPES), f"type mismatch {set(type_c)}"
    assert unresolved_sess == 0, "some gold session ids do not resolve"
    assert unresolved_note == 0, "some gold note ids do not resolve"
    assert split_c["dev"] == args.n_dev, f"dev={split_c['dev']}"
    print("OK: all gold ids resolve; six types; 500 questions.")


if __name__ == "__main__":
    main()
