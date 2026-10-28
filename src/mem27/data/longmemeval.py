"""LongMemEval (S / Oracle / M) -> notes.jsonl + questions.jsonl  (DATA-01).

One history per question (history_id = question_id); one Note per turn
(note_id = f"{qid}:{sid}:{turn_idx}", ts = session date); one session-parent
Note per session (turn_idx = -1, text = the concatenated turns).

Leakage guards
- LongMemEval names evidence sessions "answer_<hash>_<n>". Session ids are
  replaced by a salted hash so no retriever/reader can key on the name. The
  map back to original ids goes to data/_audit/<dataset>/, never next to notes.
- Turn-level `has_answer` only feeds questions.gold_note_ids, never note metadata.

Split: stratified by (question_type, abstention) with largest-remainder
allocation, seed 0 -> exactly 100 dev / 400 test on the 500 questions. The
split is keyed on question_id, so S, Oracle and M share it.

Usage:
  python -m mem27.data.longmemeval --src raw/longmemeval/longmemeval_s_cleaned.json --out data/lme_s
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from mem27.vschema import Note, Question, write_jsonl

SALT = "mem27-v1"
QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


def parse_lme_date(s: str) -> str:
    """'2023/05/20 (Sat) 02:21' -> '2023-05-20T02:21:00'."""
    s = s.strip()
    date_part, _, rest = s.partition(" (")
    time_part = rest.split(") ", 1)[1] if ") " in rest else "00:00"
    return datetime.strptime(f"{date_part} {time_part}", "%Y/%m/%d %H:%M").isoformat()


def anon_sid(orig_sid: str) -> str:
    return "s" + hashlib.sha1(f"{SALT}|{orig_sid}".encode()).hexdigest()[:12]


def is_abstention(qid: str) -> bool:
    return qid.endswith("_abs")


def convert_record(rec: dict) -> tuple[list[Note], Question, list[dict]]:
    qid = rec["question_id"]
    notes: list[Note] = []
    audit: list[dict] = []
    gold_notes: list[str] = []
    seen: Counter = Counter()
    sid_of_orig: dict[str, str] = {}
    sessions = rec["haystack_sessions"]
    for i, (orig_sid, date_raw, turns) in enumerate(
        zip(rec["haystack_session_ids"], rec["haystack_dates"], sessions)
    ):
        sid = anon_sid(orig_sid)
        seen[sid] += 1
        if seen[sid] > 1:  # duplicate session inside one haystack: keep both, disambiguate
            sid = f"{sid}d{seen[sid]}"
        sid_of_orig.setdefault(orig_sid, sid)
        ts = parse_lme_date(date_raw)
        audit.append({"history_id": qid, "session_id": sid, "orig_session_id": orig_sid, "session_idx": i})
        meta = {"date_raw": date_raw, "session_idx": i, "n_turns": len(turns)}
        for t, turn in enumerate(turns):
            nid = f"{qid}:{sid}:{t}"
            notes.append(Note(note_id=nid, history_id=qid, session_id=sid, turn_idx=t,
                              role=turn["role"], text=turn["content"], ts=ts, metadata=dict(meta)))
            if turn.get("has_answer"):
                gold_notes.append(nid)
        parent_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
        notes.append(Note(note_id=f"{qid}:{sid}:-1", history_id=qid, session_id=sid, turn_idx=-1,
                          role="session", text=parent_text, ts=ts, metadata=dict(meta)))
    gold_sessions = [sid_of_orig[s] for s in rec.get("answer_session_ids", []) if s in sid_of_orig]
    q = Question(
        question_id=qid,
        history_id=qid,
        question_type=rec["question_type"],
        question=rec["question"],
        question_date=parse_lme_date(rec["question_date"]),
        answer=str(rec["answer"]),
        gold_session_ids=gold_sessions,
        gold_note_ids=gold_notes,
        split="",
    )
    return notes, q, audit


def stratified_split(questions: list[Question], n_dev: int, seed: int = 0) -> dict[str, str]:
    """Largest-remainder allocation of n_dev over (type, abstention) strata."""
    strata: dict[tuple, list[str]] = defaultdict(list)
    for q in sorted(questions, key=lambda q: q.question_id):
        strata[(q.question_type, is_abstention(q.question_id))].append(q.question_id)
    n = len(questions)
    keys = sorted(strata)
    quota = {k: n_dev * len(strata[k]) / n for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    rest = n_dev - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(quota[k] - alloc[k]), k))[:rest]:
        alloc[k] += 1
    rng = random.Random(seed)
    split = {}
    for k in keys:
        ids = strata[k][:]
        rng.shuffle(ids)
        for j, qid in enumerate(ids):
            split[qid] = "dev" if j < alloc[k] else "test"
    return split


def iter_records(src: str | Path) -> Iterator[dict]:
    src = Path(src)
    if src.stat().st_size > 1_000_000_000:  # LongMemEval_M: stream it
        import ijson
        with open(src, "rb") as f:
            yield from ijson.items(f, "item", use_float=True)
    else:
        with open(src) as f:
            yield from json.load(f)


def convert(src: str | Path, out: str | Path, n_dev: int = 100, seed: int = 0,
            split_from: str | Path | None = None, count_tokens: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    audit_dir = out.parent / "_audit" / out.name
    audit_dir.mkdir(parents=True, exist_ok=True)
    enc = None
    if count_tokens:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

    questions: list[Question] = []
    tok_per_hist: list[int] = []
    n_turn = n_sess = 0
    with open(out / "notes.jsonl", "w") as fn, open(audit_dir / "session_id_map.jsonl", "w") as fa:
        for rec in iter_records(src):
            notes, q, audit = convert_record(rec)
            questions.append(q)
            toks = 0
            for nt in notes:
                fn.write(json.dumps(nt.__dict__, ensure_ascii=False) + "\n")
                if nt.turn_idx >= 0:
                    n_turn += 1
                    if enc is not None:
                        toks += len(enc.encode(nt.text, disallowed_special=()))
                else:
                    n_sess += 1
            tok_per_hist.append(toks)
            for a in audit:
                fa.write(json.dumps(a) + "\n")

    if split_from:  # reuse the S split so S / Oracle / M agree question-by-question
        ref = {json.loads(l)["question_id"]: json.loads(l)["split"] for l in open(split_from)}
        split = {q.question_id: ref[q.question_id] for q in questions}
    else:
        split = stratified_split(questions, n_dev=n_dev, seed=seed)
    for q in questions:
        q.split = split[q.question_id]
    write_jsonl(out / "questions.jsonl", questions)

    stats = {
        "source": str(src),
        "n_questions": len(questions),
        "n_notes": n_turn + n_sess,
        "n_turn_notes": n_turn,
        "n_session_notes": n_sess,
        "split": dict(Counter(q.split for q in questions)),
        "types": dict(Counter(q.question_type for q in questions)),
        "types_by_split": {s: dict(Counter(q.question_type for q in questions if q.split == s)) for s in ("dev", "test")},
        "abstention": sum(is_abstention(q.question_id) for q in questions),
        "abstention_by_split": dict(Counter(q.split for q in questions if is_abstention(q.question_id))),
        "questions_without_gold_turns": sum(not q.gold_note_ids for q in questions),
        "questions_without_gold_sessions": sum(not q.gold_session_ids for q in questions),
        "mean_gold_sessions": sum(len(q.gold_session_ids) for q in questions) / len(questions),
        "mean_gold_turns": sum(len(q.gold_note_ids) for q in questions) / len(questions),
    }
    if tok_per_hist and enc is not None:
        s = sorted(tok_per_hist)
        stats["history_tokens_cl100k"] = {"mean": sum(s) / len(s), "min": s[0], "median": s[len(s) // 2], "max": s[-1]}
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-from", default=None, help="questions.jsonl whose split to reuse")
    ap.add_argument("--no-tokens", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(convert(a.src, a.out, a.n_dev, a.seed, a.split_from, not a.no_tokens), indent=2))


if __name__ == "__main__":
    main()
