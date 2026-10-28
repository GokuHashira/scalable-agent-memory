"""Updates (DATA-03): sequential-feed harness + LongMemEval knowledge-update supersession pairs.

Sequential feeding: updates are applied strictly in timestamp order, one at a
time (never a full dump). An explicit supersede/correct closes the predecessor
(valid_to = update.ts) -- exactly what a system can know; implicit ones close
nothing, the system has to infer them.

stream_eval() interleaves updates.jsonl with questions_stream.jsonl: before each
question every update with ts <= question_date is fed, then retrieve() runs.
It writes bundles.jsonl (+ insert_traces.jsonl, the maintenance work per update).

LongMemEval: ku_pairs() derives natural supersession pairs from the
knowledge-update questions (old value = gold turns in earlier gold sessions, new
value = gold turns in the latest gold session), with a lexical explicit/implicit
heuristic; an LLM label can be added later (configs/prompts/ku_explicit.txt).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Iterator

from mem27.vschema import EvidenceBundle, Note, Question, RetrievalRequest, iter_jsonl, read_jsonl, session_of, write_jsonl

NOTE_FIELDS = set(Note.__dataclass_fields__)


def to_note(d: dict) -> Note:
    return Note(**{k: v for k, v in d.items() if k in NOTE_FIELDS})


class NoteStore:
    """Reference in-memory store: what apply_update() must do to the representation."""

    def __init__(self, notes: Iterable[Note] = ()):
        self.notes: dict[str, Note] = {}
        self.last_ts = ""
        for n in notes:
            self.notes[n.note_id] = n
            self.last_ts = max(self.last_ts, n.ts)

    def apply_update(self, up: dict) -> dict:
        if up["ts"] < self.last_ts:
            raise ValueError(f"update {up['update_id']} out of order: {up['ts']} < {self.last_ts}")
        self.last_ts = up["ts"]
        note = to_note(up["note"])
        closed = None
        if up["op"] in ("supersede", "correct") and up.get("explicit") and up.get("supersedes_note_id"):
            old = self.notes.get(up["supersedes_note_id"])
            if old is not None and old.valid_to is None:
                old.valid_to = up["ts"]
                closed = old.note_id
            note.supersedes_note_id = up["supersedes_note_id"]
        self.notes[note.note_id] = note
        return {"update_id": up["update_id"], "op": up["op"], "explicit": bool(up.get("explicit")),
                "closed": closed}


def iter_updates(path: str | Path) -> Iterator[dict]:
    prev = ""
    for u in iter_jsonl(path):
        if u["ts"] < prev:
            raise ValueError(f"updates.jsonl not sorted at {u['update_id']}")
        prev = u["ts"]
        yield u


def stream_eval(retriever, data_dir: str | Path, run_dir: str | Path, plan: str,
                questions_file: str = "questions_stream.jsonl", budget_tokens: int = 2048,
                max_results: int = 20, base_file: str = "notes.jsonl") -> dict:
    """Feed base notes, then updates in ts order, answering each question at its question_date."""
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    base = [to_note(n) for n in iter_jsonl(data_dir / base_file)]
    t0 = time.perf_counter()
    retriever.build(base, None)
    build_ms = (time.perf_counter() - t0) * 1000
    qs = sorted(iter_jsonl(data_dir / questions_file), key=lambda q: q["question_date"])
    ups = iter_updates(data_dir / "updates.jsonl")
    pending = next(ups, None)
    bundles, traces = [], []
    for q in qs:
        while pending is not None and pending["ts"] <= q["question_date"]:
            tr = retriever.apply_update(pending) if hasattr(retriever, "apply_update") else \
                retriever.insert(to_note(pending["note"]))
            traces.append({"update_id": pending["update_id"], "op": pending["op"],
                           "explicit": pending.get("explicit"), **(tr or {})})
            pending = next(ups, None)
        req = RetrievalRequest(query=q["question"], budget_tokens=budget_tokens, max_results=max_results,
                               as_of_time=q.get("as_of_time") or q["question_date"], plan=plan,
                               question_id=q["question_id"])
        b: EvidenceBundle = retriever.retrieve(req)
        row = asdict(b) if not isinstance(b, dict) else dict(b)
        row.update({"question_id": q["question_id"], "plan": plan})
        bundles.append(row)
    write_jsonl(run_dir / "bundles.jsonl", bundles)
    write_jsonl(run_dir / "insert_traces.jsonl", traces)
    info = {"n_questions": len(qs), "n_updates_fed": len(traces), "build_ms": build_ms}
    (run_dir / "stream_info.json").write_text(json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------- LongMemEval KU pairs

CHANGE_CUES = re.compile(
    r"\b(used to|no longer|anymore|any more|instead of|switched|changed|moved|upgraded|downgraded|replaced|"
    r"now (?:have|own|use|live|work|am|is|are)|recently (?:got|switched|changed|moved|bought|started)|"
    r"previously|before,? (?:i|my)|went from|from \w+ to)\b", re.I)


def ku_pairs(data_dir: str | Path) -> list[dict]:
    data_dir = Path(data_dir)
    qs = [q for q in read_jsonl(data_dir / "questions.jsonl", Question) if q.question_type == "knowledge-update"]
    need = {g for q in qs for g in q.gold_note_ids}
    text, ts = {}, {}
    for n in iter_jsonl(data_dir / "notes.jsonl"):
        if n["note_id"] in need:
            text[n["note_id"]], ts[n["note_id"]] = n["text"], n["ts"]
    out = []
    for q in qs:
        gold = sorted((g for g in q.gold_note_ids if g in ts), key=lambda g: (ts[g], g))
        sess = sorted({session_of(g) for g in gold}, key=lambda s: min(ts[g] for g in gold if session_of(g) == s))
        if len(sess) < 2:
            continue
        new = [g for g in gold if session_of(g) == sess[-1]]
        old = [g for g in gold if session_of(g) != sess[-1]]
        cue = CHANGE_CUES.search(" ".join(text[g] for g in new))
        out.append({"question_id": q.question_id, "split": q.split, "abstention": q.question_id.endswith("_abs"),
                    "old_note_ids": old, "new_note_ids": new, "old_ts": ts[old[-1]], "new_ts": ts[new[0]],
                    "explicit_heuristic": bool(cue), "cue": cue.group(0) if cue else None})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ku-pairs", help="LongMemEval data dir; writes ku_pairs.jsonl there")
    a = ap.parse_args(argv)
    if a.ku_pairs:
        rows = ku_pairs(a.ku_pairs)
        write_jsonl(Path(a.ku_pairs) / "ku_pairs.jsonl", rows)
        print(f"{len(rows)} KU pairs; explicit(heuristic)={sum(r['explicit_heuristic'] for r in rows)}")


if __name__ == "__main__":
    main()
