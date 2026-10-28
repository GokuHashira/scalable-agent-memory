"""LongMemEval knowledge-update: LLM explicit/implicit labels and stale-served judgments (EVAL-03).

  label: adds explicit_llm to data/<ds>/ku_pairs.jsonl (configs/prompts/ku_explicit.txt)
  stale: for each run, adds answers[].judges["stale"] (configs/prompts/stale_judge.txt) on KU questions;
         "correct"=True there means the answer served the OUTDATED value.

  python scripts/ku_llm_labels.py label --data data/lme_s --endpoint runs/endpoints/judge
  python scripts/ku_llm_labels.py stale --data data/lme_s --runs runs/vref_bm25_lme_s --answers answers.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import fill, fmt_date  # noqa: E402
from mem27.eval.judge import extra_body_for, parse_label  # noqa: E402
from mem27.eval.llm import ChatLLM  # noqa: E402
from mem27.vschema import Question, iter_jsonl, read_jsonl, write_jsonl  # noqa: E402

MAX_CHARS = 3000  # evidence turns can be very long; the changed fact is usually near the start


def _texts(data: Path, ids: set[str]) -> dict[str, tuple[str, str]]:
    return {n["note_id"]: (n["text"], n["ts"]) for n in iter_jsonl(data / "notes.jsonl") if n["note_id"] in ids}


def _llm(endpoint: str, cache: str) -> ChatLLM:
    probe = ChatLLM(endpoint)
    return ChatLLM(endpoint, model=probe.model, extra_body=extra_body_for(probe.model), cache=cache, max_workers=32)


def label(a):
    data = Path(a.data)
    pairs = list(iter_jsonl(data / "ku_pairs.jsonl"))
    tx = _texts(data, {i for p in pairs for i in p["old_note_ids"] + p["new_note_ids"]})
    tmpl = Path("configs/prompts/ku_explicit.txt").read_text()
    msgs = []
    for p in pairs:
        old = "\n".join(tx[i][0][:MAX_CHARS] for i in p["old_note_ids"])
        new = "\n".join(tx[i][0][:MAX_CHARS] for i in p["new_note_ids"])
        msgs.append([{"role": "user", "content": fill(tmpl, old=old, new=new, old_date=fmt_date(p["old_ts"]),
                                                       new_date=fmt_date(p["new_ts"]))}])
    llm = _llm(a.endpoint, "runs/_cache/ku_label.sqlite")
    outs = llm.map(msgs, max_tokens=_max_tokens(llm))
    for p, o in zip(pairs, outs):
        p["explicit_llm"] = parse_label(o.get("text", ""))
        p["explicit_llm_model"] = llm.model
    write_jsonl(data / "ku_pairs.jsonl", pairs)
    n = sum(p["explicit_llm"] for p in pairs)
    agree = sum(p["explicit_llm"] == p["explicit_heuristic"] for p in pairs)
    print(f"explicit_llm={n}/{len(pairs)}  heuristic={sum(p['explicit_heuristic'] for p in pairs)}  agree={agree}")


def _max_tokens(llm) -> int:
    # gpt-oss reasons before answering: 16 tokens leaves an empty answer (Oct 14: every stale verdict came back
    # empty and parsed as "no"). Same budget as the main judge (mem27/eval/judge.py).
    return 512 if "gpt-oss" in llm.model.lower() else 16


def stale(a):
    data = Path(a.data)
    pairs = {p["question_id"]: p for p in iter_jsonl(data / "ku_pairs.jsonl")}
    qs = {q.question_id: q for q in read_jsonl(data / "questions.jsonl", Question)}
    tx = _texts(data, {i for p in pairs.values() for i in p["old_note_ids"] + p["new_note_ids"]})
    tmpl = Path("configs/prompts/stale_judge.txt").read_text()
    llm = _llm(a.endpoint, "runs/_cache/stale_judge.sqlite")
    for run in a.runs:
        path = Path(run) / a.answers
        rows = list(iter_jsonl(path))
        idx, msgs = [], []
        for i, r in enumerate(rows):
            p = pairs.get(r["question_id"])
            if p is None:
                continue
            q = qs[r["question_id"]]
            old = "\n".join(tx[n][0][:MAX_CHARS] for n in p["old_note_ids"])
            new = "\n".join(tx[n][0][:MAX_CHARS] for n in p["new_note_ids"])
            msgs.append([{"role": "user", "content": fill(tmpl, question=q.question, old=old, new=new, answer=q.answer,
                                                           response=r.get("answer") or "")}])
            idx.append(i)
        outs = llm.map(msgs, max_tokens=_max_tokens(llm))
        n_empty = 0
        for i, o in zip(idx, outs):
            text = (o.get("text") or "").strip()
            empty = not text or bool(o.get("error"))
            n_empty += empty
            rows[i].setdefault("judges", {})["stale"] = {"correct": None if empty else parse_label(text),
                                                          "judge_model": llm.model, "raw": text,
                                                          "error": o.get("error") or ("empty output" if empty else None)}
        write_jsonl(path, rows)
        n_st = sum(bool(rows[i]["judges"]["stale"]["correct"]) for i in idx)
        print(f"{Path(run).name}: stale-served {n_st}/{len(idx) - n_empty} (empty / failed verdicts: {n_empty})")
        if n_empty:
            print(f"WARNING {Path(run).name}: {n_empty} stale verdicts are empty or failed; they are excluded")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("label")
    l.add_argument("--data", required=True)
    l.add_argument("--endpoint", default="runs/endpoints/judge")
    s = sub.add_parser("stale")
    s.add_argument("--data", required=True)
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--answers", default="answers.jsonl")
    s.add_argument("--endpoint", default="runs/endpoints/judge")
    a = ap.parse_args(argv)
    label(a) if a.cmd == "label" else stale(a)


if __name__ == "__main__":
    main()
