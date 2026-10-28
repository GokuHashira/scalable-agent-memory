"""Reader wrapper (EVAL-01): bundles.jsonl -> answers.jsonl via the vLLM endpoint.

Pinned prompt (configs/prompts/reader*.txt), greedy decoding, batched, retried,
cached. The answer passed to the judge is the "Final answer:" line when the
prompt asks for one, else the whole response; both are stored.

  python -m mem27.eval.reader --run runs/gold_lme_s --data data/lme_s \
      --endpoint runs/vllm_endpoint.txt --prompt configs/prompts/reader.txt --split dev
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from mem27.eval.common import NoteIndex, fill, fmt_date, format_history, load_prompt
from mem27.eval.llm import ChatLLM
from mem27.vschema import Question, iter_jsonl, read_jsonl, write_jsonl

FINAL_RE = re.compile(r"final answer\s*[:：]\s*(.+)", re.I | re.S)
QWEN3_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def extract_final(text: str) -> str:
    m = list(FINAL_RE.finditer(text or ""))
    if not m:
        return (text or "").strip()
    return m[-1].group(1).strip().strip("*").strip()


def build_messages(q: Question, bundle: dict, template: str, idx: NoteIndex | None, order: str,
                   date_hints: bool = False) -> list[dict]:
    history = format_history(bundle.get("note_ids", []), bundle.get("passages", []), idx, order=order,
                             question_date=q.question_date, date_hints=date_hints)
    prompt = fill(template, history=history, question_date=fmt_date(q.question_date), question=q.question)
    return [{"role": "user", "content": prompt}]


def run_reader(run_dir: str | Path, data_dir: str | Path, endpoint: str, prompt_path: str | Path,
               split: str | None = None, order: str = "chrono", max_tokens: int | None = None,
               model: str | None = None, out_name: str = "answers.jsonl", cache: str | None = None,
               think: bool = False, workers: int = 64, limit_qids: set | None = None,
               date_hints: bool = False, reuse: bool = False) -> Path:
    """reuse=True (resume): keep rows of an existing `out_name` that have no reader error and were made
    with the same prompt / order / think / date_hints / model (and, when stored, the same messages);
    only the rest go to the server. Off by default."""
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    qs = {q.question_id: q for q in read_jsonl(data_dir / "questions.jsonl", Question)}
    bundles = [b for b in iter_jsonl(run_dir / "bundles.jsonl")
               if b["question_id"] in qs and (split is None or split == "all" or qs[b["question_id"]].split == split)
               and (limit_qids is None or b["question_id"] in limit_qids)]
    histories = {qs[b["question_id"]].history_id for b in bundles}
    # streaming datasets keep later notes in updates.jsonl; notes_final.jsonl has them all
    notes_file = data_dir / "notes_final.jsonl" if (data_dir / "notes_final.jsonl").exists() else data_dir / "notes.jsonl"
    idx = NoteIndex(notes_file, histories=histories)
    template, prompt_sha = load_prompt(prompt_path)
    wants_final = "Final answer" in template
    if max_tokens is None:
        max_tokens = 1024 if wants_final else 256
    if think:
        max_tokens = max(max_tokens, 4096)
    extra = None if think else QWEN3_NO_THINK
    llm = ChatLLM(endpoint, model=model, cache=cache or (Path(run_dir).parents[0] / "_cache" / "reader.sqlite"),
                  max_workers=workers, extra_body=extra)
    msgs = [build_messages(qs[b["question_id"]], b, template, idx, order, date_hints) for b in bundles]
    msg_shas = [hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()[:16] for m in msgs]
    old = _reusable_rows(run_dir / out_name, prompt_sha, order, think, date_hints, llm.model) if reuse else {}
    keep = [old.get(b["question_id"]) for b in bundles]
    keep = [r if r is not None and r["reader"].get("msg_sha", ms) == ms else None for r, ms in zip(keep, msg_shas)]
    todo = [i for i, r in enumerate(keep) if r is None]
    if reuse:
        print(f"[reader] reuse {len(bundles) - len(todo)}/{len(bundles)} rows of {run_dir / out_name}; "
              f"{len(todo)} to read", file=sys.stderr, flush=True)
    t0 = time.time()
    todo_outs = llm.map([msgs[i] for i in todo], max_tokens=max_tokens,
                        progress=lambda d, n: print(f"[reader] {d}/{n} {time.time() - t0:.0f}s", file=sys.stderr, flush=True))
    outs: list[dict | None] = [None] * len(bundles)
    for i, o in zip(todo, todo_outs):
        outs[i] = o
    rows = []
    for b, o, k, ms in zip(bundles, outs, keep, msg_shas):
        if o is None:
            rows.append(k)
            continue
        resp = o.get("text", "")
        rows.append({
            "question_id": b["question_id"],
            "plan": b.get("plan"),
            "answer": extract_final(resp) if wants_final else resp.strip(),
            "response": resp,
            "reader": {"model": o.get("model"), "prompt": str(prompt_path), "prompt_sha": prompt_sha,
                       "order": order, "think": think, "date_hints": date_hints, "prompt_tokens": o.get("prompt_tokens"),
                       "completion_tokens": o.get("completion_tokens"), "latency_ms": o.get("latency_ms"),
                       "finish_reason": o.get("finish_reason"), "error": o.get("error"), "msg_sha": ms},
            "judge": None,
        })
    out = run_dir / out_name
    write_jsonl(out, rows)
    n_err = sum(1 for r in rows if r["reader"]["error"])
    print(f"[reader] wrote {len(rows)} answers to {out} ({n_err} errors) in {time.time() - t0:.0f}s", file=sys.stderr)
    return out


def _reusable_rows(path: Path, prompt_sha: str, order: str, think: bool, date_hints: bool,
                   model: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    bundles = path.parent / "bundles.jsonl"
    if bundles.exists() and bundles.stat().st_mtime > path.stat().st_mtime:
        print(f"[reader] not reusing {path}: bundles.jsonl is newer", file=sys.stderr)
        return {}
    out = {}
    for r in iter_jsonl(path):
        rd = r.get("reader") or {}
        if (not rd.get("error") and rd.get("prompt_sha") == prompt_sha and rd.get("order") == order
                and bool(rd.get("think")) == think and bool(rd.get("date_hints")) == date_hints
                and rd.get("model") == model):
            out[r["question_id"]] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--endpoint", default="runs/vllm_endpoint.txt")
    ap.add_argument("--prompt", default="configs/prompts/reader.txt")
    ap.add_argument("--split", default=None, help="dev | test | all")
    ap.add_argument("--order", default="chrono", choices=["chrono", "rank"])
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out-name", default="answers.jsonl")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--date-hints", action="store_true")
    a = ap.parse_args(argv)
    run_reader(a.run, a.data, a.endpoint, a.prompt, a.split, a.order, a.max_tokens, a.model, a.out_name,
               a.cache, a.think, a.workers, date_hints=a.date_hints)


if __name__ == "__main__":
    main()
