"""Judge wrapper (EVAL-01): LongMemEval judge prompts over answers.jsonl.

Open-weight judges only, served by vLLM or llama.cpp (OpenAI-compatible). Judgments
are cached by sha256(judge model, prompt) -- i.e. by (question_id, answer) -- so
re-judging an unchanged answer is free. `--as primary` fills answers[].judge;
any other name fills answers[].judges[name] (second-judge robustness).

`--promote NAME`: copy answers[].judges[NAME] into answers[].judge (the label every
metric uses). Pinned choice, Oct 13: NAME = j2 = gpt-oss-20b (reasoning low). On 21
author-adjudicated dev disagreements it was right 18 times vs 3 for the non-thinking
Qwen3-32B; the two agree on 96% of items (kappa 0.91). Qwen3-32B stays as judges["primary"].

`--ensemble` (not pinned): final label = majority of {primary (Qwen3-32B), j2 (gpt-oss-20b)}, and
where they disagree a third vote from the primary model with reasoning enabled
("tiebreak"). answers[].judge then holds the ensemble label; every vote stays in
answers[].judges. Motivation: on dev, the non-thinking 32B judge accepted ~10% of
"the information is not available" answers to answerable questions.

  python -m mem27.eval.judge --run runs/gold_lme_s --data data/lme_s \
      --endpoint runs/vllm_judge_endpoint.txt
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from mem27.eval.llm import ChatLLM
from mem27.vschema import Question, iter_jsonl, read_jsonl, write_jsonl

SECTION_RE = re.compile(r"^### (\S+)\n", re.M)


def load_judge_prompts(path: str | Path) -> dict[str, str]:
    text = Path(path).read_text()
    parts = SECTION_RE.split(text)
    return {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts), 2)}


def judge_prompt(prompts: dict[str, str], q: Question, response: str) -> str:
    if q.question_id.endswith("_abs"):
        key = "abstention"
    elif q.question_type in prompts:
        key = q.question_type
    else:
        key = "default"
    return (prompts[key].replace("{question}", q.question).replace("{answer}", q.answer)
            .replace("{response}", response))


def parse_label(text: str) -> bool:
    t = (text or "").strip().lower()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()
    return "yes" in t  # official LongMemEval rule


def extra_body_for(model: str) -> dict | None:
    m = model.lower()
    if "qwen3" in m:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if "gpt-oss" in m:
        return {"reasoning_effort": "low"}
    return None


def run_judge(run_dir: str | Path, data_dir: str | Path, endpoint: str, prompts_path: str | Path,
              name: str = "primary", answers_name: str = "answers.jsonl", model: str | None = None,
              cache: str | None = None, workers: int = 64, field: str = "answer") -> Path:
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    qs = {q.question_id: q for q in read_jsonl(data_dir / "questions.jsonl", Question)}
    prompts = load_judge_prompts(prompts_path)
    rows = list(iter_jsonl(run_dir / answers_name))
    probe = ChatLLM(endpoint, model=model, cache=None)
    llm = ChatLLM(endpoint, model=probe.model, max_workers=workers, extra_body=extra_body_for(probe.model),
                  cache=cache or (run_dir.parents[0] / "_cache" / "judge.sqlite"))
    max_tokens = 512 if "gpt-oss" in llm.model.lower() else 16
    msgs = [[{"role": "user", "content": judge_prompt(prompts, qs[r["question_id"]], r.get(field) or "")}]
            for r in rows]
    t0 = time.time()
    outs = llm.map(msgs, max_tokens=max_tokens,
                   progress=lambda d, n: print(f"[judge] {d}/{n} {time.time() - t0:.0f}s", file=sys.stderr, flush=True))
    for r, o in zip(rows, outs):
        j = {"correct": parse_label(o.get("text", "")), "judge_model": llm.model, "raw": o.get("text", ""),
             "field": field, "error": o.get("error")}
        if name == "primary":
            r["judge"] = j
        r.setdefault("judges", {})[name] = j
    write_jsonl(run_dir / answers_name, rows)
    acc = sum(r["judges"][name]["correct"] for r in rows) / max(1, len(rows))
    print(f"[judge:{name}] {llm.model} acc={acc:.4f} n={len(rows)} {time.time() - t0:.0f}s", file=sys.stderr)
    return run_dir / answers_name


def promote(run_dir: str | Path, answers_name: str, name: str) -> Path:
    path = Path(run_dir) / answers_name
    rows = list(iter_jsonl(path))
    n = 0
    for r in rows:
        j = (r.get("judges") or {}).get(name)
        if j is not None:
            r["judge"] = dict(j)
            n += 1
    write_jsonl(path, rows)
    print(f"[promote] {name} -> judge on {n}/{len(rows)} rows of {path}", file=sys.stderr)
    return path


def run_ensemble(run_dir: str | Path, data_dir: str | Path, endpoint: str, prompts_path: str | Path,
                 answers_name: str = "answers.jsonl", workers: int = 32) -> Path:
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    qs = {q.question_id: q for q in read_jsonl(data_dir / "questions.jsonl", Question)}
    prompts = load_judge_prompts(prompts_path)
    rows = list(iter_jsonl(run_dir / answers_name))
    todo = [i for i, r in enumerate(rows)
            if {"primary", "j2"} <= set(r.get("judges") or {}) and
            r["judges"]["primary"]["correct"] != r["judges"]["j2"]["correct"] and "tiebreak" not in r["judges"]]
    if todo:
        probe = ChatLLM(endpoint)
        llm = ChatLLM(endpoint, model=probe.model, max_workers=workers,
                      extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                      cache=run_dir.parents[0] / "_cache" / f"judge_tiebreak.{run_dir.name}.sqlite")
        msgs = [[{"role": "user", "content": judge_prompt(prompts, qs[rows[i]["question_id"]], rows[i].get("answer") or "")}]
                for i in todo]
        outs = llm.map(msgs, max_tokens=3072)
        for i, o in zip(todo, outs):
            text = o.get("text", "")
            rows[i]["judges"]["tiebreak"] = {"correct": parse_label(text.split("</think>")[-1]),
                                             "judge_model": llm.model + "+think", "raw": text[-200:], "error": o.get("error")}
    n_dis = 0
    for r in rows:
        js = r.get("judges") or {}
        if not {"primary", "j2"} <= set(js):
            continue
        votes = [js["primary"]["correct"], js["j2"]["correct"]]
        if votes[0] != votes[1]:
            n_dis += 1
            votes.append(js.get("tiebreak", js["primary"])["correct"])
        r["judge"] = {"correct": sum(votes) * 2 > len(votes), "judge_model": "ensemble(qwen3-32b,gpt-oss-20b;tiebreak qwen3-32b-think)",
                      "raw": "/".join("yes" if v else "no" for v in votes)}
    write_jsonl(run_dir / answers_name, rows)
    print(f"[ensemble] {run_dir.name}: {n_dis}/{len(rows)} disagreements, tiebroken {len(todo)} new", file=sys.stderr)
    return run_dir / answers_name


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--endpoint", default="runs/vllm_judge_endpoint.txt")
    ap.add_argument("--prompts", default="configs/prompts/judge.txt")
    ap.add_argument("--as", dest="name", default="primary")
    ap.add_argument("--answers", default="answers.jsonl")
    ap.add_argument("--model", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--field", default="answer", choices=["answer", "response"])
    ap.add_argument("--ensemble", action="store_true", help="tiebreak primary vs j2 with the endpoint's model + thinking")
    ap.add_argument("--promote", default=None, help="copy judges[NAME] into judge")
    a = ap.parse_args(argv)
    if a.promote:
        promote(a.run, a.answers, a.promote)
    elif a.ensemble:
        run_ensemble(a.run, a.data, a.endpoint, a.prompts, a.answers, a.workers)
    else:
        run_judge(a.run, a.data, a.endpoint, a.prompts, a.name, a.answers, a.model, a.cache, a.workers, a.field)


if __name__ == "__main__":
    main()
