"""QA-level reader + LLM judge over retrieved bundles, via vLLM (planner.md §5).

Turns the evidence-level study into a QA-level one: for each (question, fixed plan),
feed the plan's retrieved passages to an instruct model, generate an answer, then judge
it against the gold answer with the same model. Writes an `answers.jsonl` in the shared
schema ({question_id, plan, answer, judge{correct, judge_model, raw}}) so the oracle can
be re-run on `correct`-based labels — exactly V's format.

Run inside the vLLM venv:
  .venv_vllm/bin/python -m mem27.reader.vllm_reader \
      --data data/lme_s_k --runs 'runs/ref_p*_lme_s_k_seed0_warm' \
      --split test --model Qwen/Qwen2.5-7B-Instruct --out runs/analysis/answers_vllm.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

from mem27.schema import load_questions, load_bundles, Answer, write_jsonl

READER_SYS = (
    "You are answering questions about a user based ONLY on excerpts from their past "
    "conversations with an assistant. Use only the excerpts. If the answer is not present, "
    "reply exactly 'I don't know'. Answer in as few words as possible — just the fact."
)
JUDGE_SYS = (
    "You grade whether a predicted answer matches the gold answer for a question. "
    "Reply with a single word: CORRECT if the prediction conveys the same key fact as the "
    "gold answer (minor wording/format differences are fine), otherwise WRONG."
)


def reader_prompt(question, passages):
    ctx = "\n".join(f"- {p}" for p in passages)
    return (f"Conversation excerpts:\n{ctx}\n\nQuestion: {question}\nAnswer:")


def judge_prompt(question, gold, pred):
    return (f"Question: {question}\nGold answer: {gold}\nPredicted answer: {pred}\n\n"
            f"Is the predicted answer correct? Reply CORRECT or WRONG.")


def load_runs(run_dirs):
    out = {}
    for pat in run_dirs:
        for d in sorted(glob.glob(pat)) or [pat]:
            bp = os.path.join(d, "bundles.jsonl")
            if os.path.exists(bp):
                bs = load_bundles(bp)
                if bs:
                    out[bs[0].plan] = {b.question_id: b for b in bs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lme_s_k")
    ap.add_argument("--runs", nargs="+", default=["runs/ref_p*_lme_s_k_seed0_warm"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="runs/analysis/answers_vllm.jsonl")
    ap.add_argument("--max-q", type=int, default=0, help="limit questions (debug)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tp", type=int, default=1)
    args = ap.parse_args()

    questions = [q for q in load_questions(os.path.join(args.data, "questions.jsonl"))
                 if (not args.split or q.split == args.split)]
    if args.max_q:
        questions = questions[:args.max_q]
    qmap = {q.question_id: q for q in questions}
    runs = load_runs(args.runs)
    plans = sorted(runs)
    print(f"questions={len(questions)} plans={plans}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_model_len, tensor_parallel_size=args.tp,
              enforce_eager=False)
    tok = llm.get_tokenizer()

    def chat(system, user):
        return tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

    # ---- 1. reader generations ----
    reader_keys, reader_prompts = [], []
    for q in questions:
        for p in plans:
            b = runs[p].get(q.question_id)
            if not b:
                continue
            reader_keys.append((q.question_id, p))
            reader_prompts.append(chat(READER_SYS, reader_prompt(q.question, b.passages)))
    sp = SamplingParams(temperature=0.0, max_tokens=96)
    print(f"reader generations: {len(reader_prompts)}", flush=True)
    r_out = llm.generate(reader_prompts, sp)
    preds = {k: o.outputs[0].text.strip() for k, o in zip(reader_keys, r_out)}

    # ---- 2. judge ----
    judge_keys, judge_prompts = [], []
    for (qid, p), pred in preds.items():
        q = qmap[qid]
        judge_keys.append((qid, p))
        judge_prompts.append(chat(JUDGE_SYS, judge_prompt(q.question, q.answer, pred)))
    spj = SamplingParams(temperature=0.0, max_tokens=4)
    print(f"judge generations: {len(judge_prompts)}", flush=True)
    j_out = llm.generate(judge_prompts, spj)
    verdict = {}
    for k, o in zip(judge_keys, j_out):
        raw = o.outputs[0].text.strip()
        verdict[k] = (1 if re.search(r"correct", raw, re.I) and not re.search(r"wrong", raw, re.I) else 0, raw)

    # ---- 3. write answers.jsonl ----
    rows = []
    for (qid, p), pred in preds.items():
        corr, raw = verdict[(qid, p)]
        rows.append(Answer(question_id=qid, plan=p, answer=pred,
                           judge={"correct": corr, "judge_model": args.model, "raw": raw}))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_jsonl(args.out, rows)

    # quick per-plan accuracy summary
    from collections import defaultdict
    acc = defaultdict(list)
    for (qid, p), _ in preds.items():
        acc[p].append(verdict[(qid, p)][0])
    print("=== per-plan QA accuracy (judge correct) ===")
    for p in plans:
        print(f"  {p}: {sum(acc[p])/len(acc[p]):.4f}  (n={len(acc[p])})")
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
