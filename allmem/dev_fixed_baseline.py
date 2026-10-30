"""Same-split (dev) fixed-plan reader QA, for an apples-to-apples comparison with the
All-Mem (from-paper reimpl) dev number. Reuses the identical reader/judge prompts and
the :8001 endpoint, over the pre-computed ref:p* retrieval bundles (dev subset only).

  .venv/bin/python -m allmem.dev_fixed_baseline --plans ref:p5 \
      --out runs/analysis/allmem_devfixed.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from allmem.allmem import LLM, reader_prompt, judge_prompt, READER_SYS, JUDGE_SYS

DATA = "data/lme_s_k"


def load_jsonl(p):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="+", default=["ref:p5"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="runs/analysis/allmem_devfixed.json")
    args = ap.parse_args()

    qs = {q["question_id"]: q for q in load_jsonl(os.path.join(DATA, "questions.jsonl"))
          if q.get("split") == args.split}
    llm = LLM()
    results = {}
    for plan in args.plans:
        pdir = "runs/ref_" + plan.split(":")[1] + "_lme_s_k_seed0_warm"
        bundles = {b["question_id"]: b for b in load_jsonl(os.path.join(pdir, "bundles.jsonl"))
                   if b["question_id"] in qs}

        def work(qid):
            q = qs[qid]
            b = bundles[qid]
            pred = (llm.chat(READER_SYS, reader_prompt(q["question"], b["passages"]),
                             max_tokens=96) or "I don't know").strip()
            jraw = llm.chat(JUDGE_SYS, judge_prompt(q["question"], q.get("answer", ""), pred),
                            max_tokens=4)
            corr = 1 if re.search(r"correct", jraw, re.I) and not re.search(r"wrong", jraw, re.I) else 0
            return qid, q["question_type"], corr

        acc, by_type = [], defaultdict(list)
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(work, qid) for qid in bundles]
            for f in as_completed(futs):
                qid, qt, corr = f.result()
                acc.append(corr); by_type[qt].append(corr)
        results[plan] = {
            "n": len(acc),
            "qa_accuracy": round(sum(acc) / len(acc), 4) if acc else None,
            "per_type": {t: {"n": len(v), "qa_accuracy": round(sum(v) / len(v), 4)}
                         for t, v in sorted(by_type.items())},
        }
        print(f"{plan} {args.split}: acc={results[plan]['qa_accuracy']} n={len(acc)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"split": args.split, "note": "fixed-plan reader QA, same prompts/endpoint as All-Mem reimpl",
               "plans": results}, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
