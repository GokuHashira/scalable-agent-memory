"""Seeded sample of judge disagreements (primary = Qwen3-32B vs j2 = gpt-oss-20b) for author adjudication, and a
score of each judge against the adjudicated labels.

  python scripts/judge_disagreements.py sample --runs runs/gold_sess_lme_s runs/vref_bm25_lme_s runs/vref_hybrid_lme_s \
      --data data/lme_s --split test --n 24 --out runs/_tables/judge_disagreement_sample_test.json
  python scripts/judge_disagreements.py score --sample runs/_tables/judge_disagreement_sample_test.json \
      --labels runs/_tables/judge_adjudication_pinned_test.json
Labels file: {"labels": {"<run>|<question_id>": 1 (answer correct) | 0 (wrong) | null (ambiguous)}, "notes": {...}}.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import write_json  # noqa: E402
from mem27.vschema import iter_jsonl  # noqa: E402


def sample(a):
    qs = {q["question_id"]: q for q in iter_jsonl(Path(a.data) / "questions.jsonl")}
    pool, tot, by_type, dis_type = [], 0, Counter(), Counter()
    for run in a.runs:
        for r in iter_jsonl(Path(run) / "answers.jsonl"):
            q = qs[r["question_id"]]
            if a.split != "all" and q["split"] != a.split:
                continue
            js = r.get("judges") or {}
            if not {"primary", "j2"} <= set(js):
                continue
            tot += 1
            by_type[q["question_type"]] += 1
            if js["primary"]["correct"] != js["j2"]["correct"]:
                dis_type[q["question_type"]] += 1
                pool.append({"key": f"{Path(run).name}|{q['question_id']}", "type": q["question_type"],
                             "qwen": js["primary"]["correct"], "oss": js["j2"]["correct"], "question": q["question"],
                             "gold": q["answer"], "answer": r.get("answer")})
    random.Random(a.seed).shuffle(pool)
    out = {"runs": [Path(r).name for r in a.runs], "split": a.split, "seed": a.seed, "n_answers": tot,
           "n_disagree": len(pool), "disagree_rate": len(pool) / max(1, tot),
           "disagree_rate_by_type": {t: dis_type[t] / by_type[t] for t in by_type},
           "qwen_yes_oss_no": sum(1 for p in pool if p["qwen"]), "sample": pool[:a.n]}
    write_json(a.out, out)
    print(json.dumps({k: v for k, v in out.items() if k != "sample"}, indent=1))


def score(a):
    s = json.loads(Path(a.sample).read_text())
    lab = json.loads(Path(a.labels).read_text())["labels"]
    c = Counter()
    for p in s["sample"]:
        y = lab.get(p["key"])
        if y is None:
            c["ambiguous"] += 1
            continue
        c["n"] += 1
        c["qwen_right"] += int(p["qwen"] == bool(y))
        c["oss_right"] += int(p["oss"] == bool(y))
        if p["type"] == "single-session-preference":
            c["pref_n"] += 1
            c["pref_oss_right"] += int(p["oss"] == bool(y))
    print(dict(c))
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--split", default="test")
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    c = sub.add_parser("score")
    c.add_argument("--sample", required=True)
    c.add_argument("--labels", required=True)
    a = ap.parse_args(argv)
    sample(a) if a.cmd == "sample" else score(a)


if __name__ == "__main__":
    main()
