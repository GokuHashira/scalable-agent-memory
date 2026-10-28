"""Judge spot-check sheet (Slice V §2.5): 20 random dev answers (seed 0) across the Slice V bundles, as an editable
markdown table with the judge verdicts, an empty human-label column and an LLM-proposed provisional label.

  make:  sample and write the sheet (provisional labels from --llm-labels json {"<plan>|<question_id>": "correct"|"wrong"},
         if given; the human column stays empty)
  score: parse the sheet and report agreement of each judge with the human labels (and with the provisional labels),
         using whatever labels are filled in at the time

  python scripts/judge_spotcheck.py make --runs runs/gold_lme_s runs/random_lme_s runs/fullhist_lme_s_2k \
      runs/fullhist_lme_s_16k --data data/lme_s --n 20 --seed 0 --out docs/midterm/judge_spotcheck.md
  python scripts/judge_spotcheck.py score --sheet docs/midterm/judge_spotcheck.md --out runs/_tables/judge_spotcheck.json
Labels in the sheet: `correct` / `wrong` (also accepted: 1 / 0, y / n, yes / no); anything else = unlabelled.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.vschema import iter_jsonl  # noqa: E402

HDR = ["#", "bundle", "question_id", "type", "question", "gold answer", "model answer", "judge (gpt-oss-20b)",
       "judge 2 (Qwen3-32B)", "human label", "LLM label (provisional)"]


def cell(t: str, n: int) -> str:
    t = re.sub(r"\s+", " ", str(t)).replace("|", "/").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


def norm(x: str) -> bool | None:
    x = x.strip().lower().strip("*_` ")
    if x in ("correct", "1", "y", "yes", "true", "right"):
        return True
    if x in ("wrong", "0", "n", "no", "false", "incorrect"):
        return False
    return None


def make(a) -> None:
    qs = {q["question_id"]: q for q in iter_jsonl(Path(a.data) / "questions.jsonl")}
    pool = []
    for run in a.runs:
        for r in iter_jsonl(Path(run) / "answers.jsonl"):
            if qs[r["question_id"]]["split"] == "dev":
                pool.append((r["plan"], r))
    pool.sort(key=lambda x: (x[0], x[1]["question_id"]))
    pick = random.Random(a.seed).sample(pool, a.n)
    llm = json.loads(Path(a.llm_labels).read_text()) if a.llm_labels else {}
    lines = [
        "# Judge spot-check (Slice V): 20 dev answers",
        "",
        f"Sampled with `scripts/judge_spotcheck.py make` (seed {a.seed}) from all dev answers of: "
        + ", ".join(f"`{Path(r).name}`" for r in a.runs) + ".",
        "Reader Qwen3-8B (Q8_0, chain-of-note); judges with the LongMemEval judge prompts: gpt-oss-20b (official) and"
        " Qwen3-32B (second). Long texts are shortened with `…`; full texts are in each run's `answers.jsonl`.",
        "",
        "**Human label:** fill `correct` or `wrong` (is the model answer correct for the question, given the gold answer"
        " and the LongMemEval rubric?). The **LLM label** column is a provisional LLM reading, not a human label."
        " Score with `python scripts/judge_spotcheck.py score --sheet docs/midterm/judge_spotcheck.md`.",
        "",
        "| " + " | ".join(HDR) + " |",
        "|" + "---|" * len(HDR),
    ]
    for i, (plan, r) in enumerate(pick, 1):
        q = qs[r["question_id"]]
        j1 = (r.get("judges") or {}).get("j2", r.get("judge") or {})
        j2 = (r.get("judges") or {}).get("primary") or {}
        lab = llm.get(f"{plan}|{r['question_id']}", "")
        lines.append("| " + " | ".join([str(i), plan, r["question_id"], q["question_type"], cell(q["question"], 140),
                                        cell(q["answer"], 120), cell(r.get("answer") or "", 160),
                                        "correct" if j1.get("correct") else "wrong",
                                        "correct" if j2.get("correct") else "wrong", "", lab]) + " |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {a.out} with {len(pick)} rows")


def score(a) -> None:
    rows = []
    for line in Path(a.sheet).read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| #"):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) != len(HDR) or not c[0].isdigit():
            continue
        rows.append(dict(zip(HDR, c)))
    out = {"n_rows": len(rows)}
    for ref in ("human label", "LLM label (provisional)"):
        lab = [(r, norm(r[ref])) for r in rows]
        lab = [(r, y) for r, y in lab if y is not None]
        res = {"n_labelled": len(lab)}
        for j in ("judge (gpt-oss-20b)", "judge 2 (Qwen3-32B)"):
            agree = sum((norm(r[j]) is True) == y for r, y in lab)
            res[j] = {"agree": agree, "rate": agree / len(lab) if lab else None}
        res["label_correct_rate"] = sum(y for _, y in lab) / len(lab) if lab else None
        out[ref] = res
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
        md = ["| reference labels | labelled rows | agreement with gpt-oss-20b (official) | agreement with Qwen3-32B |",
              "|---|---|---|---|"]
        for ref, name in (("human label", "human (V)"), ("LLM label (provisional)", "LLM (provisional)")):
            r = out[ref]
            cells = [f"{r[j]['agree']} / {r['n_labelled']} ({100 * r[j]['rate']:.0f}%)" if r["n_labelled"] else "– (no labels yet)"
                     for j in ("judge (gpt-oss-20b)", "judge 2 (Qwen3-32B)")]
            md.append(f"| {name} | {r['n_labelled']} / {out['n_rows']} | {cells[0]} | {cells[1]} |")
        Path(a.out).with_suffix(".md").write_text("\n".join(md) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--runs", nargs="+", required=True)
    m.add_argument("--data", required=True)
    m.add_argument("--n", type=int, default=20)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--llm-labels", default=None)
    m.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--sheet", required=True)
    s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    make(a) if a.cmd == "make" else score(a)


if __name__ == "__main__":
    main()
