"""reader -> judge -> metrics over several runs (one prompt / order variant per call).

--tag '' writes the contract files answers.jsonl / metrics.json; any other tag
writes answers.<tag>.jsonl / metrics.<tag>.json (experiments never clobber the pinned run).

  python scripts/eval_runs.py --runs runs/gold_turn_lme_s runs/random_lme_s --data data/lme_s \
      --prompt configs/prompts/reader.txt --split dev --tag ''
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.judge import promote, run_judge  # noqa: E402
from mem27.eval.metrics import summary_line, write_metrics  # noqa: E402
from mem27.eval.reader import run_reader  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--prompt", default="configs/prompts/reader.txt")
    ap.add_argument("--order", default="chrono")
    ap.add_argument("--split", default="all")
    ap.add_argument("--tag", default="")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--reader-endpoint", default="runs/vllm_endpoint.txt")
    ap.add_argument("--judge-endpoint", default="runs/vllm_judge_endpoint.txt")
    ap.add_argument("--judge-prompts", default="configs/prompts/judge.txt")
    ap.add_argument("--skip-reader", action="store_true")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--date-hints", action="store_true")
    ap.add_argument("--j2-endpoint", default="runs/endpoints/judge2",
                    help="second judge (gpt-oss-20b); '' to skip")
    ap.add_argument("--official-judge", default="j2", choices=["primary", "j2"],
                    help="which judge's verdict becomes answers[].judge (pinned: j2, see mem27/eval/judge.py)")
    ap.add_argument("--cache-tag", default="",
                    help="cache file name tag (default: --tag, or 'pinned'); give concurrent tasks distinct tags")
    ap.add_argument("--reuse-answers", action="store_true",
                    help="resume: keep error-free rows of the existing answers file made with the same prompt/model")
    a = ap.parse_args(argv)
    sfx = f".{a.tag}" if a.tag else ""
    for run in a.runs:
        t0 = time.time()
        ans = f"answers{sfx}.jsonl"
        cache_tag = a.cache_tag or a.tag or "pinned"
        # one cache file per (tag, run): two processes must never share a sqlite file on BeeGFS
        # (Oct 13: a shared reader.pinned.sqlite was corrupted by concurrent tasks)
        ckey = f"{cache_tag}.{Path(run).name}"
        if not a.skip_reader:
            run_reader(run, a.data, a.reader_endpoint, a.prompt, split=a.split, order=a.order,
                       max_tokens=a.max_tokens, out_name=ans, think=a.think, workers=a.workers,
                       cache=f"runs/_cache/reader.{ckey}.sqlite", date_hints=a.date_hints,
                       reuse=a.reuse_answers)
        run_judge(run, a.data, a.judge_endpoint, a.judge_prompts, answers_name=ans, workers=a.workers,
                  cache=f"runs/_cache/judge.{ckey}.sqlite")
        if a.j2_endpoint and Path(a.j2_endpoint).exists():
            run_judge(run, a.data, a.j2_endpoint, a.judge_prompts, name="j2", answers_name=ans, workers=a.workers,
                      cache=f"runs/_cache/judge2.{Path(run).name}.{cache_tag}.sqlite")
            if a.official_judge == "j2":
                promote(run, ans, "j2")
        m = write_metrics(run, a.data, answers_name=ans, out_name=f"metrics{sfx}.json")
        for split in ("dev", "test"):
            if split in m["aggregate"]:
                print(summary_line(m, split) + f"  tag={a.tag or 'pinned'} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
