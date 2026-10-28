"""E3: full-history window scaling. The most-recent-first full history packed at more caps than Slice V's
2K / 16K, to split "evidence never entered the window" from "evidence entered but got drowned"
(metrics.json: acc_given_all_gold vs acc_given_gold_missing). Reuses scripts/gold_bundles.fullhist.

  python scripts/exp_fullhist_windows.py --data data/lme_s --caps 4096 8192 24576
  -> runs/fullhist_lme_s_4k, _8k, _24k  (plans fullhist4k, fullhist8k, fullhist24k)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import NoteIndex, collect_env  # noqa: E402
from mem27.vschema import Question, read_jsonl  # noqa: E402
from scripts.gold_bundles import fullhist, write_run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lme_s")
    ap.add_argument("--caps", type=int, nargs="+", default=[4096, 8192, 24576])
    ap.add_argument("--split", default="dev")
    a = ap.parse_args(argv)
    data = Path(a.data)
    qs = [q for q in read_jsonl(data / "questions.jsonl", Question) if a.split == "all" or q.split == a.split]
    idx = NoteIndex(data / "notes.jsonl", keep_text=True, histories={q.history_id for q in qs})
    env = collect_env(seed=0)
    for cap in a.caps:
        k = f"{cap // 1024}k"
        write_run(Path("runs") / f"fullhist_{data.name}_{k}", [fullhist(q, idx, cap, f"fullhist{k}") for q in qs], env,
                  {"experiment": "E3 window scaling", "cap": cap, "order": "most recent first", "split": a.split})


if __name__ == "__main__":
    main()
