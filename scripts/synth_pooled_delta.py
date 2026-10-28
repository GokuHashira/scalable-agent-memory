"""Pooled paired accuracy delta of a synthetic retrieval variant vs plain over several independent datasets
(different generated worlds = independent question sets), from freshness.<tag>.json per-question `correct_str`.

  python scripts/synth_pooled_delta.py --scales synth_10k synth_50k synth_100k --variants vfilter recency vfilter_recency \
      --tag concise --split test --out runs/_tables/synth_stream_reader_pooled.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.bootstrap import paired_bootstrap  # noqa: E402
from mem27.eval.common import write_json  # noqa: E402


def rows(run: Path, tag: str, split: str) -> dict[str, dict]:
    pq = json.loads((run / f"freshness.{tag}.json").read_text())["per_question"]
    return {r["question_id"]: r for r in pq if split == "all" or r.get("split") == split}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["vfilter", "recency", "vfilter_recency"])
    ap.add_argument("--tag", default="concise")
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = {}
    for v in a.variants:
        xa, xb, per = [], [], {}
        for s in a.scales:
            ref = rows(Path(a.runs) / f"ref_bm25_plain_{s}_stream", a.tag, a.split)
            var = rows(Path(a.runs) / f"ref_bm25_{v}_{s}_stream", a.tag, a.split)
            ids = sorted(ref.keys() & var.keys())
            va = [var[q]["correct_str"] for q in ids]
            vb = [ref[q]["correct_str"] for q in ids]
            per[s] = {"n": len(ids), "delta": float(np.mean(va) - np.mean(vb))}
            xa += va
            xb += vb
        bs = paired_bootstrap(np.array(xa, float), np.array(xb, float))
        out[v] = {"pooled": bs, "per_scale": per}
        print(f"{v:16s} pooled n={bs['n']} delta={100 * bs['delta']:+.1f} [{100 * bs['ci_lo']:+.1f}, {100 * bs['ci_hi']:+.1f}] "
              f"p={bs['p']:.3f} | " + " ".join(f"{s}:{100 * x['delta']:+.1f}" for s, x in per.items()))
    write_json(a.out, out)


if __name__ == "__main__":
    main()
