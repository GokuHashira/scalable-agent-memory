"""Seed mean ± sd of streaming freshness per retrieval method x variant (reads runs/ref_<method>_<variant>_<data>_stream/freshness.json).

  python scripts/synth_fresh_summary.py --data synth_10k synth_10k_s1 synth_10k_s2 --methods bm25 dense hybrid \
      --out runs/_tables/synth_freshness_methods_10k
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from scripts.reproduce_table import to_md  # noqa: E402

VARIANTS = ["plain", "vfilter", "recency", "vfilter_recency"]
GROUPS = ["ALL", "supersede:explicit", "supersede:implicit"]
METRICS = ["fresh@5", "stale_in@5", "latest_in@20"]


def ms(v: list[float]) -> str:
    return f"{st.mean(v):.3f} ± {st.stdev(v):.3f}" if len(v) > 1 else (f"{v[0]:.3f}" if v else "–")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="dataset names (one per seed)")
    ap.add_argument("--methods", nargs="+", default=["bm25", "dense", "hybrid"])
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    js, rows = {}, [["method", "variant"] + [f"{g} {m}" for g in GROUPS for m in METRICS[:2]] + ["ALL latest_in@20"]]
    for meth in a.methods:
        for v in VARIANTS:
            vals = {(g, m): [] for g in GROUPS for m in METRICS}
            for d in a.data:
                f = Path(a.runs) / f"ref_{meth}_{v}_{d}_stream" / "freshness.json"
                if not f.exists():
                    continue
                gs = json.loads(f.read_text())["groups"]
                for g in GROUPS:
                    for m in METRICS:
                        if m in gs.get(g, {}):
                            vals[(g, m)].append(gs[g][m])
            js[f"{meth}/{v}"] = {f"{g}|{m}": {"mean": st.mean(x), "sd": st.stdev(x) if len(x) > 1 else 0.0, "n_seeds": len(x),
                                               "per_seed": x} for (g, m), x in vals.items() if x}
            rows.append([meth, v] + [ms(vals[(g, m)]) for g in GROUPS for m in METRICS[:2]] + [ms(vals[("ALL", "latest_in@20")])])
    md = to_md(rows)
    Path(a.out + ".md").write_text(md)
    Path(a.out + ".json").write_text(json.dumps(js, indent=1))
    print(md)


if __name__ == "__main__":
    main()
