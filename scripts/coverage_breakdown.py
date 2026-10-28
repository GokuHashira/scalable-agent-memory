"""Accuracy by evidence coverage at turn vs session granularity (reads metrics.json per-question rows).

Bundles are turn-level, so "all gold sessions retrieved" (acc|all-gold in the main table) can hide a missing
answer turn. Buckets per question (abstention questions and questions without turn labels excluded):
  turns_all      every has_answer turn is in the bundle
  sess_all_turn_partial   every gold session is represented, some answer turns missing (none or part)
  sess_partial   some gold sessions missing
Bundle = top-20 notes (`*@20` columns; bundles hold at most 20 notes).

  python scripts/coverage_breakdown.py --runs runs/vref_bm25_lme_s runs/vref_hybrid_lme_s --split test \
      --out runs/_tables/coverage_breakdown_test [--ref runs/gold_sess_lme_s]

--ref adds, per bucket, the reference run's accuracy on the same questions and the bucket's contribution to the
overall gap: share_b x (acc_ref_b - acc_run_b) (the contributions sum to the gap on the bucketed questions).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

import numpy as np  # noqa: E402

from mem27.eval.bootstrap import paired_bootstrap  # noqa: E402
from scripts.reproduce_table import to_md  # noqa: E402

BUCKETS = ["turns_all", "sess_all_turn_partial", "sess_partial"]


def bucket(r: dict) -> str | None:
    if r.get("abstention") or r.get("turn_recall@20") is None or r.get("sess_recall@20") is None:
        return None
    if r.get("turn_all@20") == 1.0:
        return "turns_all"
    if r.get("sess_all@20") == 1.0:
        return "sess_all_turn_partial"
    return "sess_partial"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--metric", default="correct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None)
    a = ap.parse_args(argv)
    rows = [["run"] + [f"{b} (n / acc)" for b in BUCKETS] + ["sess_all but turn recall 0: n / acc"]]
    js = {}
    ref = None
    if a.ref:
        ref = {r["question_id"]: r for r in json.loads((Path(a.ref) / "metrics.json").read_text())["per_question"]}
        grows = [["run", "bucket", "n", "share", "acc run", "acc ref (same q)", "gap contribution (pts) [95% CI]"]]
    for run in a.runs:
        m = json.loads((Path(run) / "metrics.json").read_text())
        pq = [r for r in m["per_question"] if a.split == "all" or r["split"] == a.split]
        cells, res = [], {}
        for b in BUCKETS:
            xs = [r[a.metric] for r in pq if bucket(r) == b and r.get(a.metric) is not None]
            res[b] = {"n": len(xs), "acc": sum(xs) / len(xs) if xs else None}
            cells.append(f"{len(xs)} / {100 * res[b]['acc']:.1f}" if xs else "0 / –")
        z = [r[a.metric] for r in pq if bucket(r) == "sess_all_turn_partial" and r.get("turn_recall@20") == 0.0]
        res["sess_all_turn_zero"] = {"n": len(z), "acc": sum(z) / len(z) if z else None}
        cells.append(f"{len(z)} / {100 * sum(z) / len(z):.1f}" if z else "0 / –")
        rows.append([Path(run).name] + cells)
        if ref is not None:
            tot = sum(1 for r in pq if bucket(r) is not None)
            gap = 0.0
            for b in BUCKETS:
                qq = [r for r in pq if bucket(r) == b and r.get(a.metric) is not None and r["question_id"] in ref]
                if not qq:
                    continue
                ar = sum(r[a.metric] for r in qq) / len(qq)
                rr = sum(ref[r["question_id"]][a.metric] for r in qq) / len(qq)
                c = len(qq) / tot * (rr - ar)
                gap += c
                # CI: paired bootstrap over all bucketed questions of the bucket-masked deltas (ref - run)
                allq = [r for r in pq if bucket(r) is not None and r.get(a.metric) is not None and r["question_id"] in ref]
                xa = np.array([ref[r["question_id"]][a.metric] if bucket(r) == b else 0.0 for r in allq])
                xb = np.array([r[a.metric] if bucket(r) == b else 0.0 for r in allq])
                bs = paired_bootstrap(xa, xb)
                res[b].update({"acc_ref_same_q": rr, "gap_contribution": c, "share": len(qq) / tot,
                               "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"]})
                grows.append([Path(run).name, b, str(len(qq)), f"{100 * len(qq) / tot:.1f}%", f"{100 * ar:.1f}",
                              f"{100 * rr:.1f}", f"{100 * c:+.1f} [{100 * bs['ci_lo']:+.1f}, {100 * bs['ci_hi']:+.1f}]"])
            allq = [r for r in pq if bucket(r) is not None and r.get(a.metric) is not None and r["question_id"] in ref]
            bs = paired_bootstrap(np.array([ref[r["question_id"]][a.metric] for r in allq]), np.array([r[a.metric] for r in allq]))
            res["gap_total_bucketed"] = {"gap": gap, "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"], "n": len(allq)}
            grows.append([Path(run).name, "total", str(tot), "100%", "", "",
                          f"{100 * gap:+.1f} [{100 * bs['ci_lo']:+.1f}, {100 * bs['ci_hi']:+.1f}]"])
        js[Path(run).name] = res
    md = to_md(rows)
    if ref is not None:
        md += f"\nGap to {Path(a.ref).name} by bucket (same questions):\n\n" + to_md(grows)
    Path(a.out + ".md").write_text(md)
    Path(a.out + ".json").write_text(json.dumps(js, indent=1))
    print(md)


if __name__ == "__main__":
    main()
