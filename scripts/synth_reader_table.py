"""Reader results on the synthetic workload: accuracy and stale-served per variant, with paired bootstrap CIs.

Reads `freshness.<tag>.json` (per_question rows written by `mem27.eval.freshness`) for each run and compares
every run against `--ref` on the same questions. Rows are split by update group (explicit / implicit
supersede or correct, no change) and by question type.

    python scripts/synth_reader_table.py --runs runs/ref_bm25_{plain,vfilter,recency,vfilter_recency}_synth_10k_stream \
        --ref runs/ref_bm25_plain_synth_10k_stream --tag concise --split test --out runs/_tables/synth_reader_stream_10k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mem27.eval.bootstrap import paired_bootstrap

METRICS = ("correct_str", "correct_judge", "stale_only", "mentions_stale", "fresh@5")


def load(run: Path, tag: str) -> dict[str, dict]:
    f = run / (f"freshness.{tag}.json" if tag else "freshness.json")
    return {r["question_id"]: r for r in json.loads(f.read_text())["per_question"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tag", default="concise")
    ap.add_argument("--split", default="test", help="test | dev | all")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ref = load(Path(a.ref), a.tag)
    out, md = [], []
    md.append(f"Split: {a.split}. Ref: `{Path(a.ref).name}`. Delta = run - ref, paired bootstrap 95% CI "
              f"({a.n_boot} resamples, seed 0). correct_str = string match; stale_only = answer contains a "
              f"superseded value and not the current one.\n")
    md.append("| run | group | n | metric | run | ref | delta | 95% CI |")
    md.append("|---|---|---|---|---|---|---|---|")
    for run in a.runs:
        rows = load(Path(run), a.tag)
        qids = [q for q in rows if q in ref and (a.split == "all" or rows[q].get("split") == a.split)]
        groups: dict[str, list[str]] = {"ALL": qids}
        for q in qids:
            groups.setdefault(rows[q]["group"], []).append(q)
            groups.setdefault("type:" + rows[q]["question_type"], []).append(q)
        upd = [q for q in qids if rows[q]["group"] != "no_change"]
        groups["updated"] = upd
        for g in sorted(groups, key=lambda s: (s != "ALL", s != "updated", s)):
            qs = groups[g]
            for m in METRICS:
                pairs = [(rows[q][m], ref[q][m]) for q in qs if m in rows[q] and m in ref[q]]
                if not pairs:
                    continue
                x, y = np.array(pairs, float).T
                bs = paired_bootstrap(x, y, n_boot=a.n_boot) if Path(run) != Path(a.ref) else \
                    {"n": len(x), "mean_a": float(x.mean()), "mean_b": float(y.mean()), "delta": 0.0,
                     "ci_lo": 0.0, "ci_hi": 0.0}
                rec = {"run": Path(run).name, "group": g, "metric": m, **bs}
                out.append(rec)
                if m in ("correct_str", "stale_only") and (g in ("ALL", "updated") or ":" in g):
                    md.append(f"| {rec['run']} | {g} | {bs['n']} | {m} | {bs['mean_a']:.3f} | {bs['mean_b']:.3f} | "
                              f"{bs['delta']:+.3f} | [{bs['ci_lo']:+.3f}, {bs['ci_hi']:+.3f}] |")
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
