"""Paired bootstrap CIs between runs (EVAL-04).

For each run A vs a reference run B on the same questions: delta_i = A_i - B_i,
10K resamples of questions with replacement (seed 0), percentile 95% CI, a
two-sided bootstrap p-value, and a verdict with the pre-registered +-2 pt margin:
  equivalent   95% CI inside [-margin, +margin]
  better/worse 95% CI excludes 0
  inconclusive otherwise
Power note: s_d = SD of the paired deltas; MDE(80%, alpha .05) = 2.80 s_d / sqrt(n);
n needed to show +-margin equivalence at true delta 0 = (1.96 s_d / margin)^2.

  python -m mem27.eval.bootstrap --runs runs/p0_lme_s runs/p1_lme_s ... --ref runs/p2_lme_s \
      --metrics correct sess_recall@20 turn_recall@20 total_tokens --out runs/_tables/bootstrap_vs_p2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_METRICS = ["correct", "sess_recall@5", "sess_recall@20", "turn_recall@20", "total_tokens", "latency_total"]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0,
                     margin: float = 0.02, alpha: float = 0.05) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    n = len(d)
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for s in range(0, n_boot, 1000):  # chunked to bound memory at large n
        e = min(n_boot, s + 1000)
        means[s:e] = d[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min((means <= 0).mean(), (means >= 0).mean()))
    if lo >= -margin and hi <= margin:
        verdict = "equivalent"
    elif lo > 0:
        verdict = "better"
    elif hi < 0:
        verdict = "worse"
    else:
        verdict = "inconclusive"
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n, "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)), "delta": float(d.mean()),
        "ci_lo": float(lo), "ci_hi": float(hi), "p": float(p), "verdict": verdict,
        "discordant": int((d != 0).sum()), "sd_delta": sd,
        "mde80": 2.80 * sd / math.sqrt(n) if n > 1 else None,
        "n_for_equiv": math.ceil((1.96 * sd / margin) ** 2) if margin else None,
    }


def single_bootstrap(x: np.ndarray, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI of one run's mean (e.g. accuracy): resample questions with replacement."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for s in range(0, n_boot, 1000):
        e = min(n_boot, s + 1000)
        means[s:e] = x[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n": n, "mean": float(x.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": n_boot, "seed": seed}


def single_table(runs: list[str], metric: str = "correct", split: str = "dev", n_boot: int = 10_000, seed: int = 0,
                 metrics_name: str = "metrics.json") -> list[dict]:
    """Per run: mean of `metric` over the split's questions with a single-run percentile-bootstrap 95% CI."""
    out = []
    for run in runs:
        rows = load_rows(run, metrics_name)
        x = [r[metric] for r in rows.values() if (split == "all" or r["split"] == split) and r.get(metric) is not None]
        out.append({"run": Path(run).name, "metric": metric, "split": split, **single_bootstrap(np.array(x), n_boot, seed)})
    return out


def load_rows(run: str | Path, metrics_name: str = "metrics.json") -> dict[str, dict]:
    m = json.loads((Path(run) / metrics_name).read_text())
    return {r["question_id"]: r for r in m["per_question"]}


def compare(runs: list[str], ref: str, metrics: list[str], split: str = "test", n_boot: int = 10_000,
            seed: int = 0, margin: float = 0.02, by_type: bool = True, metrics_name: str = "metrics.json") -> list[dict]:
    ref_rows = load_rows(ref, metrics_name)
    out = []
    for run in runs:
        if Path(run).resolve() == Path(ref).resolve():
            continue
        rows = load_rows(run, metrics_name)
        qids = sorted(q for q in rows.keys() & ref_rows.keys() if split == "all" or rows[q]["split"] == split)
        groups = {"ALL": qids}
        if by_type:
            for t in sorted({rows[q]["question_type"] for q in qids}):
                groups[t] = [q for q in qids if rows[q]["question_type"] == t]
        for metric in metrics:
            for g, gq in groups.items():
                pairs = [(rows[q].get(metric), ref_rows[q].get(metric)) for q in gq]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                if not pairs:
                    continue
                a, b = np.array([p[0] for p in pairs], float), np.array([p[1] for p in pairs], float)
                # margin is in metric units: +-2 pts for rates; for tokens/latency report CI only
                mg = margin if metric.startswith(("correct", "sess_", "turn_", "ctx_", "ndcg", "mrr")) else 0.0
                res = paired_bootstrap(a, b, n_boot, seed, mg)
                if not mg:
                    res["verdict"] = {"better": "higher", "worse": "lower"}.get(res["verdict"], res["verdict"])
                    if res["verdict"] == "equivalent":
                        res["verdict"] = "inconclusive"
                out.append({"run": Path(run).name, "ref": Path(ref).name, "metric": metric, "group": g,
                            "split": split, **res})
    return out


def oracle_run(runs: list[str], metric: str = "correct", cost: str = "total_tokens",
               metrics_name: str = "metrics.json") -> dict[str, dict]:
    """Per-question oracle over runs: cheapest run whose `metric` is 1 (else cheapest overall).
    Helper for K's PLAN-04 oracle-gap analysis; returns rows usable by paired_bootstrap."""
    all_rows = {Path(r).name: load_rows(r, metrics_name) for r in runs}
    qids = set.intersection(*(set(v) for v in all_rows.values()))
    out = {}
    for q in sorted(qids):
        cands = [(rows[q].get(cost) or 0, name, rows[q]) for name, rows in all_rows.items()]
        good = [c for c in cands if c[2].get(metric) == 1.0]
        pick = min(good or cands, key=lambda c: (c[0], c[1]))
        out[q] = {**pick[2], "oracle_choice": pick[1]}
    return out


def to_markdown(rows: list[dict], metric_filter: str | None = None) -> str:
    hdr = "| run | vs | metric | group | n | A | B | Δ | 95% CI | p | verdict | MDE80 |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    lines = []
    for r in rows:
        if metric_filter and r["metric"] != metric_filter:
            continue
        pct = r["metric"] not in ("total_tokens", "latency_total")
        f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.1f}")
        lines.append(f"| {r['run']} | {r['ref']} | {r['metric']} | {r['group']} | {r['n']} | {f(r['mean_a'])} | "
                     f"{f(r['mean_b'])} | {f(r['delta'])} | [{f(r['ci_lo'])}, {f(r['ci_hi'])}] | {r['p']:.3f} | "
                     f"{r['verdict']} | {f(r['mde80']) if r.get('mde80') is not None else '-'} |")
    return hdr + "\n".join(lines) + "\n"


def write_outputs(rows: list[dict], out_prefix: str | Path) -> None:
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(f"{out_prefix}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    Path(f"{out_prefix}.md").write_text(to_markdown(rows))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--no-by-type", action="store_true")
    ap.add_argument("--metrics-name", default="metrics.json")
    ap.add_argument("--out", required=True, help="output prefix (writes .csv and .md)")
    a = ap.parse_args(argv)
    rows = compare(a.runs, a.ref, a.metrics, a.split, a.n_boot, a.seed, a.margin, not a.no_by_type, a.metrics_name)
    write_outputs(rows, a.out)
    print(to_markdown([r for r in rows if r["group"] == "ALL"]))


if __name__ == "__main__":
    main()
