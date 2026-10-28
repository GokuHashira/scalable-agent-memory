"""Tables for the Oct 14 experiments (E1 dilution, E2 version resolution, E3 window scaling).

Writes runs/_tables/exp_oct14.json and runs/_tables/exp_oct14.md. Accuracy uses answers[].judge
(gpt-oss-20b, the official judge). 95% CIs: percentile bootstrap, 10k resamples, seed 0; deltas paired.

  python scripts/exp_summarize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.bootstrap import paired_bootstrap  # noqa: E402
from mem27.vschema import iter_jsonl  # noqa: E402

R = Path("runs")
E1 = ["base", "hard_4k_last", "hard_4k_first", "hard_16k_last", "hard_16k_first", "easy_4k_last", "easy_16k_last"]
E3 = ["2k", "4k", "8k", "16k", "24k"]


def ci(x) -> tuple[float, float]:
    a = np.asarray(x, float)
    if not len(a):
        return (float("nan"), float("nan"))
    m = a[np.random.default_rng(0).integers(0, len(a), size=(10_000, len(a)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def pct(x) -> str:
    return "–" if x is None or x != x else f"{100 * x:.1f}"


def fmt(v, c) -> str:
    return f"{pct(v)} [{pct(c[0])}, {pct(c[1])}]"


# 3 questions whose 16K-distractor bundles exceed the reader's native 32K window; dropped from every E1 condition
E1_EXCLUDE = {"gpt4_7fce9456", "81507db6", "gpt4_f420262d"}
E1_METRICS = "metrics.gpu.json"  # E1 is read on GPU (vLLM bf16) for all 7 conditions; CPU pass was incomplete


def rows_of(run: str, name: str = "metrics.json") -> dict[str, dict]:
    m = json.loads((R / run / name).read_text())
    return {r["question_id"]: r for r in m["per_question"] if "correct" in r}


def e1() -> tuple[dict, str]:
    res, base = {}, None
    md = ["## E1. Context dilution: the same gold evidence + D distractor tokens",
          "Reader Qwen3-8B bf16 (vLLM, GPU), judge gpt-oss-20b (llama.cpp MXFP4, CPU). Answerable = 100 dev minus 6 abstention minus the 3 "
          "over-long questions (n = 91); unanswerable = all 30 abstention questions (6 dev + 24 test; diagnostic only).", "",
          "| condition | tokens (mean) | answerable acc [95% CI] | Δ vs evidence only [95% CI] | unanswerable: correctly declined |",
          "|---|---|---|---|---|"]
    for c in E1:
        run = f"xdil_{c}_lme_s"
        if not (R / run / E1_METRICS).exists():
            continue
        rows = {q: r for q, r in rows_of(run, E1_METRICS).items() if q not in E1_EXCLUDE}
        ans = {q: r for q, r in rows.items() if r["split"] == "dev" and not r["abstention"]}
        ab = [r["correct"] for r in rows.values() if r["abstention"]]
        if c == "base":
            base = ans
        d = {"n_answerable": len(ans), "acc": float(np.mean([r["correct"] for r in ans.values()])),
             "acc_ci": ci([r["correct"] for r in ans.values()]), "abst_n": len(ab),
             "abst_acc": float(np.mean(ab)) if ab else None, "abst_ci": ci(ab) if ab else None,
             "tokens_mean": float(np.mean([r["total_tokens"] for r in rows.values()]))}
        if base is not None and c != "base":
            ids = sorted(ans.keys() & base.keys())
            d["delta"] = paired_bootstrap(np.array([ans[i]["correct"] for i in ids]), np.array([base[i]["correct"] for i in ids]))
            b_ab = {q: r for q, r in rows_of("xdil_base_lme_s", E1_METRICS).items() if r["abstention"] and q not in E1_EXCLUDE}
            ids2 = sorted(set(b_ab) & {q for q, r in rows.items() if r["abstention"]})
            d["abst_delta"] = paired_bootstrap(np.array([rows[i]["correct"] for i in ids2]), np.array([b_ab[i]["correct"] for i in ids2]))
        res[c] = d
        delta = "ref" if c == "base" else (f"{100 * d['delta']['delta']:+.1f} [{100 * d['delta']['ci_lo']:+.1f}, "
                                           f"{100 * d['delta']['ci_hi']:+.1f}]" if "delta" in d else "–")
        md.append(f"| {c} | {d['tokens_mean']:,.0f} | {fmt(d['acc'], d['acc_ci'])} | {delta} | "
                  f"{sum(ab):.0f} / {len(ab)} ({pct(d['abst_acc'])}) |")
    # backend agreement: llama.cpp Q8_0 (CPU, metrics.json) vs vLLM bf16 (GPU) on conditions the CPU pass completed
    agree = []
    md += ["", "Backend check (same prompts): CPU llama.cpp Q8_0 vs GPU vLLM bf16 reader", "",
           "| condition | n | same verdict | CPU acc | GPU acc | Δ GPU−CPU [95% CI] |", "|---|---|---|---|---|---|"]
    for c in E1:
        run = f"xdil_{c}_lme_s"
        if not ((R / run / "metrics.json").exists() and (R / run / E1_METRICS).exists()):
            continue
        cpu, gpu = rows_of(run), rows_of(run, E1_METRICS)
        errs = sum(1 for x in iter_jsonl(R / run / "answers.jsonl") if (x.get("reader") or {}).get("error"))
        if errs:
            continue
        ids = sorted((cpu.keys() & gpu.keys()) - E1_EXCLUDE)
        x, y = np.array([gpu[i]["correct"] for i in ids]), np.array([cpu[i]["correct"] for i in ids])
        pb = paired_bootstrap(x, y)
        agree.append({"cond": c, "n": len(ids), "agreement": float((x == y).mean()), "cpu": float(y.mean()), "gpu": float(x.mean()), "delta": pb})
        md.append(f"| {c} | {len(ids)} | {100 * (x == y).mean():.1f}% | {pct(y.mean())} | {pct(x.mean())} | "
                  f"{100 * pb['delta']:+.1f} [{100 * pb['ci_lo']:+.1f}, {100 * pb['ci_hi']:+.1f}] |")
    res["_backend_agreement"] = agree
    return res, "\n".join(md) + "\n"


def e3() -> tuple[dict, str]:
    res = {}
    md = ["## E3. Full-history window scaling (most recent first; dev n=100)", "",
          "| window | accuracy [95% CI] | all gold sessions in window | acc when all gold in window | acc when some gold missing |",
          "|---|---|---|---|---|"]
    for k in E3:
        run = f"fullhist_lme_s_{k}"
        if not (R / run / "metrics.json").exists():
            continue
        m = json.loads((R / run / "metrics.json").read_text())
        rows = [r for r in m["per_question"] if r["split"] == "dev" and "correct" in r]
        cov = [r for r in rows if r.get("sess_all_bundle") is not None]
        inw = [r["correct"] for r in cov if r["sess_all_bundle"] == 1.0]
        outw = [r["correct"] for r in cov if r["sess_all_bundle"] == 0.0]
        acc = [r["correct"] for r in rows]
        res[k] = {"acc": float(np.mean(acc)), "acc_ci": ci(acc), "frac_all_gold_in_window": len(inw) / max(1, len(cov)),
                  "acc_in": float(np.mean(inw)) if inw else None, "n_in": len(inw), "acc_ci_in": ci(inw),
                  "acc_out": float(np.mean(outw)) if outw else None, "n_out": len(outw)}
        d = res[k]
        md.append(f"| {k} | {fmt(d['acc'], d['acc_ci'])} | {pct(d['frac_all_gold_in_window'])}% | "
                  f"{fmt(d['acc_in'], d['acc_ci_in'])} (n={d['n_in']}) | {pct(d['acc_out'])} (n={d['n_out']}) |")
    return res, "\n".join(md) + "\n"


def e2() -> tuple[dict, str]:
    res, md = {}, []
    for ds in ("synth_10k", "synth_10k_high"):
        f = R / f"xver_{ds}" / "results.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        res[ds] = r
        md += [f"## E2. Version resolution on `{ds}` (n={r['n_questions']} questions whose fact changed; exact string scoring)", "",
               "| condition | correct (current value) [95% CI] | stale value served | abstained |", "|---|---|---|---|"]
        for c in ("latest_only", "chrono_dates", "shuffled_dates", "shuffled_nodates", "stale_only"):
            a = r["conditions"][c]["ALL"]
            md.append(f"| {c} | {fmt(a['correct'], a['correct_ci'])} | {pct(a['stale_served'])} | {pct(a['abstained'])} |")
        md += ["", "By how the latest change happened (chrono_dates / shuffled_nodates / stale_only):", "",
               "| change | n | correct chrono | correct shuffled, no dates | stale served, no dates | stale served when update missing |",
               "|---|---|---|---|---|---|"]
        for g in sorted(r["conditions"]["chrono_dates"]):
            if g == "ALL":
                continue
            a, b, s = (r["conditions"][c].get(g, {}) for c in ("chrono_dates", "shuffled_nodates", "stale_only"))
            md.append(f"| {g} | {a.get('n')} | {pct(a.get('correct'))} | {pct(b.get('correct'))} | "
                      f"{pct(b.get('stale_served'))} | {pct(s.get('stale_served'))} |")
        md += ["", "Paired deltas vs chrono_dates: " + "; ".join(
            f"{k}: {100 * v['delta']:+.1f} [{100 * v['ci_lo']:+.1f}, {100 * v['ci_hi']:+.1f}]"
            for k, v in r["paired"].items()), ""]
    return res, "\n".join(md) + "\n"


def main():
    out, md = {}, ["# Oct 14 experiments (reader Qwen3-8B Q8_0 chain-of-note; judge gpt-oss-20b)", ""]
    for name, fn in (("E1", e1), ("E3", e3), ("E2", e2)):
        r, m = fn()
        out[name] = r
        md.append(m)
    (R / "_tables").mkdir(exist_ok=True)
    (R / "_tables" / "exp_oct14.json").write_text(json.dumps(out, indent=1, default=str))
    (R / "_tables" / "exp_oct14.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
