"""Per-question budget oracle (upper bound for a budget-adaptive planner; supports H1/H2).

Given the same plan packed at several token caps (runs/<plan>_b<cap> + the default 2,048 run), per question:
  parity oracle : the cheapest cap whose answer is correct if the 2,048 run is correct, else the cheapest
                  cap overall -> accuracy >= fixed-2,048 by construction; reports tokens saved.
  free oracle   : the cheapest correct cap if any, else the cheapest cap -> accuracy headroom (H2-style).
Paired bootstrap CIs vs the fixed 2,048 run.

  python scripts/budget_oracle.py --runs runs/vref_hybrid_lme_s_b256 ... runs/vref_hybrid_lme_s ... \
      --ref runs/vref_hybrid_lme_s --metrics-name metrics.json --split test
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


def cap_of(run: Path) -> int:
    b = run / "budget.json"
    return json.loads(b.read_text())["cap"] if b.exists() else 2048


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--metrics-name", default="metrics.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    runs = sorted({Path(r) for r in a.runs} | {Path(a.ref)}, key=cap_of)
    rows = {}
    for r in runs:
        m = json.loads((r / a.metrics_name).read_text())
        rows[cap_of(r)] = {q["question_id"]: q for q in m["per_question"]
                           if (a.split == "all" or q["split"] == a.split) and "correct" in q}
    caps = sorted(rows)
    ref_cap = cap_of(Path(a.ref))
    qids = sorted(set.intersection(*(set(v) for v in rows.values())))
    par_acc, par_tok, free_acc, free_tok, choice = [], [], [], [], []
    ref_acc = np.array([rows[ref_cap][q]["correct"] for q in qids])
    ref_tok = np.array([rows[ref_cap][q]["total_tokens"] for q in qids])
    for i, q in enumerate(qids):
        ok = [c for c in caps if rows[c][q]["correct"] == 1.0]
        free = ok[0] if ok else caps[0]
        par = (ok[0] if ok else caps[0]) if ref_acc[i] == 1.0 else caps[0]
        free_acc.append(rows[free][q]["correct"])
        free_tok.append(rows[free][q]["total_tokens"])
        par_acc.append(rows[par][q]["correct"])
        par_tok.append(rows[par][q]["total_tokens"])
        choice.append(par)
    par_acc, par_tok, free_acc, free_tok = map(np.array, (par_acc, par_tok, free_acc, free_tok))
    res = {
        "n": len(qids), "caps": caps, "ref_cap": ref_cap,
        "fixed": {str(c): {"acc": float(np.mean([rows[c][q]["correct"] for q in qids])),
                           "tokens": float(np.mean([rows[c][q]["total_tokens"] for q in qids]))} for c in caps},
        "parity_oracle": {"acc": float(par_acc.mean()), "tokens": float(par_tok.mean()),
                          "token_saving_vs_ref": float(1 - par_tok.mean() / ref_tok.mean()),
                          "cap_histogram": {str(c): choice.count(c) for c in caps},
                          "vs_ref": paired_bootstrap(par_acc, ref_acc)},
        "free_oracle": {"acc": float(free_acc.mean()), "tokens": float(free_tok.mean()),
                        "vs_ref": paired_bootstrap(free_acc, ref_acc)},
    }
    print(json.dumps({k: v for k, v in res.items() if k != "fixed"}, indent=1))
    print("fixed:", {c: (round(v["acc"], 3), round(v["tokens"])) for c, v in res["fixed"].items()})
    if a.out:
        write_json(a.out, res)


if __name__ == "__main__":
    main()
