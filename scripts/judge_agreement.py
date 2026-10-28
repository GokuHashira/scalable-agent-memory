"""Judge agreement (robustness): Cohen's kappa and confusion between judges over answer files,
plus a seeded sample of disagreements for manual adjudication.

  python scripts/judge_agreement.py --files runs/*_lme_s/answers.dev_*.jsonl --data data/lme_s \
      --a primary --b j2 --sample 40 --out runs/_tables/judge_agreement
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import write_json  # noqa: E402
from mem27.vschema import iter_jsonl  # noqa: E402


def kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    if not n:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--a", default="primary")
    ap.add_argument("--b", default="j2")
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--out", default="runs/_tables/judge_agreement")
    a = ap.parse_args(argv)
    qs = {q["question_id"]: q for q in iter_jsonl(Path(a.data) / "questions.jsonl")}
    files = sorted({f for pat in a.files for f in glob.glob(pat)})
    pairs, by_type, dis = [], {}, []
    for f in files:
        for r in iter_jsonl(f):
            js = r.get("judges") or {}
            if a.a not in js or a.b not in js:
                continue
            x, y = bool(js[a.a]["correct"]), bool(js[a.b]["correct"])
            q = qs[r["question_id"]]
            t = "abstention" if q["question_id"].endswith("_abs") else q["question_type"]
            pairs.append((x, y))
            by_type.setdefault(t, []).append((x, y))
            if x != y:
                dis.append({"file": f, "question_id": q["question_id"], "type": t, "question": q["question"],
                            "gold": q["answer"][:300], "answer": (r.get("answer") or "")[:600],
                            a.a: x, a.b: y, "tiebreak": (js.get("tiebreak") or {}).get("correct")})
    conf = Counter(f"{a.a}={'Y' if x else 'N'}/{a.b}={'Y' if y else 'N'}" for x, y in pairs)
    res = {"n": len(pairs), "agreement": sum(x == y for x, y in pairs) / max(1, len(pairs)), "kappa": kappa(pairs),
           "confusion": dict(conf), "acc_a": sum(x for x, _ in pairs) / max(1, len(pairs)),
           "acc_b": sum(y for _, y in pairs) / max(1, len(pairs)),
           "by_type": {t: {"n": len(v), "agreement": sum(x == y for x, y in v) / len(v), "kappa": kappa(v)}
                       for t, v in sorted(by_type.items())}}
    rng = random.Random(0)
    rng.shuffle(dis)
    res["disagreement_sample"] = dis[: a.sample]
    write_json(f"{a.out}.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "disagreement_sample"}, indent=1))


if __name__ == "__main__":
    main()
