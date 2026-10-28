"""Reader accuracy vs ranking freshness on the synthetic stream: paired comparison of retrieval variants.

For each variant run (same questions, same reader prompt), against a reference variant:
  - accuracy (string match on the invented value, `correct_str`) and stale-only rate per group
    (supersede/correct x explicit/implicit, no_change) and per question type,
  - paired bootstrap 95% CI of the accuracy delta (mem27.eval.bootstrap.paired_bootstrap) and an exact
    sign test on the discordant questions,
  - what the lost questions look like: gold note in both bundles?  answer = stale value / abstention / other.
Reads only freshness.<tag>.json, answers.<tag>.jsonl, bundles.jsonl and the questions file.

  python scripts/synth_reader_compare.py --runs runs/ref_bm25_{plain,vfilter,recency,vfilter_recency}_synth_10k_stream \
      --data data/synth_10k --questions questions_stream.jsonl --tag concise --split test \
      --out runs/_tables/synth_stream_reader_compare.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.bootstrap import paired_bootstrap  # noqa: E402
from mem27.eval.common import write_json  # noqa: E402
from mem27.vschema import iter_jsonl  # noqa: E402

ABSTAIN_RE = re.compile(r"not (?:available|mentioned|provided|specified|stated)|no information|does not (?:specify|mention)|"
                        r"cannot (?:be )?determine", re.I)


def sign_test(w: int, l: int) -> float:
    n, k = w + l, min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def load(run: Path, tag: str):
    sfx = f".{tag}" if tag else ""
    fq = {r["question_id"]: r for r in json.loads((run / f"freshness{sfx}.json").read_text())["per_question"]}
    ans = {r["question_id"]: r for r in iter_jsonl(run / f"answers{sfx}.jsonl")}
    bun = {r["question_id"]: r for r in iter_jsonl(run / "bundles.jsonl")}
    return fq, ans, bun


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="first run is the reference")
    ap.add_argument("--data", required=True)
    ap.add_argument("--questions", default="questions_stream.jsonl")
    ap.add_argument("--tag", default="concise")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--notes", default="notes_final.jsonl", help="note texts, to compare the entity evidence of two bundles")
    a = ap.parse_args(argv)
    texts = {n["note_id"]: n["text"] for n in iter_jsonl(Path(a.data) / a.notes)}
    qs = {q["question_id"]: q for q in iter_jsonl(Path(a.data) / a.questions)
          if a.split in ("all", None) or q.get("split") == a.split}
    runs = [Path(r) for r in a.runs]
    loaded = {r.name: load(r, a.tag) for r in runs}
    ref = runs[0].name
    fr, ar, br = loaded[ref]
    ids = sorted(q for q in qs if all(q in loaded[r.name][0] for r in runs))
    groups = defaultdict(list)
    for q in ids:
        groups["ALL"].append(q)
        groups["group:" + fr[q]["group"]].append(q)
        groups["type:" + qs[q]["question_type"]].append(q)
    out = {"reference": ref, "split": a.split, "n": len(ids), "tag": a.tag, "variants": {}}
    for r in runs:
        f, an, bu = loaded[r.name]
        v = {"groups": {}}
        for g, qq in sorted(groups.items()):
            acc = np.array([f[q]["correct_str"] for q in qq], float)
            racc = np.array([fr[q]["correct_str"] for q in qq], float)
            row = {"n": len(qq), "acc": float(acc.mean()),
                   "stale_only": float(np.mean([f[q].get("stale_only") or 0 for q in qq])),
                   "fresh@5": (float(np.mean(fv)) if (fv := [f[q]["fresh@5"] for q in qq if "fresh@5" in f[q]]) else None)}
            if r.name != ref:
                bs = paired_bootstrap(acc, racc)
                w = int(((racc == 1) & (acc == 0)).sum())
                l = int(((racc == 0) & (acc == 1)).sum())
                row.update({"delta": bs["delta"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                            "ref_only_correct": w, "this_only_correct": l, "sign_p": sign_test(w, l)})
            v["groups"][g] = row
        if r.name != ref:
            lost = [q for q in ids if fr[q]["correct_str"] == 1 and f[q]["correct_str"] == 0]
            kinds = Counter()
            same_ev, n_diff = 0, []
            for q in lost:
                # is the evidence about the asked entity identical in both bundles (same notes, other ranks)?
                ent = (qs[q].get("fact") or {}).get("entity") or ""
                ev_a = {n for n in br[q]["note_ids"] if ent and ent in texts.get(n, "")}
                ev_b = {n for n in bu[q]["note_ids"] if ent and ent in texts.get(n, "")}
                same_ev += ev_a == ev_b
                n_diff.append(len(set(bu[q]["note_ids"]) - set(br[q]["note_ids"])))
                gold = set(qs[q]["gold_note_ids"])
                gin = "gold_in_bundle" if gold & set(bu[q]["note_ids"]) else "gold_missing"
                txt = (an[q].get("answer") or "").lower()
                stale = [s.lower() for s in (qs[q].get("fact") or {}).get("stale_values", [])]
                kind = ("stale_value" if any(s and s in txt for s in stale)
                        else "abstention" if ABSTAIN_RE.search(txt) else "other")
                kinds[f"{gin}/{kind}"] += 1
            v["lost_vs_ref"] = {"n": len(lost), "kinds": dict(kinds.most_common()),
                                "same_entity_evidence": same_ev,
                                "mean_notes_not_in_ref_bundle": float(np.mean(n_diff)) if n_diff else None}
        out["variants"][r.name] = v
    write_json(a.out, out)
    for r in runs:
        g = out["variants"][r.name]["groups"]
        line = " ".join(f"{k.split(':')[-1]}={g[k]['acc']:.3f}" for k in ("ALL", "group:supersede:explicit", "group:supersede:implicit", "group:no_change") if k in g)
        extra = ""
        if r.name != ref:
            x = g["ALL"]
            extra = f" | delta={x['delta']:+.3f} [{x['ci_lo']:+.3f},{x['ci_hi']:+.3f}] sign p={x['sign_p']:.3f} lost={out['variants'][r.name]['lost_vs_ref']}"
        print(f"{r.name:45s} {line}{extra}")


if __name__ == "__main__":
    main()
