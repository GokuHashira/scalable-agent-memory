"""Slice V failure taxonomy for retriever-free bundles (gold / random / fullhist): rule labels for every wrong answer,
a seeded sample for hand tagging, counts per category, and agreement with hand labels.

Categories (first match wins):
  judge-disagreement  the official judge (gpt-oss-20b) says wrong and the second judge (Qwen3-32B) says correct
  wrong-version       knowledge-update question whose answer the stale judge flags as stating the superseded value
  truncation          the evidence was cut by truncation: for fullhist bundles (whole history, most recent first, cut at
                      the cap) no answer turn survives the cut, i.e. the evidence is older than the window; for any
                      bundle, an answer turn whose passage was cut by the token cap
  missing-evidence    no answer turn in the bundle for a bundle that does not truncate a history (random: the evidence
                      was not sampled); also partial evidence (some answer turns missing)
  reader-error        every answer turn is in the bundle (uncut) and both judges say wrong

  python scripts/taxonomy_slice.py --runs runs/fullhist_lme_s_2k runs/random_lme_s --data data/lme_s --split dev \
      --sample 30 --seed 0 --out runs/_tables/taxonomy_slice_dev [--manual runs/_tables/taxonomy_slice_manual_dev.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.vschema import iter_jsonl, session_of  # noqa: E402
from scripts.reproduce_table import to_md  # noqa: E402

CATS = ["judge-disagreement", "wrong-version", "truncation", "missing-evidence", "reader-error"]


def classify(q: dict, b: dict, a: dict, texts: dict[str, str]) -> tuple[str, dict]:
    js = a.get("judges") or {}
    gold_t = set(q["gold_note_ids"]) or None
    nids = b["note_ids"]
    info = {"n_gold_turns": len(q["gold_note_ids"])}
    if js.get("primary", {}).get("correct") and not a["judge"]["correct"]:
        return "judge-disagreement", info
    st = js.get("stale") or {}
    if q["question_type"] == "knowledge-update" and st.get("correct"):
        return "wrong-version", info
    if gold_t:
        present = [n for n in nids if n in gold_t]
        info["gold_turns_in_bundle"] = len(present)
        for n, p in zip(nids, b["passages"]):
            if n in gold_t and n in texts and len(p) < len(texts[n]) - 5:
                info["truncated_note"] = n
                return "truncation", info
        if not present:
            return ("truncation" if b.get("plan", "").startswith("fullhist") else "missing-evidence"), info
        if len(present) < len(gold_t):
            info["partial"] = True
            return "missing-evidence", info
        return "reader-error", info
    cov = {session_of(n) for n in nids}
    if not (set(q["gold_session_ids"]) & cov):
        return ("truncation" if b.get("plan", "").startswith("fullhist") else "missing-evidence"), info
    return "reader-error", info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manual", default=None, help='json {"<plan>|<qid>": category}')
    ap.add_argument("--answers", default="answers.jsonl", help="answers file name (e.g. a copy carrying judges.stale)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    data = Path(a.data)
    qs = {q["question_id"]: q for q in iter_jsonl(data / "questions.jsonl")}
    gold_ids = {g for q in qs.values() for g in q["gold_note_ids"]}
    texts = {n["note_id"]: n["text"] for n in iter_jsonl(data / "notes.jsonl") if n["note_id"] in gold_ids}
    out = {"split": a.split, "counts": {}, "labels": {}, "rows": {}}
    pools = {}
    for run in map(Path, a.runs):
        B = {b["question_id"]: b for b in iter_jsonl(run / "bundles.jsonl")}
        cnt, wrong = Counter(), []
        for r in iter_jsonl(run / a.answers):
            q = qs[r["question_id"]]
            if (a.split != "all" and q["split"] != a.split) or r["judge"]["correct"]:
                continue
            cat, info = classify(q, B[q["question_id"]], r, texts)
            key = f"{r['plan']}|{q['question_id']}"
            cnt[cat] += 1
            out["labels"][key] = cat
            out["rows"][key] = {"type": q["question_type"], "question": q["question"], "gold": str(q["answer"]),
                                "answer": r.get("answer") or "", "rule": cat, **info}
            wrong.append(key)
        out["counts"][run.name] = {"wrong": sum(cnt.values()), **{c: cnt.get(c, 0) for c in CATS}}
        pools[run.name] = sorted(wrong)
    rng = random.Random(a.seed)
    per = a.sample // len(a.runs)
    out["sample"] = [k for name in sorted(pools) for k in rng.sample(pools[name], min(per, len(pools[name])))]
    rows = [["category"] + [f"{r} (n wrong = {out['counts'][r]['wrong']})" for r in out["counts"]]]
    for c in CATS:
        rows.append([c] + [f"{out['counts'][r][c]} ({100 * out['counts'][r][c] / max(1, out['counts'][r]['wrong']):.0f}%)"
                           for r in out["counts"]])
    md = f"Rule labels for every wrong {a.split} answer (`scripts/taxonomy_slice.py`):\n\n" + to_md(rows)
    if a.manual and Path(a.manual).exists():
        man = json.loads(Path(a.manual).read_text())
        both = [k for k in man if k in out["labels"]]
        agree = sum(man[k] == out["labels"][k] for k in both)
        mc = Counter(man[k] for k in both)
        out["manual"] = {"n": len(both), "agree": agree, "rate": agree / len(both) if both else None,
                         "counts": {c: mc.get(c, 0) for c in CATS},
                         "disagreements": {k: {"manual": man[k], "rule": out["labels"][k]} for k in both if man[k] != out["labels"][k]}}
        mrows = [["category", f"hand-tagged sample (n = {len(both)})"]] + [[c, str(mc.get(c, 0))] for c in CATS]
        md += (f"\nHand-tagged sample ({len(both)} wrong answers, seed {a.seed}, half from each run): rule label agrees"
               f" with the hand label on {agree} / {len(both)}.\n\n" + to_md(mrows))
    Path(a.out + ".md").write_text(md)
    Path(a.out + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(md)


if __name__ == "__main__":
    main()
