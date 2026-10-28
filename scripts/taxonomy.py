"""Failure taxonomy (EVAL-06): rule-based primary failure mode for every wrong answer.

Modes (research_plan §5.4 + workstream V):
  hallucinated-abstention  _abs question answered instead of declined
  turn-miss                every gold session is represented in the bundle, but by other turns: none of the
                           answer (has_answer) turns was packed (under-retrieval at turn granularity)
  turn-partial             every gold session represented, some but not all answer turns packed
                           (so reader-error / over-retrieval below mean: all answer turns were in the bundle)
  false-abstention         answerable question declined although gold sessions and answer turns were retrieved
  truncation               a gold turn is in the bundle but its passage was cut by the token cap
  wrong-version            knowledge-update question with both old and new evidence retrieved
  lexical-miss             gold missing from the bundle, plan is lexical (BM25), dense ranks it in its top 20
  semantic-drift           gold missing from the bundle, plan is dense, BM25 ranks it in its top 20
  under-retrieval          gold session missing from the bundle but in the plan's own top-50 (a larger budget would reach it)
  retrieval-miss           gold session missing and not in the plan's top-50 either (v1.1; v1 used top-200, ~40% of a history)
  partial-retrieval        multi-evidence question with only some gold sessions retrieved
  over-retrieval           all gold present but < 15% of bundle tokens are from gold sessions
  reader-error             all gold present with enough signal; the reader reasoned wrongly
                           (sub-labelled temporal-arithmetic / preference / aggregation by question type)
Writes runs/_tables/taxonomy_<tag>.json and candidate examples for docs/taxonomy.md.

  python scripts/taxonomy.py --runs runs/vref_bm25_lme_s runs/vref_dense_lme_s runs/vref_hybrid_lme_s \
      --data data/lme_s --answers answers.jsonl --tag lme_s
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import write_json  # noqa: E402
from mem27.vschema import Question, iter_jsonl, read_jsonl, session_of  # noqa: E402

ABSTAIN_RE = re.compile(r"(not (?:mentioned|available|provided|specified|stated|discussed|found)|no information|"
                        r"(?:don't|do not|cannot|can't) (?:know|find|determine|tell)|doesn't (?:say|mention)|"
                        r"there is no (?:mention|record|information)|unable to (?:determine|find)|isn't mentioned)", re.I)


def load_rankings(run: Path) -> dict[str, list[str]]:
    f = run / "rankings.jsonl"
    return {r["question_id"]: r["note_ids"] for r in iter_jsonl(f)} if f.exists() else {}


def classify(q: Question, b: dict, a: dict, texts: dict[str, str], ranks: dict[str, dict[str, list[str]]],
             plan_key: str, ku: dict | None) -> tuple[str, dict]:
    ans = a.get("answer") or ""
    abst = q.question_id.endswith("_abs")
    said_abstain = bool(ABSTAIN_RE.search(ans))
    nids = b["note_ids"]
    gold_s, gold_t = set(q.gold_session_ids), set(q.gold_note_ids)
    cov = {session_of(n) for n in nids}
    info = {"answer": ans[:300], "gold": q.answer[:200], "said_abstain": said_abstain,
            "gold_sessions_in_bundle": len(gold_s & cov), "n_gold_sessions": len(gold_s)}
    if abst:
        return "hallucinated-abstention", info
    all_gold = gold_s <= cov
    info["gold_turns_in_bundle"] = len(gold_t & set(nids))
    info["n_gold_turns"] = len(gold_t)
    for n, p in zip(nids, b["passages"]):
        if n in gold_t and n in texts and len(p) < len(texts[n]) - 5:
            info["truncated_note"] = n
            return "truncation", info
    # bundles are turn-level: every gold session can be "covered" by some turn while none of the answer
    # turns (has_answer) made it in; abstaining is then correct given the bundle (v1.1, Oct 13)
    if all_gold and gold_t and not (gold_t & set(nids)):
        return "turn-miss", info
    # some but not all answer turns packed (typical for multi-session counting / summing)
    if all_gold and gold_t and not (gold_t <= set(nids)):
        return "turn-partial", info
    if all_gold and said_abstain:
        return "false-abstention", info
    if ku is not None and q.question_type == "knowledge-update":
        ids = set(nids)
        if ids & set(ku["old_note_ids"]) and ids & set(ku["new_note_ids"]):
            return "wrong-version", info
    if not all_gold:
        if len(gold_s & cov) > 0 and len(gold_s) > 1:
            return "partial-retrieval", info
        own = ranks.get(plan_key, {}).get(q.question_id, [])
        in_own = any(session_of(n) in gold_s for n in own[:50])
        bm = ranks.get("bm25", {}).get(q.question_id, [])
        de = ranks.get("dense", {}).get(q.question_id, [])
        bm_hit = any(session_of(n) in gold_s for n in bm[:20])
        de_hit = any(session_of(n) in gold_s for n in de[:20])
        if "bm25" in plan_key and de_hit:
            return "lexical-miss", info
        if "dense" in plan_key and bm_hit:
            return "semantic-drift", info
        if in_own:
            return "under-retrieval", info
        return "retrieval-miss", info
    gold_tok = sum(len(p) for n, p in zip(nids, b["passages"]) if session_of(n) in gold_s)
    tot = sum(len(p) for p in b["passages"]) or 1
    info["gold_char_share"] = round(gold_tok / tot, 3)
    if gold_tok / tot < 0.15:
        return "over-retrieval", info
    sub = {"temporal-reasoning": "reader-error:temporal", "single-session-preference": "reader-error:preference",
           "multi-session": "reader-error:aggregation"}.get(q.question_type, "reader-error:extraction")
    return sub, info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--answers", default="answers.jsonl")
    ap.add_argument("--split", default="all")
    ap.add_argument("--tag", default="lme_s")
    ap.add_argument("--n-examples", type=int, default=40)
    a = ap.parse_args(argv)
    data = Path(a.data)
    qs = {q.question_id: q for q in read_jsonl(data / "questions.jsonl", Question)}
    gold_ids = {g for q in qs.values() for g in q.gold_note_ids}
    texts = {n["note_id"]: n["text"] for n in iter_jsonl(data / "notes.jsonl") if n["note_id"] in gold_ids}
    ku = {p["question_id"]: p for p in iter_jsonl(data / "ku_pairs.jsonl")} if (data / "ku_pairs.jsonl").exists() else {}
    ranks = {}
    for run in a.runs:
        name = Path(run).name
        for k in ("bm25", "dense", "hybrid"):
            if f"_{k}_" in name and k not in ranks:
                ranks[k] = load_rankings(Path(run))
    out = {"counts": {}, "by_type": {}, "examples": []}
    rng = random.Random(0)
    pool = []
    for run in a.runs:
        run = Path(run)
        bundles = {b["question_id"]: b for b in iter_jsonl(run / "bundles.jsonl")}
        answers = list(iter_jsonl(run / a.answers))
        plan_key = next((k for k in ("hybrid", "dense", "bm25") if f"_{k}_" in run.name), run.name)
        ranks.setdefault(plan_key, load_rankings(run))
        cnt, by_t = Counter(), defaultdict(Counter)
        for ans in answers:
            q = qs[ans["question_id"]]
            if a.split != "all" and q.split != a.split:
                continue
            if (ans.get("judge") or {}).get("correct", True):
                continue
            mode, info = classify(q, bundles[q.question_id], ans, texts, ranks, plan_key, ku.get(q.question_id))
            cnt[mode] += 1
            by_t[q.question_type][mode] += 1
            pool.append({"run": run.name, "question_id": q.question_id, "type": q.question_type,
                         "question": q.question, "mode": mode, **info})
        out["counts"][run.name] = dict(cnt.most_common())
        out["by_type"][run.name] = {t: dict(c) for t, c in by_t.items()}
        print(run.name, dict(cnt.most_common()))
    by_mode = defaultdict(list)
    for p in pool:
        by_mode[p["mode"]].append(p)
    for mode, ps in sorted(by_mode.items()):
        rng.shuffle(ps)
        out["examples"].extend(ps[: max(2, a.n_examples // max(1, len(by_mode)))])
    out["labels"] = {f"{p['run']}|{p['question_id']}": p["mode"] for p in pool}  # every wrong answer's rule label
    write_json(f"runs/_tables/taxonomy_{a.tag}.json", out)


if __name__ == "__main__":
    main()
