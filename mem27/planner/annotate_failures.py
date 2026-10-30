"""Find + categorize cascade (a2) failures for the midterm writeup.

A failure = a2 retrieves no gold turn in top-5 (note_hit@5 == 0) on test. Each is
assigned one of five categories using trace + lexical/semantic signals:
  under-retrieval, over-retrieval, lexical-miss, semantic-drift, traversal-blowup.
Picks one annotated example per category.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict

import numpy as np

from mem27.schema import load_notes, load_questions
from mem27 import metrics as M
from mem27.analysis import load_runs
from mem27.planner.refcommon import bm25_tokenize
from mem27.planner.ref_retrievers import load_query_embeddings, load_history_index

CATS = ["under-retrieval", "over-retrieval", "lexical-miss", "semantic-drift", "traversal-blowup"]


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def categorize(q, ba, runs, index, qvec, note_by_id):
    gold_notes = set(q.gold_note_ids)
    gold_sess = set(q.gold_session_ids)
    top5 = set(ba.note_ids[:5])
    top20 = set(ba.note_ids[:20])
    stop = ba.trace.get("stop_reason")
    final_plan = ba.trace.get("final_plan", ba.trace.get("chosen_plan"))
    gold_in_5 = bool(gold_notes & top5)
    gold_in_20 = bool(gold_notes & top20)
    gold_sess_in_20 = bool(gold_sess & set(M.session_of(n) for n in top20))
    p5_hit5 = M.note_hit_at_k(runs["ref:p5"][q.question_id].note_ids, q.gold_note_ids, 5)
    p5_pool = runs["ref:p5"][q.question_id].trace.get("pool_size", 0)

    qtok = set(bm25_tokenize(q.question))
    lex = max((jaccard(qtok, set(bm25_tokenize(note_by_id[g].text)))
               for g in gold_notes if g in note_by_id), default=0.0)
    idxs = [i for i, n in enumerate(index.notes) if n.note_id in top5]
    top5_cos = float(np.mean([index.emb[i] @ qvec for i in idxs])) if (index.emb is not None and idxs) else 0.0

    # priority: cascade-specific first, then retrieval-quality signatures
    if stop == "confident" and p5_hit5 and not gold_in_5:
        cat = "under-retrieval"
        why = (f"a2 stopped early at {final_plan} (confident) and missed the gold turn; the "
               f"full plan ref:p5 WOULD have put a gold turn in top-5. The cascade's "
               f"confidence signal fired too soon — escalation was warranted.")
    elif not gold_in_20 and lex < 0.02:
        cat = "lexical-miss"
        why = (f"query shares almost no surface tokens with the gold turn (jaccard={lex:.3f}) and "
               f"top-5 similarity is weak (cos={top5_cos:.3f}); the query paraphrases the memory, "
               f"BM25 ranks it near the bottom and dense is too weak to recover it.")
    elif not gold_sess_in_20 and lex >= 0.02:
        cat = "semantic-drift"
        why = (f"the gold session never reached top-20 despite real surface overlap "
               f"(jaccard={lex:.3f}): the dense/hybrid ranking drifted to topically adjacent "
               f"distractor sessions and outranked the true session.")
    elif final_plan in ("ref:p4", "ref:p5") and p5_pool and p5_pool >= 45 and gold_in_20 and not gold_in_5:
        cat = "traversal-blowup"
        why = (f"traversal expanded to a large candidate pool (≈{p5_pool}, L=50); the gold turn "
               f"made the top-20 but re-ranking that many loosely linked turns pushed it below the "
               f"top-5 budget.")
    elif gold_in_20 and not gold_in_5:
        cat = "over-retrieval"
        why = ("gold turn was retrieved but ranked 6-20: distractor turns crowded the top-5, "
               "diluting the gold evidence out of the tight budget.")
    else:
        cat = "over-retrieval"
        why = "gold session surfaced but the specific gold turn was crowded out of the budget."
    return cat, why, {"stop": stop, "final_plan": final_plan, "lex_jaccard": round(lex, 4),
                      "top5_cos": round(top5_cos, 4), "gold_in_top20": gold_in_20,
                      "gold_sess_in_top20": gold_sess_in_20, "p5_would_hit@5": int(p5_hit5),
                      "pool_size": ba.trace.get("pool_size")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lme_s_k")
    ap.add_argument("--emb", default=None)
    ap.add_argument("--out", default="runs/analysis/cascade_failures.json")
    args = ap.parse_args()
    emb_dir = args.emb or os.path.join(args.data, "emb")

    runs = load_runs(["runs/ref_p*_lme_s_k_seed0_warm", "runs/a2_lme_s_k_seed0_warm"])
    qs = load_questions(os.path.join(args.data, "questions.jsonl"))
    test = [q for q in qs if q.split == "test"]
    a2 = runs["a2"]

    # failing questions (a2 misses gold turn in top-5)
    failing = [q for q in test
               if M.note_hit_at_k(a2[q.question_id].note_ids, q.gold_note_ids, 5) == 0]
    fail_hids = set(q.history_id for q in failing)
    print(f"a2 note-miss@5 on test: {len(failing)}/{len(test)}")

    # load notes only for failing histories
    by_hist = OrderedDict()
    for n in load_notes(os.path.join(args.data, "notes.jsonl")):
        if n.history_id in fail_hids:
            by_hist.setdefault(n.history_id, []).append(n)
    qemb = load_query_embeddings(emb_dir)

    picked = {}
    allcats = []
    for q in failing:
        notes = by_hist[q.history_id]
        note_by_id = {n.note_id: n for n in notes}
        index = load_history_index(q.question_id, notes, emb_dir)
        qvec = qemb.get(q.question_id, np.zeros(768, np.float32))
        cat, why, sig = categorize(q, a2[q.question_id], runs, index, qvec, note_by_id)
        gold_txt = ""
        if q.gold_note_ids and q.gold_note_ids[0] in note_by_id:
            gold_txt = note_by_id[q.gold_note_ids[0]].text[:160]
        rec = {"category": cat, "question_id": q.question_id, "type": q.question_type,
               "question": q.question[:160], "answer": str(q.answer if q.answer is not None else "")[:80],
               "gold_turn_text": gold_txt, "why": why, "signals": sig}
        allcats.append(rec)
        if cat not in picked:
            picked[cat] = rec

    # ensure all 5 categories represented where possible
    examples = [picked[c] for c in CATS if c in picked]
    json.dump({"n_failures": len(failing), "examples": examples,
               "category_counts": {c: sum(1 for r in allcats if r["category"] == c) for c in CATS}},
              open(args.out, "w"), indent=2)
    print("category counts:", {c: sum(1 for r in allcats if r['category'] == c) for c in CATS})
    for r in examples:
        print(f"\n[{r['category']}] {r['question_id']} ({r['type']})")
        print(f"  Q: {r['question']}")
        print(f"  why: {r['why']}")
        print(f"  signals: {r['signals']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
