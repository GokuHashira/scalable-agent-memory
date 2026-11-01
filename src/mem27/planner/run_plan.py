"""Run plans over all questions and write runs/<plan>_<tag>/bundles.jsonl.

history_id == question_id (1:1), so each question owns a unique HistoryIndex; we
parallelize over questions with multiprocessing. Dense plans use cached embeddings.

  python -m mem27.planner.run_plan --data data/lme_s_k --emb data/lme_s_k/emb \
      --plans all --tag lme_s_k_seed0_warm --workers 12
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import OrderedDict

import numpy as np

from mem27.planner.schema import (load_notes, load_questions, Bundle, write_jsonl,
                          MAX_RESULTS, TOKEN_CAP)
from mem27 import registry
from mem27.planner.refcommon import passage_text, count_tokens
from mem27.planner import ref_retrievers  # self-registers ref:p0..p5
from mem27.planner.ref_retrievers import (HistoryIndex, RankedResult,
                                          load_query_embeddings)
from mem27.planner.planners import PLANNER_FACTORIES, PlanDecision

ALL_FIXED = registry.FIXED_PLANS
ALL_PLANNERS = ["a0", "a1", "a2"]

# globals for workers
_G = {}


def _assemble(result: RankedResult, notes) -> tuple[list, list, int, int]:
    """Top-MAX_RESULTS note_ids + token-capped passages. Returns
    (note_ids_topk, passages_in_budget, tokens, n_in_budget)."""
    note_ids = [notes[i].note_id for i in result.indices[:MAX_RESULTS]]
    passages, tokens, n_fit = [], 0, 0
    for i in result.indices[:MAX_RESULTS]:
        p = passage_text(notes[i].role, notes[i].text)
        t = count_tokens(p)
        if tokens + t > TOKEN_CAP and n_fit > 0:
            break
        passages.append(p)
        tokens += t
        n_fit += 1
    return note_ids, passages, tokens, n_fit


def _bundle_for_fixed(plan, qid, qtext, qvec, index) -> Bundle:
    t0 = time.perf_counter()
    r = registry.get(plan).retrieve(qtext, qvec, index)
    dt = (time.perf_counter() - t0) * 1000
    note_ids, passages, tokens, n_fit = _assemble(r, index.notes)
    tr = dict(r.trace)
    tr["scores_topk"] = [round(s, 5) for s in r.scores[:MAX_RESULTS]]
    tr["n_in_budget"] = n_fit
    return Bundle(question_id=qid, plan=plan, note_ids=note_ids, passages=passages,
                  scores=[round(s, 5) for s in r.scores[:MAX_RESULTS]],
                  total_tokens=tokens, latency_ms={"total": round(dt, 3), "stages": {}},
                  stale_flags=[], trace=tr)


def _bundle_for_planner(plan, qid, qtext, qvec, index, qtype, planner_obj) -> Bundle:
    t0 = time.perf_counter()
    if plan == "a1":
        dec: PlanDecision = planner_obj.decide(qtext, qvec, index, question_type=qtype)
    else:
        dec = planner_obj.decide(qtext, qvec, index)
    dt = (time.perf_counter() - t0) * 1000
    note_ids, passages, final_tokens, n_fit = _assemble(dec.final, index.notes)
    # reader-token cost = FINAL bundle only (that is what the reader pays for).
    # cumulative across rungs = retrieval work (latency), recorded in trace.
    cumulative = 0
    for cr in dec.cost_results:
        _, _, tk, _ = _assemble(cr, index.notes)
        cumulative += tk
    total_tokens = final_tokens
    tr = dict(dec.trace)
    tr["n_in_budget"] = n_fit
    tr["chosen_plan"] = dec.chosen_plan
    tr["cumulative_retrieval_tokens"] = cumulative
    return Bundle(question_id=qid, plan=plan, note_ids=note_ids, passages=passages,
                  scores=[round(s, 5) for s in dec.final.scores[:MAX_RESULTS]],
                  total_tokens=total_tokens, latency_ms={"total": round(dt, 3), "stages": {}},
                  stale_flags=[], trace=tr)


def _init_worker(emb_dir, a0_plan, a2_tau):
    # Apply the active fixed-plan impl (planner §5): "ref" is a no-op (ref_retrievers
    # self-register on import); "g"/"gi" alias the canonical ref:pK names onto G's / the
    # independent family, so planners and oracle run unchanged through the swap path.
    registry.ensure_fixed_plans()
    _G["emb_dir"] = emb_dir
    _G["qemb"] = load_query_embeddings(emb_dir)
    _G["planners"] = {
        "a0": PLANNER_FACTORIES["a0"](default_plan=a0_plan),
        "a1": PLANNER_FACTORIES["a1"](),
        "a2": PLANNER_FACTORIES["a2"](tau_margin=a2_tau),
    }


def _work(task):
    qid, qtext, qtype, notes, plans = task
    emb_dir = _G["emb_dir"]
    qvec = _G["qemb"].get(qid)
    if qvec is None:
        qvec = np.zeros(768, dtype=np.float32)
    index = ref_retrievers.load_history_index(qid, notes, emb_dir)
    out = {}
    for plan in plans:
        if plan in PLANNER_FACTORIES:
            out[plan] = _bundle_for_planner(plan, qid, qtext, qvec, index, qtype,
                                            _G["planners"][plan])
        else:
            out[plan] = _bundle_for_fixed(plan, qid, qtext, qvec, index)
    return qid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lme_s_k")
    ap.add_argument("--emb", default=None)
    ap.add_argument("--plans", default="all")
    ap.add_argument("--tag", default="lme_s_k_seed0_warm")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default=None, help="filter dev/test (default: all)")
    ap.add_argument("--a0-plan", default="ref:p2")
    ap.add_argument("--a2-tau", type=float, default=0.1)
    args = ap.parse_args()
    emb_dir = args.emb or os.path.join(args.data, "emb")

    if args.plans == "all":
        plans = ALL_FIXED + ALL_PLANNERS
    else:
        plans = args.plans.split(",")

    notes = load_notes(os.path.join(args.data, "notes.jsonl"))
    by_hist = OrderedDict()
    for n in notes:
        by_hist.setdefault(n.history_id, []).append(n)
    questions = load_questions(os.path.join(args.data, "questions.jsonl"))
    if args.split:
        questions = [q for q in questions if q.split == args.split]
    if args.limit:
        questions = questions[:args.limit]

    tasks = [(q.question_id, q.question, q.question_type,
              by_hist[q.history_id], plans) for q in questions]
    print(f"plans={plans} questions={len(tasks)} workers={args.workers}", flush=True)

    results: dict[str, list] = {p: [] for p in plans}
    t0 = time.time()
    if args.workers <= 1:
        _init_worker(emb_dir, args.a0_plan, args.a2_tau)
        for i, task in enumerate(tasks):
            _, out = _work(task)
            for p, b in out.items():
                results[p].append(b)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init_worker,
                      initargs=(emb_dir, args.a0_plan, args.a2_tau)) as pool:
            for i, (_, out) in enumerate(pool.imap_unordered(_work, tasks, chunksize=4)):
                for p, b in out.items():
                    results[p].append(b)
                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{len(tasks)}  ({time.time()-t0:.0f}s)", flush=True)
    elapsed = time.time() - t0

    env = {
        "node": platform.node(), "python": platform.python_version(),
        "cpu_count": os.cpu_count(), "workers": args.workers,
        "hardware": "4x NVIDIA RTX A6000 (embeddings); retrieval on CPU",
        "embed_model": "BAAI/bge-base-en-v1.5", "token_cap": TOKEN_CAP,
        "max_results": MAX_RESULTS, "a0_plan": args.a0_plan, "a2_tau": args.a2_tau,
        "split": args.split or "all", "n_questions": len(tasks),
        "wall_seconds": round(elapsed, 1), "plans": plans,
    }
    for p in plans:
        short = p.replace(":", "_")
        d = os.path.join(args.out_root, f"{short}_{args.tag}")
        os.makedirs(d, exist_ok=True)
        # keep deterministic ordering by question order
        order = {q.question_id: i for i, q in enumerate(questions)}
        rows = sorted(results[p], key=lambda b: order.get(b.question_id, 1e9))
        write_jsonl(os.path.join(d, "bundles.jsonl"), rows)
        json.dump(env, open(os.path.join(d, "env.json"), "w"), indent=2)
    print(f"done in {elapsed:.1f}s; wrote {len(plans)} run dirs under {args.out_root}/", flush=True)


if __name__ == "__main__":
    main()
