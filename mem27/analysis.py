"""Compute the planner.md section-4 metrics tables from run bundles.

Main table per plan: gold-session recall@5/@20, gold-turn recall@20, mean tokens,
p50/p95 latency. Per-type table. A2-vs-best-fixed (recall at tokens, stop_reason).
"""
from __future__ import annotations

import glob
import os
from collections import defaultdict, Counter

from mem27.schema import load_questions, load_bundles, QUESTION_TYPES
from mem27 import metrics as M
from mem27 import registry


def load_runs(run_glob_or_dirs):
    out = {}
    for pat in run_glob_or_dirs:
        for d in (sorted(glob.glob(pat)) or [pat]):
            bp = os.path.join(d, "bundles.jsonl")
            if os.path.exists(bp):
                bs = load_bundles(bp)
                if bs:
                    out[bs[0].plan] = {b.question_id: b for b in bs}
    return out


def plan_metrics(bundles_by_qid, questions, latency_lookup=None):
    """Return dict of aggregate metrics for one plan over the given questions."""
    s5, s20, nr20, toks, lat, nbud = [], [], [], [], [], []
    for q in questions:
        b = bundles_by_qid.get(q.question_id)
        if not b:
            continue
        s5.append(M.session_recall_at_k(b.note_ids, q.gold_session_ids, 5))
        s20.append(M.session_recall_at_k(b.note_ids, q.gold_session_ids, 20))
        nr20.append(M.note_recall_at_k(b.note_ids, q.gold_note_ids, 20))
        toks.append(b.total_tokens)
        nbud.append(b.trace.get("n_in_budget", len(b.passages)))
        lt = (latency_lookup or {}).get(q.question_id, b.latency_ms.get("total", 0.0))
        lat.append(lt)
    return {
        "sess_recall@5": M.nanmean(s5),
        "sess_recall@20": M.nanmean(s20),
        "note_recall@20": M.nanmean(nr20),
        "sess_hit@5": M.nanmean([1.0 if x and x > 0 else 0.0 for x in s5]),
        "sess_hit@20": M.nanmean([1.0 if x and x > 0 else 0.0 for x in s20]),
        "mean_tokens": sum(toks) / len(toks) if toks else 0.0,
        "mean_notes_in_budget": sum(nbud) / len(nbud) if nbud else 0.0,
        "p50_latency_ms": M.percentile(lat, 50),
        "p95_latency_ms": M.percentile(lat, 95),
        "n": len(toks),
    }


def main_table(runs, questions, latency_by_plan=None, plan_order=None):
    latency_by_plan = latency_by_plan or {}
    plan_order = plan_order or (registry.FIXED_PLANS + ["a0", "a1", "a2"])
    rows = {}
    for p in plan_order:
        if p in runs:
            rows[p] = plan_metrics(runs[p], questions, latency_by_plan.get(p))
    return rows


def per_type_recall(runs, questions, plan_order=None):
    plan_order = plan_order or registry.FIXED_PLANS
    by_type = defaultdict(list)
    for q in questions:
        by_type[q.question_type].append(q)
    table = {}
    for t in QUESTION_TYPES:
        qs = by_type.get(t, [])
        if not qs:
            continue
        row = {}
        for p in plan_order:
            if p in runs:
                vals = [M.session_recall_at_k(runs[p][q.question_id].note_ids,
                                              q.gold_session_ids, 20)
                        for q in qs if q.question_id in runs[p]]
                row[p] = M.nanmean(vals)
        row["n"] = len(qs)
        row["best_plan"] = max([p for p in plan_order if p in row and p != "n"],
                               key=lambda p: row[p])
        table[t] = row
    return table


def a2_analysis(runs, questions, best_fixed_plan, k=20, success_on="session"):
    """stop_reason histogram + recall-at-tokens vs best fixed."""
    a2 = runs.get("a2", {})
    bf = runs.get(best_fixed_plan, {})
    stop_hist = Counter()
    rungs_hist = Counter()
    a2_succ, bf_succ, a2_tok, bf_tok = [], [], [], []
    final_plan_hist = Counter()
    for q in questions:
        ba = a2.get(q.question_id)
        bb = bf.get(q.question_id)
        if not ba or not bb:
            continue
        stop_hist[ba.trace.get("stop_reason", "?")] += 1
        rungs_hist[ba.trace.get("rungs_run", "?")] += 1
        final_plan_hist[ba.trace.get("final_plan", ba.trace.get("chosen_plan", "?"))] += 1
        a2_succ.append(M.success(ba.note_ids, q, success_on, k))
        bf_succ.append(M.success(bb.note_ids, q, success_on, k))
        a2_tok.append(ba.total_tokens)
        bf_tok.append(bb.total_tokens)
    n = len(a2_succ) or 1
    return {
        "stop_reason_hist": dict(stop_hist),
        "rungs_run_hist": dict(rungs_hist),
        "final_plan_hist": dict(final_plan_hist),
        "a2_success": sum(a2_succ) / n,
        "best_fixed_success": sum(bf_succ) / n,
        "a2_mean_tokens": sum(a2_tok) / n,
        "best_fixed_mean_tokens": sum(bf_tok) / n,
        "best_fixed_plan": best_fixed_plan,
        "n": len(a2_succ),
    }
