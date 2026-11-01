"""Run the All-Mem (from-paper reimpl) QA baseline on the LongMemEval dev subset.

For each dev question we build a per-history topology-structured memory bank from that
history's turns (allmem.build_bank), retrieve top memory nodes for the question
(allmem.retrieve), materialize their immutable source turns, answer with a local 7B
reader, and judge correctness vs. the gold answer with the same model — all against the
vLLM endpoint at http://localhost:8001/v1 (planner.md Sec. 5).

Parallelizes across questions (one history per worker) with a concurrency cap on the
endpoint. Checkpoints answers to JSONL as it goes so a restart resumes.

Usage:
  .venv/bin/python -m allmem.run_allmem --split dev --concurrency 7 \
      --answers runs/analysis/allmem_answers.jsonl --metrics runs/analysis/allmem_100.json

NOTE: this is a clean-room reimplementation from arXiv:2603.19595, NOT the authors' code.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# embedder on a free GPU (0,1 host the vLLM servers); must be set before torch import.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

from allmem.allmem import (AllMemConfig, LLM, build_bank, retrieve, materialize_evidence,
                           reader_prompt, judge_prompt, get_sbert, embed_texts,
                           READER_SYS, JUDGE_SYS)

DATA = "data/lme_s_k"


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def group_history_notes(hids: set[str]) -> dict[str, list[dict]]:
    """Stream notes.jsonl once; collect turns for the wanted histories, ordered."""
    by = defaultdict(list)
    with open(os.path.join(DATA, "notes.jsonl")) as f:
        for line in f:
            if '"history_id"' not in line:
                continue
            n = json.loads(line)
            if n["history_id"] in hids:
                by[n["history_id"]].append(n)
    # order by (session_id, turn_idx) — matches the cached-embedding row order
    order = {}
    for hid, ns in by.items():
        sess_order, seen = [], set()
        for n in ns:
            if n["session_id"] not in seen:
                seen.add(n["session_id"]); sess_order.append(n["session_id"])
        rank = {s: i for i, s in enumerate(sess_order)}
        ns.sort(key=lambda n: (rank[n["session_id"]], n["turn_idx"]))
        order[hid] = ns
    return order


def judge_correct(raw: str) -> int:
    return 1 if re.search(r"correct", raw, re.I) and not re.search(r"wrong", raw, re.I) else 0


class LocalLLM:
    """Per-question proxy over the shared client so llm_calls is counted per history
    (the shared counter is polluted by concurrent threads)."""
    def __init__(self, shared):
        self._s = shared
        self.model = shared.model
        self.n_calls = 0

    def chat(self, *a, **k):
        self.n_calls += 1
        return self._s.chat(*a, **k)


def process_question(q, notes, turn_embs, qz, shared_llm, cfg, note_text) -> dict:
    stats = defaultdict(int)
    t0 = time.time()
    llm = LocalLLM(shared_llm)
    bank = build_bank(notes, turn_embs, llm, cfg, stats)
    units = retrieve(bank, qz, cfg)
    passages, ev_ids = materialize_evidence(units, note_text, cfg)
    # reader
    pred = llm.chat(READER_SYS, reader_prompt(q["question"], passages),
                    max_tokens=cfg.reader_max_tokens) or "I don't know"
    pred = pred.strip()
    # judge
    jraw = llm.chat(JUDGE_SYS, judge_prompt(q["question"], q.get("answer", ""), pred),
                    max_tokens=cfg.judge_max_tokens)
    corr = judge_correct(jraw)
    # retrieval hit: did materialized evidence include any gold note?
    gold = set(q.get("gold_note_ids") or [])
    gold_sess = set(q.get("gold_session_ids") or [])
    note_sess = {nid: (nid.split(":")[1] if ":" in nid else None) for nid in ev_ids}
    hit_note = 1 if gold & set(ev_ids) else 0
    hit_sess = 1 if gold_sess & {note_sess[n] for n in ev_ids if note_sess[n]} else 0
    llm_calls = llm.n_calls
    return {
        "answer_row": {
            "question_id": q["question_id"], "plan": "allmem_reimpl", "answer": pred,
            "judge": {"correct": corr, "judge_model": cfg.model, "raw": jraw},
        },
        "meta": {
            "question_id": q["question_id"], "question_type": q["question_type"],
            "n_units_total": stats["n_units_total"], "n_units_visible": stats["n_units_visible"],
            "n_ops": stats["n_ops"], "split": stats["split"], "merge": stats["merge"],
            "update": stats["update"], "proposed": stats["proposed"],
            "proposed_split_raw": stats.get("proposed_split_raw", 0),
            "proposed_merge_raw": stats.get("proposed_merge_raw", 0),
            "proposed_update_raw": stats.get("proposed_update_raw", 0),
            "split_rej_badid": stats.get("split_rej_badid", 0),
            "split_rej_invisible": stats.get("split_rej_invisible", 0),
            "split_rej_lowconf": stats.get("split_rej_lowconf", 0),
            "llm_calls": llm_calls, "n_evidence": len(ev_ids),
            "hit_note": hit_note, "hit_sess": hit_sess, "correct": corr,
            "wall_s": round(time.time() - t0, 2),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--concurrency", type=int, default=7)
    ap.add_argument("--max-q", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--theta", type=float, default=0.9,
                    help="operator confidence gate (paper default 0.9; lower exercises "
                         "SPLIT/MERGE to characterize the precision/coverage tradeoff)")
    ap.add_argument("--answers", default="runs/analysis/allmem_answers.jsonl")
    ap.add_argument("--metrics", default="runs/analysis/allmem_100.json")
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--shard", type=int, default=0, help="this shard index (0-based)")
    ap.add_argument("--num-shards", type=int, default=1, help="partition questions across N shards by index")
    args = ap.parse_args()

    cfg = AllMemConfig(batch_size=args.batch_size, theta=args.theta)
    questions = [q for q in load_jsonl(os.path.join(DATA, "questions.jsonl"))
                 if q.get("split") == args.split]
    if args.max_q:
        questions = questions[:args.max_q]
    if args.num_shards > 1:
        questions = [q for i, q in enumerate(questions) if i % args.num_shards == args.shard]
    hids = {q["history_id"] for q in questions}
    print(f"[run] split={args.split} questions={len(questions)} histories={len(hids)}", flush=True)

    # resume: skip already-answered question_ids
    done = set()
    if os.path.exists(args.answers):
        for r in load_jsonl(args.answers):
            done.add(r["question_id"])
    questions = [q for q in questions if q["question_id"] not in done]
    print(f"[run] resuming: {len(done)} done, {len(questions)} to do", flush=True)

    # query embeddings (cached)
    qids = json.load(open(os.path.join(DATA, "emb", "queries_ids.json")))
    qmat = np.load(os.path.join(DATA, "emb", "queries.npy"))
    qrow = {qid: qmat[i] for i, qid in enumerate(qids)}

    print("[run] loading histories + cached turn embeddings …", flush=True)
    hist_notes = group_history_notes(hids)
    note_text = {}
    turn_embs = {}
    for hid, ns in hist_notes.items():
        emb = np.load(os.path.join(DATA, "emb", f"{hid}.npy"))
        if emb.shape[0] != len(ns):
            print(f"[warn] {hid}: emb {emb.shape[0]} != notes {len(ns)}; re-embedding", flush=True)
            emb = embed_texts([(n.get("role", "") + ": " + n.get("text", "")) for n in ns])
        turn_embs[hid] = emb
        for n in ns:
            note_text[n["note_id"]] = (n.get("role", "") + ": " + n.get("text", "")) if n.get("role") else n.get("text", "")

    # warm embedder + endpoint
    get_sbert(device="cuda")
    llm = LLM(base_url=args.base_url, model=cfg.model)
    try:
        llm.chat("You are a helper.", "Reply with the single word: ready", max_tokens=8)
    except Exception as e:
        print(f"[warn] warmup failed: {e}", flush=True)

    os.makedirs(os.path.dirname(args.answers) or ".", exist_ok=True)
    write_lock = threading.Lock()
    metas = []
    af = open(args.answers, "a")

    def work(q):
        return process_question(q, hist_notes[q["history_id"]], turn_embs[q["history_id"]],
                                qrow[q["question_id"]].astype(np.float32), llm, cfg, note_text)

    t_start = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(work, q): q for q in questions}
        for fut in as_completed(futs):
            q = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[err] {q['question_id']}: {e}", flush=True)
                continue
            with write_lock:
                af.write(json.dumps(res["answer_row"], ensure_ascii=False) + "\n"); af.flush()
                metas.append(res["meta"])
                n_done += 1
                m = res["meta"]
                print(f"[{n_done}/{len(questions)}] {q['question_id']} {m['question_type'][:18]:18s} "
                      f"corr={m['correct']} hitN={m['hit_note']} units={m['n_units_visible']}/{m['n_units_total']} "
                      f"ops={m['n_ops']} calls={m['llm_calls']} {m['wall_s']}s", flush=True)
    af.close()
    wall = time.time() - t_start

    # ---- aggregate over ALL answered rows (resumed + new), reload answers + metas ----
    # metas only covers this run; reload full answer set for accuracy, and persist metas.
    meta_path = args.metrics.replace(".json", "_meta.jsonl")
    with open(meta_path, "a") as mf:
        for m in metas:
            mf.write(json.dumps(m) + "\n")
    all_meta = list(load_jsonl(meta_path)) if os.path.exists(meta_path) else metas
    # dedupe meta by question_id (keep last)
    mm = {m["question_id"]: m for m in all_meta}
    all_meta = list(mm.values())

    qtype = {q["question_id"]: q["question_type"] for q in load_jsonl(os.path.join(DATA, "questions.jsonl"))}
    acc = [m["correct"] for m in all_meta]
    hitn = [m["hit_note"] for m in all_meta]
    hits = [m["hit_sess"] for m in all_meta]
    by_type = defaultdict(list)
    by_type_hit = defaultdict(list)
    for m in all_meta:
        by_type[qtype.get(m["question_id"], "?")].append(m["correct"])
        by_type_hit[qtype.get(m["question_id"], "?")].append(m["hit_note"])

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    agg = {
        "plan": "allmem_reimpl",
        "note": "All-Mem (from-paper reimpl) of arXiv:2603.19595 — NOT the authors' code.",
        "split": args.split,
        "model": cfg.model,
        "embed_model": "BAAI/bge-base-en-v1.5",
        "n_completed": len(all_meta),
        "qa_accuracy": mean(acc),
        "retrieval_hit_rate_note": mean(hitn),
        "retrieval_hit_rate_session": mean(hits),
        "mean_nodes_total_per_history": mean([m["n_units_total"] for m in all_meta]),
        "mean_nodes_visible_per_history": mean([m["n_units_visible"] for m in all_meta]),
        "mean_llm_calls_per_history": mean([m["llm_calls"] for m in all_meta]),
        "mean_ops_per_history": mean([m["n_ops"] for m in all_meta]),
        "mean_split": mean([m["split"] for m in all_meta]),
        "mean_merge": mean([m["merge"] for m in all_meta]),
        "mean_update": mean([m["update"] for m in all_meta]),
        "mean_proposed_split_raw": mean([m.get("proposed_split_raw", 0) for m in all_meta]),
        "mean_proposed_merge_raw": mean([m.get("proposed_merge_raw", 0) for m in all_meta]),
        "mean_proposed_update_raw": mean([m.get("proposed_update_raw", 0) for m in all_meta]),
        "mean_evidence_turns": mean([m["n_evidence"] for m in all_meta]),
        "per_type": {t: {"n": len(v), "qa_accuracy": mean(v),
                         "hit_rate_note": mean(by_type_hit[t])} for t, v in sorted(by_type.items())},
        "wall_clock_s_this_run": round(wall, 1),
        "mean_wall_s_per_history": mean([m["wall_s"] for m in all_meta]),
        "config": {"theta": cfg.theta, "gamma_merge": cfg.gamma_merge,
                   "batch_size": cfg.batch_size, "anchor_k": cfg.anchor_k,
                   "expand_L": cfg.expand_L, "expand_H": cfg.expand_H,
                   "final_K": cfg.final_K, "max_context_tokens": cfg.max_context_tokens},
    }
    with open(args.metrics, "w") as f:
        json.dump(agg, f, indent=2)
    print("\n=== All-Mem (from-paper reimpl) dev summary ===", flush=True)
    print(json.dumps({k: agg[k] for k in ["n_completed", "qa_accuracy", "retrieval_hit_rate_note",
          "retrieval_hit_rate_session", "mean_nodes_visible_per_history",
          "mean_llm_calls_per_history", "mean_ops_per_history", "wall_clock_s_this_run"]}, indent=2), flush=True)
    print(f"wrote {args.answers} and {args.metrics}", flush=True)


if __name__ == "__main__":
    main()
