"""LongMemEval_M feasibility (V -> K): per-history BM25 build/query time on 20 questions, memory,
unique-text count (drives dense-embedding cost), and session recall vs S for the same questions.

  python scripts/lme_m_feasibility.py --data data/lme_m --n 20 --out runs/_tables/lme_m_feasibility.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import write_json  # noqa: E402
from mem27.vschema import Question, iter_jsonl, read_jsonl, session_of  # noqa: E402
from scripts.vref_retrieve import tok  # noqa: E402


def main(argv=None):
    from rank_bm25 import BM25Okapi

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lme_m")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="runs/_tables/lme_m_feasibility.json")
    a = ap.parse_args(argv)
    data = Path(a.data)
    qs = read_jsonl(data / "questions.jsonl", Question)
    test = [q for q in qs if q.split == "dev" and not q.question_id.endswith("_abs")]
    pick = {q.question_id for q in sorted(test, key=lambda q: q.question_id)[: a.n]}
    texts = defaultdict(list)
    uniq = set()
    t0 = time.time()
    n_turns = 0
    for n in iter_jsonl(data / "notes.jsonl"):
        if n["turn_idx"] < 0:
            continue
        n_turns += 1
        uniq.add(hashlib.md5(n["text"].encode()).hexdigest())
        if n["history_id"] in pick:
            texts[n["history_id"]].append((n["note_id"], n["text"]))
    scan_s = time.time() - t0
    rows = []
    for q in qs:
        if q.question_id not in pick:
            continue
        ids = [i for i, _ in texts[q.history_id]]
        t0 = time.perf_counter()
        bm = BM25Okapi([tok(t) or ["_"] for _, t in texts[q.history_id]])
        build = time.perf_counter() - t0
        t0 = time.perf_counter()
        s = bm.get_scores(tok(q.question) or ["_"])
        order = np.argsort(-s)[:200]
        query = time.perf_counter() - t0
        sess = []
        for i in order:
            sid = session_of(ids[i])
            if sid not in sess:
                sess.append(sid)
        gold = set(q.gold_session_ids)
        rows.append({"question_id": q.question_id, "n_turns": len(ids), "bm25_build_s": build, "bm25_query_ms": query * 1000,
                     "sess_recall@5": len(gold & set(sess[:5])) / len(gold), "sess_recall@20": len(gold & set(sess[:20])) / len(gold)})
    res = {
        "n_turn_notes": n_turns, "unique_turn_texts": len(uniq), "notes_scan_s": scan_s,
        "turns_per_history_mean": float(np.mean([r["n_turns"] for r in rows])),
        "bm25_build_s_mean": float(np.mean([r["bm25_build_s"] for r in rows])),
        "bm25_query_ms_p50": float(np.percentile([r["bm25_query_ms"] for r in rows], 50)),
        "bm25_query_ms_p95": float(np.percentile([r["bm25_query_ms"] for r in rows], 95)),
        "sess_recall@5": float(np.mean([r["sess_recall@5"] for r in rows])),
        "sess_recall@20": float(np.mean([r["sess_recall@20"] for r in rows])),
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "dense_embedding_estimate": {"cpu_node_hours": len(uniq) / 189515 * 3.5,
                                     "note": "scaled from LongMemEval_S: 189,515 unique turns took 4 EPYC nodes x 52 min"},
        "per_question": rows,
    }
    write_json(a.out, res)
    print(json.dumps({k: v for k, v in res.items() if k != "per_question"}, indent=1))


if __name__ == "__main__":
    main()
