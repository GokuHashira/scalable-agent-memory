"""V's reference retrievers for harness validation (NOT G's P0-P5; G's runs supersede these).

vref_bm25 (rank_bm25 Okapi), vref_dense (bge-base cosine, exact), vref_hybrid (RRF k=60 over the
top-100 of each). Turn-level notes only, per-history index. Writes, per method,
runs/<method>_<tag>/rankings.jsonl with the top-200 note ids + scores and per-stage
latency, then packs the default bundle (2,048 tok / 20 notes) via pack_rankings.

  python scripts/vref_retrieve.py --data data/lme_s --tag lme_s --emb runs/_emb/bge_base
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import NoteIndex, collect_env, write_json  # noqa: E402
from mem27.eval.embed import QUERY_PREFIX, EmbCache  # noqa: E402
from mem27.vschema import Question, read_jsonl, write_jsonl  # noqa: E402

TOP = 200
RRF_K = 60
STOP = set("""a an the and or but if of to in on at for with by from as is are was were be been being i me my
mine we our you your he she it its they them their this that these those what which who whom whose when where why
how do does did done have has had having can could should would will shall may might must not no yes so than too
very just about into over under again further then once there here all any both each few more most other some such
only own same s t don now also up down out off""".split())
TOKEN_RE = re.compile(r"[a-z0-9]+")


def tok(text: str) -> list[str]:
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOP]


def rrf(rankings: list[list[str]], k: int = RRF_K, depth: int = 100) -> list[tuple[str, float]]:
    score: dict[str, float] = {}
    for r in rankings:
        for i, nid in enumerate(r[:depth]):
            score[nid] = score.get(nid, 0.0) + 1.0 / (k + i + 1)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


def main(argv=None):
    from rank_bm25 import BM25Okapi

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--emb", default="runs/_emb/bge_base")
    ap.add_argument("--methods", nargs="+", default=["vref_bm25", "vref_dense", "vref_hybrid"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    data = Path(a.data)
    qs = read_jsonl(data / "questions.jsonl", Question)
    idx = NoteIndex(data / "notes.jsonl", keep_text=True)
    need_dense = any(m != "vref_bm25" for m in a.methods)
    emb = None
    if need_dense:
        emb = EmbCache(a.emb, device=a.device)
        turn_texts = [idx.text[n] for n in idx.meta if idx.meta[n][2] >= 0]
        info = emb.add(turn_texts)
        print(f"[embed] corpus: {info}", flush=True)
        emb.model  # load before timing queries
        emb.encode([QUERY_PREFIX + "warm up"])
    env = collect_env(seed=0)
    out = {m: [] for m in a.methods}
    t_start = time.time()
    for qi, q in enumerate(qs):
        ids = [n for n in idx.by_history[q.history_id] if idx.meta[n][2] >= 0]
        texts = [idx.text[n] for n in ids]
        t0 = time.perf_counter()
        bm = BM25Okapi([tok(t) or ["_"] for t in texts])
        build_bm25 = (time.perf_counter() - t0) * 1000
        res: dict[str, tuple[list[str], list[float], dict]] = {}
        if any(m in ("vref_bm25", "vref_hybrid") for m in a.methods):
            t0 = time.perf_counter()
            s = bm.get_scores(tok(q.question) or ["_"])
            order = np.argsort(-s, kind="stable")[:TOP]
            bm_ms = (time.perf_counter() - t0) * 1000
            res["vref_bm25"] = ([ids[i] for i in order], [float(s[i]) for i in order], {"bm25": bm_ms})
        if need_dense:
            t0 = time.perf_counter()
            qv = emb.encode([QUERY_PREFIX + q.question])[0].astype(np.float32)
            qe_ms = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            M = emb.get(texts).astype(np.float32)
            s = M @ qv
            order = np.argsort(-s, kind="stable")[:TOP]
            de_ms = (time.perf_counter() - t0) * 1000
            res["vref_dense"] = ([ids[i] for i in order], [float(s[i]) for i in order],
                                 {"query_embed": qe_ms, "dense_search": de_ms})
        if "vref_hybrid" in a.methods:
            t0 = time.perf_counter()
            fused = rrf([res["vref_bm25"][0], res["vref_dense"][0]])[:TOP]
            f_ms = (time.perf_counter() - t0) * 1000
            st = {**res["vref_bm25"][2], **res["vref_dense"][2], "rrf": f_ms}
            res["vref_hybrid"] = ([n for n, _ in fused], [s for _, s in fused], st)
        for m in a.methods:
            r_ids, r_sc, stages = res[m]
            out[m].append({"question_id": q.question_id, "plan": m, "note_ids": r_ids, "scores": r_sc,
                           "stages": stages, "index_build_ms": {"bm25": build_bm25}})
        if qi % 100 == 0:
            print(f"[retrieve] {qi}/{len(qs)} {time.time() - t_start:.0f}s", flush=True)
    for m in a.methods:
        run = Path(a.runs) / f"{m}_{a.tag}"
        write_jsonl(run / "rankings.jsonl", out[m])
        write_json(run / "env.json", env)
        write_json(run / "run_config.json", {"method": m, "data": str(data), "top": TOP, "rrf_k": RRF_K,
                                             "embedding": "BAAI/bge-base-en-v1.5", "granularity": "turn"})
        print(f"wrote {run}/rankings.jsonl", flush=True)


if __name__ == "__main__":
    main()
