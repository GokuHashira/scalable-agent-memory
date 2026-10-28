"""Streaming freshness on the synthetic workload with dense and hybrid reference retrievers (CPU, no LLM).

Companion of scripts/synth_ref_stream.py (BM25). Embeds every note text of notes_final.jsonl and every question
once into an EmbCache (bge-base-en-v1.5), then runs stream_eval (updates fed in ts order, questions answered
mid-stream) and a static run on notes_final for each variant, and writes freshness.json per run.

  python scripts/synth_ref_stream_dense.py --data data/synth_10k --emb runs/_emb/bge_base_synth \
      --summary-out runs/_tables/synth_ref_freshness_dense.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.data.updates import stream_eval, to_note  # noqa: E402
from mem27.eval.common import write_json  # noqa: E402
from mem27.eval.embed import EmbCache  # noqa: E402
from mem27.eval.freshness import synth_freshness  # noqa: E402
from mem27.eval.ref_dense import RefDense, RefHybrid  # noqa: E402
from mem27.vschema import RetrievalRequest, iter_jsonl, write_jsonl  # noqa: E402

VARIANTS = {
    "plain": dict(version_filter="none", recency_boost=0.0),
    "vfilter": dict(version_filter="explicit", recency_boost=0.0),
    "recency": dict(version_filter="none", recency_boost=0.5),
    "vfilter_recency": dict(version_filter="explicit", recency_boost=0.5),
}
CLASSES = {"dense": RefDense, "hybrid": RefHybrid}


def static_run(data: Path, run: Path, plan: str, make) -> None:
    r = make()
    r.build([to_note(n) for n in iter_jsonl(data / "notes_final.jsonl")])
    rows = []
    for q in iter_jsonl(data / "questions.jsonl"):
        b = r.retrieve(RetrievalRequest(query=q["question"], as_of_time=q.get("as_of_time") or q["question_date"],
                                        plan=plan, question_id=q["question_id"]))
        rows.append({**asdict(b), "question_id": q["question_id"], "plan": plan})
    write_jsonl(run / "bundles.jsonl", rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--emb", default="runs/_emb/bge_base_synth")
    ap.add_argument("--methods", nargs="+", default=list(CLASSES))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--summary-out", required=True)
    a = ap.parse_args(argv)
    emb = EmbCache(a.emb)
    summary = {}
    for d in a.data:
        data = Path(d)
        texts = [n["text"] for n in iter_jsonl(data / "notes_final.jsonl")]
        info = emb.add(texts)
        print(f"[embed] {data.name}: {info}", flush=True)
        # one batch of query embeddings per dataset, shared by every retriever instance
        qtexts = [q["question"] for f in ("questions.jsonl", "questions_stream.jsonl") for q in iter_jsonl(data / f)]
        shared: dict = {}

        def _new(m, v):
            r = CLASSES[m](emb, **VARIANTS[v])
            (r.de if hasattr(r, "de") else r)._qcache = shared
            r.prefetch_queries(qtexts)
            return r
        t0 = time.time()
        _new("dense", "plain")
        print(f"[embed] {len(shared)} queries in {time.time() - t0:.0f}s", flush=True)
        for m in a.methods:
            for v in a.variants:
                plan = f"ref_{m}_{v}"
                make = lambda m=m, v=v: _new(m, v)  # noqa: E731
                t0 = time.time()
                run = Path(a.runs) / f"{plan}_{data.name}_stream"
                info = stream_eval(make(), data, run, plan)
                fr = synth_freshness(run, data, "questions_stream.jsonl")
                write_json(run / "freshness.json", fr)
                srun = Path(a.runs) / f"{plan}_{data.name}"
                static_run(data, srun, plan, make)
                fs = synth_freshness(srun, data, "questions.jsonl")
                write_json(srun / "freshness.json", fs)
                key = f"{data.name}/{m}/{v}"
                pick = lambda g: {k: round(x, 3) for k, x in g.items() if k in ("n", "fresh@1", "fresh@5", "stale_in@5", "latest_in@20")}  # noqa: E731
                summary[key] = {"stream": {g: pick(x) for g, x in fr["groups"].items() if not g.startswith("type:")},
                                "static_types": {g: pick(x) for g, x in fs["groups"].items() if g.startswith("type:")},
                                "seconds": round(time.time() - t0, 1), "updates_fed": info["n_updates_fed"]}
                print(key, json.dumps(summary[key]["stream"]), flush=True)
                write_json(a.summary_out, summary)


if __name__ == "__main__":
    main()
