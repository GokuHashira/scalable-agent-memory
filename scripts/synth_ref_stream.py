"""Streaming freshness on the synthetic workload with V's reference BM25 (CPU only).

Variants: plain (no version handling) / vfilter (hide notes closed by explicit
supersessions) / recency (score x (1 + 0.5 recency)) / vfilter+recency.
For each synthetic dataset: stream_eval over questions_stream.jsonl (updates fed
in ts order, questions answered mid-stream) and a static run over questions.jsonl
on notes_final.jsonl, then freshness.json per run.

  python scripts/synth_ref_stream.py --data data/synth_10k data/synth_10k_high
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
from mem27.eval.freshness import synth_freshness  # noqa: E402
from mem27.eval.ref_retriever import RefBM25  # noqa: E402
from mem27.vschema import RetrievalRequest, iter_jsonl, write_jsonl  # noqa: E402

VARIANTS = {
    "plain": dict(version_filter="none", recency_boost=0.0),
    "vfilter": dict(version_filter="explicit", recency_boost=0.0),
    "recency": dict(version_filter="none", recency_boost=0.5),
    "vfilter_recency": dict(version_filter="explicit", recency_boost=0.5),
}


def static_run(data: Path, run: Path, plan: str, kw: dict) -> None:
    r = RefBM25(**kw)
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
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--summary-out", default=None,
                    help="summary json (default runs/_tables/synth_ref_freshness.json); use a new name to keep old tables")
    a = ap.parse_args(argv)
    summary = {}
    for d in a.data:
        data = Path(d)
        for v in a.variants:
            plan = f"ref_bm25_{v}"
            t0 = time.time()
            run = Path(a.runs) / f"{plan}_{data.name}_stream"
            info = stream_eval(RefBM25(**VARIANTS[v]), data, run, plan)
            fr = synth_freshness(run, data, "questions_stream.jsonl")
            write_json(run / "freshness.json", fr)
            srun = Path(a.runs) / f"{plan}_{data.name}"
            static_run(data, srun, plan, VARIANTS[v])
            fs = synth_freshness(srun, data, "questions.jsonl")
            write_json(srun / "freshness.json", fs)
            key = f"{data.name}/{v}"
            pick = lambda g: {k: round(x, 3) for k, x in g.items() if k in ("n", "fresh@1", "fresh@5", "stale_in@5")}  # noqa: E731
            summary[key] = {"stream": {g: pick(x) for g, x in fr["groups"].items() if not g.startswith("type:")},
                            "static_types": {g: pick(x) for g, x in fs["groups"].items() if g.startswith("type:")},
                            "seconds": round(time.time() - t0, 1), "updates_fed": info["n_updates_fed"]}
            print(key, json.dumps(summary[key]["stream"]), flush=True)
    write_json(a.summary_out or Path(a.runs) / "_tables" / "synth_ref_freshness.json", summary)


if __name__ == "__main__":
    main()
