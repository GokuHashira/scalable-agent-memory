"""Evaluation-side probe (vprobe_*, evidence for G, not a system): recency-aware rerank of an existing ranking.

score' = score * (1 + b * r), r = (ts_note - ts_first_note_of_history) / (question_date - ts_first_note) in [0, 1].
b = 0.5 is the value fixed on the synthetic stream (findings §3); it is not tuned here. The reranked top-200
is packed under the default budget (2,048 tokens / 20 notes), then metrics and KU Freshness@k are computed, so
the probe shows both the freshness gain on knowledge-update pairs and the recall cost on other questions.

  python scripts/vprobe_recency.py --run runs/vref_hybrid_lme_s --data data/lme_s --boost 0.5
  -> runs/vprobe_recency_hybrid_lme_s/{rankings,bundles}.jsonl, metrics.json (no answers), freshness_ku.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import NoteIndex, write_json  # noqa: E402
from mem27.vschema import Question, iter_jsonl, read_jsonl, write_jsonl  # noqa: E402
from scripts.pack_rankings import pack_run  # noqa: E402


def ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--boost", type=float, default=0.5)
    ap.add_argument("--prefix", default="vprobe_recency")
    a = ap.parse_args(argv)
    data = Path(a.data)
    qs = {q.question_id: q for q in read_jsonl(data / "questions.jsonl", Question)}
    idx = NoteIndex(data / "notes.jsonl", keep_text=True)
    first = {h: min(ts(idx.meta[n][4]) for n in ns) for h, ns in idx.by_history.items()}
    for run in map(Path, a.run):
        method = run.name.split("_lme")[0].replace("vref_", "")
        out = run.parent / f"{a.prefix}_{method}_{data.name}"
        rows = []
        for r in iter_jsonl(run / "rankings.jsonl"):
            q = qs[r["question_id"]]
            t0 = time.perf_counter()
            lo, hi = first[q.history_id], ts(q.question_date)
            new = []
            for nid, s in zip(r["note_ids"], r["scores"]):
                rec = (ts(idx.meta[nid][4]) - lo) / max(1.0, hi - lo)
                new.append((nid, s * (1 + a.boost * max(0.0, min(1.0, rec)))))
            new.sort(key=lambda kv: (-kv[1], kv[0]))
            stages = {**r["stages"], "recency": (time.perf_counter() - t0) * 1000}
            rows.append({**r, "plan": f"{a.prefix}_{method}", "note_ids": [n for n, _ in new],
                         "scores": [s for _, s in new], "stages": stages})
        write_jsonl(out / "rankings.jsonl", rows)
        for f in ("env.json",):
            if (run / f).exists():
                (out / f).write_text((run / f).read_text())
        cfg = json.loads((run / "run_config.json").read_text()) if (run / "run_config.json").exists() else {}
        write_json(out / "run_config.json", {**cfg, "probe": "recency rerank of " + run.name, "boost": a.boost,
                                             "recency": "(ts - first note of history) / (question_date - first note)"})
        pack_run(out, data, 2048, 20, idx)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
