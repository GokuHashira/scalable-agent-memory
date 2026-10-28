"""rankings.jsonl -> bundles.jsonl under a token / result budget (CPU only).

Default budget (2,048 tok, 20 notes) writes into the ranking's own run dir; any
other budget writes runs/<run>_b<cap> so the budget sweep stays separate.

  python scripts/pack_rankings.py --run runs/vref_hybrid_lme_s --data data/lme_s [--cap 1024 --max-results 10]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import MAX_RESULTS, TOKEN_CAP, NoteIndex, write_json  # noqa: E402
from mem27.eval.packing import pack  # noqa: E402
from mem27.vschema import iter_jsonl, write_jsonl  # noqa: E402


def pack_run(run: Path, data: Path, cap: int, max_results: int, idx: NoteIndex | None = None) -> Path:
    rankings = list(iter_jsonl(run / "rankings.jsonl"))
    if idx is None:
        idx = NoteIndex(data / "notes.jsonl", keep_text=True)
    default = cap == TOKEN_CAP and max_results == MAX_RESULTS
    out = run if default else run.parent / f"{run.name}_b{cap}"
    rows = []
    for r in rankings:
        t0 = time.perf_counter()
        b = pack(r["note_ids"], idx.text, r["scores"], token_cap=cap, max_results=max_results)
        stages = dict(r["stages"])
        stages["pack"] = (time.perf_counter() - t0) * 1000
        b.update({"question_id": r["question_id"], "plan": r["plan"],
                  "latency_ms": {"total": sum(stages.values()), "stages": stages},
                  "stale_flags": [False] * len(b["note_ids"]),
                  "trace": {"cap": cap, "max_results": max_results, "index_build_ms": r.get("index_build_ms")}})
        rows.append(b)
    write_jsonl(out / "bundles.jsonl", rows)
    if not default:
        for f in ("env.json", "run_config.json"):
            if (run / f).exists():
                shutil.copy(run / f, out / f)
    write_json(out / "budget.json", {"cap": cap, "max_results": max_results, "source": run.name})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cap", type=int, nargs="+", default=[TOKEN_CAP])
    ap.add_argument("--max-results", type=int, nargs="+", default=None,
                    help="one per cap; default scales 20 * cap / 2048")
    a = ap.parse_args(argv)
    data = Path(a.data)
    idx = NoteIndex(data / "notes.jsonl", keep_text=True)
    mrs = a.max_results or [max(1, round(MAX_RESULTS * c / TOKEN_CAP)) for c in a.cap]
    for run in a.run:
        for cap, mr in zip(a.cap, mrs):
            print(pack_run(Path(run), data, cap, mr, idx))


if __name__ == "__main__":
    main()
