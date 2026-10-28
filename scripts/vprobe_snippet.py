"""Evaluation-side probe (vprobe_*, evidence for G, not a system): snippet packing.

Same ranking as the source run, same 2,048-token cap, but every turn is cut to its first `--snippet` tokens
(cl100k) before packing, and up to `--max-results` turns may enter. Tests findings §4: the cap binds at about 10
full turns, so shorter units fit more distinct hits. Head truncation is crude (the answer can sit later in a
long turn), so the reader pass decides whether coverage converts into accuracy.

  python scripts/vprobe_snippet.py --run runs/vref_hybrid_lme_s --data data/lme_s --snippet 128 --max-results 40
  -> runs/vprobe_snippet128_hybrid_lme_s/{rankings,bundles}.jsonl, budget.json, env.json, run_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import NoteIndex, truncate_tokens, write_json  # noqa: E402
from mem27.eval.packing import pack  # noqa: E402
from mem27.vschema import iter_jsonl, write_jsonl  # noqa: E402

CAP = 2048


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--snippet", type=int, default=128)
    ap.add_argument("--max-results", type=int, default=40)
    a = ap.parse_args(argv)
    run, data = Path(a.run), Path(a.data)
    method = run.name.split("_lme")[0].replace("vref_", "")
    out = run.parent / f"vprobe_snippet{a.snippet}_{method}_{data.name}"
    rankings = list(iter_jsonl(run / "rankings.jsonl"))
    need = {n for r in rankings for n in r["note_ids"][: 4 * a.max_results]}
    idx = NoteIndex(data / "notes.jsonl", keep_text=True)
    snip = {n: truncate_tokens(idx.text[n], a.snippet) for n in need}
    rows = []
    for r in rankings:
        t0 = time.perf_counter()
        ids = r["note_ids"][: 4 * a.max_results]
        b = pack(ids, snip, r["scores"][: len(ids)], token_cap=CAP, max_results=a.max_results)
        stages = {**r["stages"], "pack": (time.perf_counter() - t0) * 1000}
        b.update({"question_id": r["question_id"], "plan": f"vprobe_snippet{a.snippet}_{method}",
                  "latency_ms": {"total": sum(stages.values()), "stages": stages},
                  "stale_flags": [False] * len(b["note_ids"]),
                  "trace": {"cap": CAP, "max_results": a.max_results, "snippet_tokens": a.snippet, "source": run.name}})
        rows.append(b)
    write_jsonl(out / "bundles.jsonl", rows)
    write_jsonl(out / "rankings.jsonl", rankings)
    for f in ("env.json",):
        if (run / f).exists():
            (out / f).write_text((run / f).read_text())
    cfg = json.loads((run / "run_config.json").read_text()) if (run / "run_config.json").exists() else {}
    write_json(out / "run_config.json", {**cfg, "probe": f"snippet packing ({a.snippet} tok/turn) of {run.name}"})
    write_json(out / "budget.json", {"cap": CAP, "max_results": a.max_results, "source": run.name, "snippet": a.snippet})
    print(f"wrote {out}: mean notes/bundle {sum(len(b['note_ids']) for b in rows) / len(rows):.1f}", flush=True)


if __name__ == "__main__":
    main()
