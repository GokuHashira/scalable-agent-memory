"""Counterfactual bundles: where does the gap between retrieval and the gold-session ceiling come from?

All variants start from the same ranking (default vref_hybrid) and keep the 2,048-token cap:
  cf_nbr1     each ranked turn is packed together with its +-1 neighbouring turns (local context)
  cf_goldonly oracle filter: drop ranked turns outside gold sessions (cost of distractors at equal order)
  cf_sessexp  sessions in order of first hit; each is packed outward from its hit turns (session context)
max_results is lifted to 60 so the comparison is purely about which tokens fill the same budget.

  python scripts/counterfactual_bundles.py --run runs/vref_hybrid_lme_s --data data/lme_s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import NoteIndex, write_json  # noqa: E402
from mem27.eval.packing import pack  # noqa: E402
from mem27.vschema import Question, iter_jsonl, read_jsonl, session_of, write_jsonl  # noqa: E402

CAP, MAXR = 2048, 60


def nbr(nid: str, d: int) -> str:
    h, s, t = nid.rsplit(":", 2)
    return f"{h}:{s}:{int(t) + d}"


def order_nbr1(ranked: list[str], idx: NoteIndex) -> list[str]:
    out, seen = [], set()
    for n in ranked:
        for m in (nbr(n, -1), n, nbr(n, +1)):
            if m in idx.meta and m not in seen and idx.turn(m) >= 0:
                out.append(m)
                seen.add(m)
    return out


def order_sessexp(ranked: list[str], idx: NoteIndex) -> list[str]:
    hits: dict[str, list[int]] = {}
    for n in ranked:
        hits.setdefault(session_of(n), []).append(idx.turn(n))
    out = []
    for s, ts in hits.items():
        h = ranked[0].rsplit(":", 2)[0]
        seen = set()
        for dist in range(0, 200):
            added = False
            for t in ts:
                for c in {t - dist, t + dist}:
                    m = f"{h}:{s}:{c}"
                    if m in idx.meta and m not in seen:
                        out.append(m)
                        seen.add(m)
                        added = True
            if not added and dist > 0:
                break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/vref_hybrid_lme_s")
    ap.add_argument("--data", default="data/lme_s")
    a = ap.parse_args(argv)
    run, data = Path(a.run), Path(a.data)
    qs = {q.question_id: q for q in read_jsonl(data / "questions.jsonl", Question)}
    idx = NoteIndex(data / "notes.jsonl", keep_text=True)
    rankings = list(iter_jsonl(run / "rankings.jsonl"))
    variants = {"cf_nbr1": [], "cf_goldonly": [], "cf_sessexp": []}
    for r in rankings:
        q = qs[r["question_id"]]
        ranked = r["note_ids"]
        orders = {
            "cf_nbr1": order_nbr1(ranked, idx),
            "cf_goldonly": [n for n in ranked if session_of(n) in set(q.gold_session_ids)] or ranked,
            "cf_sessexp": order_sessexp(ranked, idx),
        }
        for k, o in orders.items():
            b = pack(o, idx.text, token_cap=CAP, max_results=MAXR)
            b.update({"question_id": q.question_id, "plan": k, "latency_ms": {"total": 0.0, "stages": {}},
                      "stale_flags": [False] * len(b["note_ids"]), "trace": {"source": run.name}})
            variants[k].append(b)
    for k, rows in variants.items():
        out = run.parent / f"{k}_{run.name.replace('vref_', '')}"
        write_jsonl(out / "bundles.jsonl", rows)
        write_json(out / "run_config.json", {"variant": k, "source": run.name, "cap": CAP, "max_results": MAXR})
        print(out, sum(len(b["note_ids"]) for b in rows) / len(rows), "notes/bundle")


if __name__ == "__main__":
    main()
