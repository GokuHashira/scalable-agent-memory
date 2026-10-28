"""Freshness + maintenance metrics (EVAL-03).

Freshness@k (per updated fact): 1 if a note asserting the latest version is in the
  top-k AND ranks above every superseded version present in the ranking.
  The ranking is bundle.trace.ranked_ids when present (full ranking), else note_ids.
Stale-served rate (per answered question about an updated fact): the answer states a
  superseded value and not the latest one ("stale_only"); "mentions_stale" also
  counts answers that give both. Synthetic: word-boundary string match (values are
  invented, so matches are unambiguous). LongMemEval KU: LLM stale judge
  (configs/prompts/stale_judge.txt), since old/new values are free text.
Both are split by explicit vs implicit supersession (and by supersede vs correct).
latest_in@k and fresh_given_latest@k separate ordering errors from plain recall misses.
Maintenance work per update: mean / p50 / p95 of each insert-trace field
  (embed_ms, bm25_ms, ann_ms, summary_ms, summary_tokens, links_ms) and of their ms sum,
  per op and explicit flag.

  python -m mem27.eval.freshness --run runs/x --data data/synth_10k [--questions questions_stream.jsonl]
  python -m mem27.eval.freshness --run runs/vref_hybrid_lme_s --data data/lme_s --ku
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from mem27.eval.common import write_json
from mem27.vschema import iter_jsonl

KS = (1, 5, 10, 20)
MS_FIELDS = ("embed_ms", "bm25_ms", "ann_ms", "summary_ms", "links_ms")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\- ]", " ", (s or "").lower())).strip()


def contains(text: str, value: str) -> bool:
    v = norm(value)
    return bool(v) and re.search(rf"(?<![a-z0-9]){re.escape(v)}(?![a-z0-9])", norm(text)) is not None


def ranks(ranking: list[str], ids: set[str]) -> int | None:
    for i, n in enumerate(ranking):
        if n in ids:
            return i
    return None


def fresh_row(ranking: list[str], latest: set[str], stale: set[str]) -> dict:
    rl, rs = ranks(ranking, latest), ranks(ranking, stale)
    row = {"rank_latest": rl, "rank_stale": rs}
    for k in KS:
        row[f"fresh@{k}"] = float(rl is not None and rl < k and (rs is None or rl < rs))
        row[f"stale_in@{k}"] = float(rs is not None and rs < k)
        row[f"latest_in@{k}"] = float(rl is not None and rl < k)
    return row


def synth_correct(answer: str, q: dict) -> bool:
    if q["question_type"] == "what_changed":
        f = q["fact"]
        return contains(answer, f["old_value"]) and contains(answer, f["new_value"])
    return contains(answer, q["answer"])


def _group_key(q: dict) -> str:
    f = q.get("fact") or {}
    e = f.get("latest_change_explicit")
    kind = f.get("latest_change_kind")
    if kind is None:
        return "no_change"
    return f"{kind}:{'explicit' if e else 'implicit'}"


def _agg(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    keys = list(dict.fromkeys(k for r in rows for k in r))  # union: the first row may lack fresh@k (no update)
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        if vals and k.startswith(("fresh", "stale", "latest", "correct", "mentions")):
            out[k] = float(np.mean(vals))
    for k in KS:  # freshness conditional on the latest version being retrieved (separates ordering from recall)
        hit = [r for r in rows if r.get(f"latest_in@{k}") == 1.0]
        if hit:
            out[f"fresh_given_latest@{k}"] = float(np.mean([r[f"fresh@{k}"] for r in hit]))
    return out


def synth_freshness(run_dir: Path, data_dir: Path, questions_file: str, answers_name: str = "answers.jsonl") -> dict:
    qs = {q["question_id"]: q for q in iter_jsonl(data_dir / questions_file)}
    answers = {a["question_id"]: a for a in iter_jsonl(run_dir / answers_name)} \
        if (run_dir / answers_name).exists() else {}
    rows = []
    for b in iter_jsonl(run_dir / "bundles.jsonl"):
        q = qs.get(b["question_id"])
        if q is None:
            continue
        f = q.get("fact") or {}
        ranking = (b.get("trace") or {}).get("ranked_ids") or b["note_ids"]
        r = {"question_id": q["question_id"], "question_type": q["question_type"], "split": q.get("split"),
             "group": _group_key(q), "n_versions": f.get("n_versions")}
        stale_ids = set(f.get("superseded_note_ids") or [])
        if stale_ids and q["question_type"] in ("latest_value", "exact_lookup", "semantic_recall", "point_in_time"):
            r.update(fresh_row(ranking, set(q["gold_note_ids"]), stale_ids))
        a = answers.get(q["question_id"])
        if a is not None:
            ans = a.get("answer", "")
            r["correct_str"] = float(synth_correct(ans, q))
            if (a.get("judge") or {}).get("correct") is not None:
                r["correct_judge"] = float(bool(a["judge"]["correct"]))
            sv = f.get("stale_values") or []
            if sv and q["question_type"] != "what_changed":
                ment = any(contains(ans, v) for v in sv)
                r["mentions_stale"] = float(ment)
                r["stale_only"] = float(ment and not contains(ans, q["answer"]))
        rows.append(r)
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
        groups[f"type:{r['question_type']}"].append(r)
        groups["ALL"].append(r)
    updated = [r for r in rows if "fresh@5" in r]
    return {"run_id": run_dir.name, "dataset": data_dir.name, "questions_file": questions_file,
            "definitions": __doc__, "groups": {g: _agg(v) for g, v in sorted(groups.items())},
            "updated_facts": _agg(updated), "per_question": rows}


def ku_freshness(run_dir: Path, data_dir: Path, answers_name: str = "answers.jsonl") -> dict:
    pairs = {p["question_id"]: p for p in iter_jsonl(data_dir / "ku_pairs.jsonl")}
    answers = {a["question_id"]: a for a in iter_jsonl(run_dir / answers_name)} \
        if (run_dir / answers_name).exists() else {}
    src = run_dir / "rankings.jsonl" if (run_dir / "rankings.jsonl").exists() else run_dir / "bundles.jsonl"
    rows = []
    for b in iter_jsonl(src):
        p = pairs.get(b["question_id"])
        if p is None:
            continue
        r = {"question_id": p["question_id"], "split": p["split"],
             "group": "explicit" if p.get("explicit_llm", p["explicit_heuristic"]) else "implicit"}
        r.update(fresh_row(b["note_ids"], set(p["new_note_ids"]), set(p["old_note_ids"])))
        a = answers.get(p["question_id"])
        if a is not None:
            if a.get("judge"):
                r["correct_judge"] = float(bool(a["judge"]["correct"]))
            st = (a.get("judges") or {}).get("stale")
            if st is not None and st.get("correct") is not None:  # empty / failed verdicts are excluded
                r["stale_only"] = float(bool(st["correct"]))
                # stale_wrong: states the superseded value AND is judged wrong. Some KU questions ask for the
                # earlier value ("initially", "before"), where stating it is correct (Oct 14 spot check).
                if a.get("judge"):
                    r["stale_wrong"] = float(bool(st["correct"]) and not a["judge"]["correct"])
        rows.append(r)
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
        groups["ALL"].append(r)
        groups[f"split:{r['split']}"].append(r)
    return {"run_id": run_dir.name, "dataset": data_dir.name, "source": src.name, "definitions": __doc__,
            "groups": {g: _agg(v) for g, v in sorted(groups.items())}, "per_question": rows}


def maintenance(traces_path: str | Path) -> dict:
    groups = defaultdict(list)
    for t in iter_jsonl(traces_path):
        t = dict(t)
        t["work_ms"] = sum(float(t.get(f) or 0.0) for f in MS_FIELDS)
        groups["ALL"].append(t)
        groups[f"{t.get('op')}:{'explicit' if t.get('explicit') else 'implicit'}"].append(t)
    out = {}
    for g, ts in groups.items():
        d = {"n": len(ts)}
        for f in MS_FIELDS + ("summary_tokens", "work_ms"):
            v = [float(t.get(f) or 0.0) for t in ts]
            d[f] = {"mean": float(np.mean(v)), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                    "total": float(np.sum(v))}
        out[g] = d
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--questions", default="questions.jsonl")
    ap.add_argument("--answers", default="answers.jsonl")
    ap.add_argument("--ku", action="store_true")
    ap.add_argument("--out-name", default="freshness.json")
    a = ap.parse_args(argv)
    for run in a.run:
        run = Path(run)
        res = ku_freshness(run, Path(a.data), a.answers) if a.ku else \
            synth_freshness(run, Path(a.data), a.questions, a.answers)
        if (run / "insert_traces.jsonl").exists():
            res["maintenance"] = maintenance(run / "insert_traces.jsonl")
        write_json(run / a.out_name, res)
        brief = {g: {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()
                     if k in ("n", "fresh@5", "fresh@20", "stale_in@5", "correct_str", "correct_judge", "stale_only", "stale_wrong")}
                 for g, d in res["groups"].items() if not g.startswith("type:")}
        print(run.name, json.dumps(brief))


if __name__ == "__main__":
    main()
