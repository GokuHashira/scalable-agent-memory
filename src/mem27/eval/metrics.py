"""Metrics (EVAL-02): bundles.jsonl + questions.jsonl (+ answers.jsonl) -> metrics.json.

Definitions (per question, then averaged):
- sess_recall@k  = |gold sessions covered by the top-k notes| / |gold sessions|
- sess_all@k     = 1 if every gold session is covered by the top-k notes
- turn_recall@k  = |gold turns covered by the top-k notes| / |gold turns|; a retrieved
                   session-parent note (turn_idx -1) covers every turn of its session
- ndcg_sess@10   = binary-relevance nDCG over the session order of first appearance
- ctx_precision  = share of bundle notes that come from a gold session (count-based);
  ctx_precision_tok = same, weighted by cl100k tokens of the passages
- tokens         = bundle.total_tokens (cap violations counted against TOKEN_CAP)
- latency        = p50 / p95 / mean of latency_ms.total and of each stage
- accuracy       = judge.correct; also task-averaged (macro over the 6 types),
  abstention accuracy, and accuracy conditioned on full gold-session coverage
Recall / precision exclude abstention questions (their "gold" is a decoy) and,
for turn recall, questions without turn labels. "k" counts notes in rank order.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mem27.eval.common import TOKEN_CAP, collect_env, n_tokens, write_json
from mem27.vschema import Question, iter_jsonl, read_jsonl, session_of

KS = (5, 10, 20)


def _ndcg(rels: list[int], n_rel: int, k: int) -> float:
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel, k)))
    return dcg / idcg if idcg else 0.0


def question_row(q: Question, b: dict, ans: dict | None, tok_cap: int = TOKEN_CAP) -> dict:
    nids = list(b.get("note_ids", []))
    passages = list(b.get("passages", []))
    gold_s = set(q.gold_session_ids)
    gold_t = set(q.gold_note_ids)
    abst = q.question_id.endswith("_abs")
    lat = b.get("latency_ms") or {}
    row = {
        "question_id": q.question_id,
        "question_type": q.question_type,
        "split": q.split,
        "abstention": abst,
        "plan": b.get("plan"),
        "n_notes": len(nids),
        "total_tokens": b.get("total_tokens", 0),
        "over_cap": int((b.get("total_tokens") or 0) > tok_cap),
        "latency_total": lat.get("total"),
        "stages": lat.get("stages", {}),
        "n_stale": sum(bool(x) for x in b.get("stale_flags", []) or []),
    }
    sess_order: list[str] = []
    for nid in nids:
        s = session_of(nid)
        if s not in sess_order:
            sess_order.append(s)
    scored = not abst and bool(gold_s)
    for k in KS:
        top = nids[:k]
        cov_s = {session_of(n) for n in top}
        parents = {session_of(n) for n in top if n.endswith(":-1")}
        if scored:
            row[f"sess_recall@{k}"] = len(gold_s & cov_s) / len(gold_s)
            row[f"sess_all@{k}"] = float(gold_s <= cov_s)
        if scored and gold_t:
            top_set = set(top)
            hit = [g for g in gold_t if g in top_set or session_of(g) in parents]
            row[f"turn_recall@{k}"] = len(hit) / len(gold_t)
            row[f"turn_all@{k}"] = float(len(hit) == len(gold_t))
    if scored:
        cov_all = {session_of(n) for n in nids}
        row["sess_recall_bundle"] = len(gold_s & cov_all) / len(gold_s)
        row["sess_all_bundle"] = float(gold_s <= cov_all)
        row["ndcg_sess@10"] = _ndcg([int(s in gold_s) for s in sess_order], len(gold_s), 10)
        first = next((i for i, s in enumerate(sess_order) if s in gold_s), None)
        row["mrr_sess"] = 1.0 / (first + 1) if first is not None else 0.0
        if nids:
            rel = [session_of(n) in gold_s for n in nids]
            row["ctx_precision"] = sum(rel) / len(nids)
            toks = [n_tokens(p) for p in passages] if passages else [1] * len(nids)
            row["ctx_precision_tok"] = sum(t for t, r in zip(toks, rel) if r) / max(1, sum(toks))
        else:
            row["ctx_precision"] = row["ctx_precision_tok"] = 0.0
    if ans is not None and ans.get("judge"):
        row["correct"] = float(bool(ans["judge"]["correct"]))
        for name, j in (ans.get("judges") or {}).items():
            row[f"correct[{name}]"] = float(bool(j["correct"]))
        row["answer_tokens"] = (ans.get("reader") or {}).get("completion_tokens")
    return row


def _pct(xs: list[float], p: float) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, p)) if xs else None


def aggregate(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows)}
    if not rows:
        return out
    keys = sorted({k for r in rows for k in r if k.startswith(("sess_", "turn_", "ndcg", "mrr", "ctx_", "correct"))})
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = float(np.mean(vals))
            out[f"n[{k}]"] = len(vals)
    toks = [r["total_tokens"] for r in rows]
    out.update({"tokens_mean": float(np.mean(toks)), "tokens_total": int(np.sum(toks)),
                "tokens_p95": _pct(toks, 95), "over_cap": int(sum(r["over_cap"] for r in rows)),
                "notes_mean": float(np.mean([r["n_notes"] for r in rows]))})
    lat = [r["latency_total"] for r in rows]
    if any(x is not None for x in lat):
        out.update({"latency_p50": _pct(lat, 50), "latency_p95": _pct(lat, 95),
                    "latency_mean": float(np.mean([x for x in lat if x is not None]))})
        stages = defaultdict(list)
        for r in rows:
            for s, v in (r.get("stages") or {}).items():
                stages[s].append(v)
        out["stages"] = {s: {"p50": _pct(v, 50), "p95": _pct(v, 95), "mean": float(np.mean(v))} for s, v in stages.items()}
    if any("correct" in r for r in rows):
        judged = [r for r in rows if "correct" in r]
        out["accuracy"] = float(np.mean([r["correct"] for r in judged]))
        by_t = defaultdict(list)
        for r in judged:
            by_t[r["question_type"]].append(r["correct"])
        out["accuracy_task_avg"] = float(np.mean([np.mean(v) for v in by_t.values()]))
        ab = [r["correct"] for r in judged if r["abstention"]]
        out["accuracy_abstention"] = float(np.mean(ab)) if ab else None
        na = [r["correct"] for r in judged if not r["abstention"]]
        out["accuracy_non_abstention"] = float(np.mean(na)) if na else None
        hit = [r["correct"] for r in judged if r.get("sess_all_bundle") == 1.0]
        miss = [r["correct"] for r in judged if r.get("sess_all_bundle") == 0.0]
        out["acc_given_all_gold"] = float(np.mean(hit)) if hit else None
        out["acc_given_gold_missing"] = float(np.mean(miss)) if miss else None
        out["n_all_gold"], out["n_gold_missing"] = len(hit), len(miss)
    return out


def compute(run_dir: str | Path, data_dir: str | Path, answers_name: str = "answers.jsonl") -> dict:
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    qs = {q.question_id: q for q in read_jsonl(data_dir / "questions.jsonl", Question)}
    answers = {}
    if (run_dir / answers_name).exists():
        answers = {a["question_id"]: a for a in iter_jsonl(run_dir / answers_name)}
    rows = [question_row(qs[b["question_id"]], b, answers.get(b["question_id"]))
            for b in iter_jsonl(run_dir / "bundles.jsonl") if b["question_id"] in qs]
    splits = {"all": rows, "dev": [r for r in rows if r["split"] == "dev"],
              "test": [r for r in rows if r["split"] == "test"]}
    m = {
        "run_id": run_dir.name,
        "dataset": data_dir.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plan": rows[0]["plan"] if rows else None,
        "definitions": __doc__,
        "aggregate": {s: aggregate(r) for s, r in splits.items() if r},
        "per_type": {s: {t: aggregate([x for x in r if x["question_type"] == t])
                         for t in sorted({x["question_type"] for x in r})} for s, r in splits.items() if r},
        "per_question": rows,
    }
    if answers:
        a0 = next(iter(answers.values()))
        m["reader"] = a0.get("reader")
        m["judge_model"] = (a0.get("judge") or {}).get("judge_model")
    env_path = run_dir / "env.json"
    m["env"] = json.loads(env_path.read_text()) if env_path.exists() else collect_env()
    for extra in ("storage.json", "run_config.json"):
        if (run_dir / extra).exists():
            m[extra.split(".")[0]] = json.loads((run_dir / extra).read_text())
    return m


def write_metrics(run_dir: str | Path, data_dir: str | Path, answers_name: str = "answers.jsonl",
                  out_name: str = "metrics.json") -> dict:
    m = compute(run_dir, data_dir, answers_name)
    write_json(Path(run_dir) / out_name, m)
    return m


def summary_line(m: dict, split: str = "test") -> str:
    a = m["aggregate"].get(split, {})
    f = lambda k: (f"{a[k]:.3f}" if isinstance(a.get(k), float) else "-")  # noqa: E731
    return (f"{m['run_id']:<28} {split:<4} n={a.get('n', 0):<4} acc={f('accuracy')} task_avg={f('accuracy_task_avg')} "
            f"sR@5={f('sess_recall@5')} sR@20={f('sess_recall@20')} tR@20={f('turn_recall@20')} "
            f"prec={f('ctx_precision')} tok={a.get('tokens_mean', 0):.0f} p95ms={a.get('latency_p95') or 0:.1f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--answers", default="answers.jsonl")
    ap.add_argument("--out-name", default="metrics.json")
    ap.add_argument("--split", default="test")
    a = ap.parse_args(argv)
    for r in a.run:
        m = write_metrics(r, a.data, a.answers, a.out_name)
        print(summary_line(m, a.split))


if __name__ == "__main__":
    main()
