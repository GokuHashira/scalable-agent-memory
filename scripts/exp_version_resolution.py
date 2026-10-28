"""E2: can the reader resolve versions when retrieval hands it both the old and the new value?

Synthetic questions whose fact changed (latest_value / exact_lookup / semantic_recall with superseded
versions). Each bundle holds the notes asserting the current value, every superseded note, and 4 notes about
the same entity's other attributes (realistic noise). Conditions (same questions, same notes):
  chrono_dates      notes in time order, each with its date      (what our reader wrapper does)
  shuffled_dates    seeded random order, dates kept
  shuffled_nodates  seeded random order, dates removed           (no recency signal at all)
  stale_only        superseded + noise only, time order, dates   (retrieval missed the update)
  latest_only       current + noise only, time order, dates      (retrieval filtered perfectly)
Prompt: configs/prompts/reader_con_neutral.txt, the pinned chain-of-note prompt without the claim that
the notes are chronological and dated, identical for all conditions. Values are invented, so scoring is exact:
correct = the current value is in the final answer; stale_served = a superseded value is and the current one
is not; abstained = "not available"-style answer. Split by how the latest change happened (explicit vs
implicit, supersede vs correct).

  python scripts/exp_version_resolution.py --data data/synth_10k --endpoint runs/endpoints/reader
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.bootstrap import paired_bootstrap  # noqa: E402
from mem27.eval.common import fill, fmt_date, write_json  # noqa: E402
from mem27.eval.freshness import contains  # noqa: E402
from mem27.eval.llm import ChatLLM  # noqa: E402
from mem27.eval.reader import QWEN3_NO_THINK, extract_final  # noqa: E402
from mem27.vschema import iter_jsonl, write_jsonl  # noqa: E402

CONDS = ["chrono_dates", "shuffled_dates", "shuffled_nodates", "stale_only", "latest_only"]
ABSTAIN = re.compile(r"not (?:available|mentioned|provided|specified|stated)|no information|cannot (?:be )?determine", re.I)


def render(notes: list[dict], dates: bool) -> str:
    out = []
    for k, n in enumerate(notes):
        head = f"### Note {k + 1}" + (f" (date: {fmt_date(n['ts'])})" if dates else "")
        out.append(f"{head}\nuser: {n['text']}")
    return "\n\n".join(out)


def ci(x: list[float], seed: int = 0) -> tuple[float, float]:
    a = np.asarray(x, float)
    if not len(a):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = a[rng.integers(0, len(a), size=(10_000, len(a)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth_10k")
    ap.add_argument("--endpoint", default="runs/endpoints/reader")
    ap.add_argument("--prompt", default="configs/prompts/reader_con_neutral.txt")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    data = Path(a.data)
    notes = {n["note_id"]: n for n in iter_jsonl(data / "notes_final.jsonl")}
    facts = list(iter_jsonl(data.parent / "_audit" / data.name / "note_facts.jsonl"))
    by_entity = defaultdict(list)
    for f in facts:
        if "entity" in f and "attr" in f:
            by_entity[f["entity"]].append(f)
    qs = [q for q in iter_jsonl(data / "questions.jsonl")
          if q["question_type"] in ("latest_value", "exact_lookup", "semantic_recall")
          and q["fact"].get("stale_values") and q["fact"].get("superseded_note_ids")]
    tmpl = Path(a.prompt).read_text()
    rng = random.Random(a.seed)
    items = []  # (cond, q, prompt)
    for q in qs:
        f = q["fact"]
        latest = [notes[i] for i in q["gold_note_ids"] if i in notes]
        stale = [notes[i] for i in f["superseded_note_ids"] if i in notes]
        others = [x for x in by_entity[f["entity"]] if x["attr"] != f["attr"] and x["note_id"] in notes
                  and notes[x["note_id"]]["ts"] <= q["question_date"]]
        noise = [notes[x["note_id"]] for x in rng.sample(others, min(4, len(others)))]
        full = latest + stale + noise
        chrono = sorted(full, key=lambda n: n["ts"])
        shuf = full[:]
        random.Random(f"{a.seed}-{q['question_id']}").shuffle(shuf)
        hist = {
            "chrono_dates": render(chrono, True),
            "shuffled_dates": render(shuf, True),
            "shuffled_nodates": render(shuf, False),
            "stale_only": render(sorted(stale + noise, key=lambda n: n["ts"]), True),
            "latest_only": render(sorted(latest + noise, key=lambda n: n["ts"]), True),
        }
        for c in CONDS:
            p = fill(tmpl, history=hist[c], question_date=fmt_date(q["question_date"]), question=q["question"])
            items.append((c, q, p))
    out_dir = Path(a.runs) / f"xver_{data.name}"
    llm = ChatLLM(a.endpoint, max_workers=a.workers, extra_body=QWEN3_NO_THINK,
                  cache=out_dir / "reader_cache.sqlite")
    outs = llm.map([[{"role": "user", "content": p}] for _, _, p in items], max_tokens=1024,
                   progress=lambda d, n: print(f"[xver] {d}/{n}", flush=True))
    rows = []
    for (c, q, _), o in zip(items, outs):
        ans = extract_final(o.get("text", ""))
        f = q["fact"]
        cur = contains(ans, q["answer"])
        st = any(contains(ans, v) for v in f["stale_values"]) and not cur
        kind = f.get("latest_change_kind")
        how = f"{kind}:{'explicit' if f.get('latest_change_explicit') else 'implicit'}" if kind else "restated"
        rows.append({"cond": c, "question_id": q["question_id"], "question_type": q["question_type"], "how": how,
                     "n_versions": f.get("n_versions"), "answer": ans, "error": o.get("error"),
                     "correct": float(cur), "stale_served": float(st), "abstained": float(bool(ABSTAIN.search(ans)))})
    write_jsonl(out_dir / "answers.jsonl", rows)
    res = {"data": str(data), "n_questions": len(qs), "prompt": a.prompt, "model": llm.model,
           "errors": sum(1 for r in rows if r["error"]), "conditions": {}, "paired": {}}
    by = defaultdict(list)
    for r in rows:
        by[(r["cond"], "ALL")].append(r)
        by[(r["cond"], r["how"])].append(r)
    for (c, g), rs in sorted(by.items()):
        d = res["conditions"].setdefault(c, {})
        d[g] = {"n": len(rs)}
        for m in ("correct", "stale_served", "abstained"):
            v = [r[m] for r in rs]
            d[g][m] = float(np.mean(v))
            d[g][f"{m}_ci"] = ci(v)
    per = {(r["cond"], r["question_id"]): r for r in rows}
    qids = [q["question_id"] for q in qs]
    for c in CONDS:
        if c == "chrono_dates":
            continue
        for m in ("correct", "stale_served"):
            x = np.array([per[(c, i)][m] for i in qids])
            y = np.array([per[("chrono_dates", i)][m] for i in qids])
            res["paired"][f"{c} - chrono_dates : {m}"] = paired_bootstrap(x, y)
    write_json(out_dir / "results.json", res)
    print(f"model={llm.model} n_questions={len(qs)} errors={res['errors']}")
    for c in CONDS:
        al = res["conditions"][c]["ALL"]
        print(f"{c:18s} correct={al['correct']:.3f} {tuple(round(x, 3) for x in al['correct_ci'])} "
              f"stale_served={al['stale_served']:.3f} abstained={al['abstained']:.3f}")
        for g, d in sorted(res["conditions"][c].items()):
            if g != "ALL":
                print(f"   {g:22s} n={d['n']:3d} correct={d['correct']:.3f} stale={d['stale_served']:.3f}")


if __name__ == "__main__":
    main()
