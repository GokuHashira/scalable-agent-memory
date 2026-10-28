"""Dataset card numbers for a converted LongMemEval dataset, and a question-level diff between two conversions.

  stats: per question type (all / dev / test) n, abstention n, mean sessions per history, mean cl100k tokens per
         history (sum over turn notes, same definition as mem27/data/longmemeval.py), mean gold sessions / turns.
  diff:  question-level comparison of two conversions with the same question ids (e.g. cleaned vs original release):
         ids only in one, differing question / answer / type / date, differing session or turn counts, gold counts,
         tokens per history.

  python scripts/data_card.py stats --data data/lme_s --out runs/_tables/data_card_lme_s
  python scripts/data_card.py diff --a data/lme_s --b data/lme_s_orig --out runs/_tables/data_diff_lme_s_vs_orig
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.vschema import iter_jsonl  # noqa: E402
from scripts.reproduce_table import TYPES, to_md  # noqa: E402


def per_history(data: Path) -> dict[str, dict]:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    h = defaultdict(lambda: {"sessions": 0, "turns": 0, "tokens": 0})
    for n in iter_jsonl(data / "notes.jsonl"):
        d = h[n["history_id"]]
        if n["turn_idx"] < 0:
            d["sessions"] += 1
        else:
            d["turns"] += 1
            d["tokens"] += len(enc.encode(n["text"], disallowed_special=()))
    return dict(h)


def stats(a) -> None:
    data = Path(a.data)
    qs = list(iter_jsonl(data / "questions.jsonl"))
    hist = per_history(data)
    rows = [["question type", "n (dev / test)", "abstention", "mean sessions / history", "mean tokens / history (cl100k)",
             "mean gold sessions", "mean gold turns"]]
    js = {"dataset": data.name, "n_questions": len(qs), "split": {}, "by_type": {}}
    for t in TYPES + ["ALL"]:
        sel = [q for q in qs if t == "ALL" or q["question_type"] == t]
        h = [hist[q["history_id"]] for q in sel]
        nd = sum(q["split"] == "dev" for q in sel)
        rec = {"n": len(sel), "dev": nd, "test": len(sel) - nd,
               "abstention": sum(q["question_id"].endswith("_abs") for q in sel),
               "mean_sessions": st.mean(x["sessions"] for x in h), "mean_turns": st.mean(x["turns"] for x in h),
               "mean_tokens": st.mean(x["tokens"] for x in h), "min_tokens": min(x["tokens"] for x in h),
               "max_tokens": max(x["tokens"] for x in h),
               "mean_gold_sessions": st.mean(len(q["gold_session_ids"]) for q in sel),
               "mean_gold_turns": st.mean(len(q["gold_note_ids"]) for q in sel)}
        js["by_type"][t] = rec
        rows.append([("**all**" if t == "ALL" else t), f"{rec['n']} ({rec['dev']} / {rec['test']})", str(rec["abstention"]),
                     f"{rec['mean_sessions']:.1f}", f"{rec['mean_tokens']:,.0f}", f"{rec['mean_gold_sessions']:.2f}",
                     f"{rec['mean_gold_turns']:.2f}"])
    js["split"] = {s: sum(q["split"] == s for q in qs) for s in ("dev", "test")}
    Path(a.out + ".md").write_text(to_md(rows))
    Path(a.out + ".json").write_text(json.dumps(js, indent=1))
    print(to_md(rows))


def diff(a) -> None:
    A, B = Path(a.a), Path(a.b)
    qa = {q["question_id"]: q for q in iter_jsonl(A / "questions.jsonl")}
    qb = {q["question_id"]: q for q in iter_jsonl(B / "questions.jsonl")}
    ha, hb = per_history(A), per_history(B)
    both = sorted(qa.keys() & qb.keys())
    out = {"a": A.name, "b": B.name, "n_a": len(qa), "n_b": len(qb), "only_a": sorted(qa.keys() - qb.keys()),
           "only_b": sorted(qb.keys() - qa.keys()), "n_both": len(both), "differ": {}}
    fields = {"question": lambda q, h: q["question"], "answer": lambda q, h: str(q["answer"]),
              "question_type": lambda q, h: q["question_type"], "question_date": lambda q, h: q["question_date"],
              "split": lambda q, h: q["split"], "n_gold_sessions": lambda q, h: len(q["gold_session_ids"]),
              "n_gold_turns": lambda q, h: len(q["gold_note_ids"]), "n_sessions": lambda q, h: h["sessions"],
              "n_turns": lambda q, h: h["turns"], "tokens": lambda q, h: h["tokens"]}
    for name, f in fields.items():
        ids = [q for q in both if f(qa[q], ha[qa[q]["history_id"]]) != f(qb[q], hb[qb[q]["history_id"]])]
        out["differ"][name] = {"n": len(ids), "examples": ids[:5]}
    tok_a = sum(ha[qa[q]["history_id"]]["tokens"] for q in both)
    tok_b = sum(hb[qb[q]["history_id"]]["tokens"] for q in both)
    out["total_tokens"] = {"a": tok_a, "b": tok_b, "relative_change": (tok_b - tok_a) / tok_a}
    out["total_turns"] = {"a": sum(ha[qa[q]["history_id"]]["turns"] for q in both),
                          "b": sum(hb[qb[q]["history_id"]]["turns"] for q in both)}
    out["total_sessions"] = {"a": sum(ha[qa[q]["history_id"]]["sessions"] for q in both),
                             "b": sum(hb[qb[q]["history_id"]]["sessions"] for q in both)}
    Path(a.out + ".json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("only_a", "only_b")}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    d = sub.add_parser("diff")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    stats(a) if a.cmd == "stats" else diff(a)


if __name__ == "__main__":
    main()
