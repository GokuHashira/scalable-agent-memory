"""LongMemEval-V2 Small, text-only (DATA-05 stretch): trajectories -> notes, questions -> contract format.

Source: HF xiaowu0162/longmemeval-v2 (Apache-2.0) in raw/lme_v2/: questions.jsonl, trajectories.jsonl,
haystacks/lme_v2_small.json. Text-only = the 29 questions with a screenshot are dropped (422 remain);
trajectory screenshots are never used.

In the Small tier every question of a domain shares one 100-trajectory haystack, so history_id = domain.
Notes per trajectory state (session_id = trajectory id):
  step note   turn_idx = 1000*state_index      goal / outcome / url / action / agent thought (no tree)
  tree chunk  turn_idx = 1000*state_index + c  the accessibility tree split into <= CHUNK cl100k tokens,
                                               each prefixed with the same goal / step / url header
Questions keep the official `eval_function`, `domain`, `question_type` and `answer` fields (extra keys).

  python scripts/lme_v2_prepare.py --src raw/lme_v2 --out data/lme_v2_small
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import encoder, write_json  # noqa: E402

CHUNK = 512


def header(t: dict, s: dict) -> str:
    return (f"[trajectory {t['id']} | goal: {t.get('goal') or ''} | outcome: {t.get('outcome')} | "
            f"step {s.get('state_index')} | url: {s.get('url') or ''}]")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw/lme_v2")
    ap.add_argument("--out", default="data/lme_v2_small")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    a = ap.parse_args(argv)
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    enc = encoder()
    hay = json.loads((src / "haystacks" / "lme_v2_small.json").read_text())
    qs_all = [json.loads(l) for l in open(src / "questions.jsonl")]
    qs = [q for q in qs_all if not q.get("image")]
    dom_hay: dict[str, list[str]] = {}
    for q in qs_all:
        h = hay[q["id"]]
        if q["domain"] in dom_hay:
            assert dom_hay[q["domain"]] == h, "Small haystack differs within a domain"
        dom_hay[q["domain"]] = h
    need = {tid: d for d, ids in dom_hay.items() for tid in ids}
    n_notes, n_states, toks, kinds = 0, 0, Counter(), Counter()
    with open(out / "notes.jsonl", "w") as fn:
        for line in open(src / "trajectories.jsonl"):
            t = json.loads(line)
            d = need.get(t["id"])
            if d is None:
                continue
            for s in t["states"]:
                n_states += 1
                hd = header(t, s)
                si = int(s.get("state_index") or 0)
                base = {"history_id": d, "session_id": t["id"], "role": "agent", "ts": "", "valid_to": None,
                        "version": 1, "supersedes_note_id": None, "entity_tags": [], "summary_id": None, "links": []}
                step_text = (f"{hd}\naction: {s.get('action') or '(initial state)'}\n"
                             f"thought: {s.get('thought') or ''}")
                rows = [(1000 * si, step_text, "step")]
                ids = enc.encode(s.get("accessibility_tree") or "", disallowed_special=())
                for c in range(0, len(ids), a.chunk):
                    rows.append((1000 * si + 1 + c // a.chunk, f"{hd}\naccessibility tree (part {c // a.chunk + 1}):\n"
                                 + enc.decode(ids[c:c + a.chunk]), "tree"))
                for turn, text, kind in rows:
                    nid = f"{d}:{t['id']}:{turn}"
                    fn.write(json.dumps({"note_id": nid, "turn_idx": turn, "text": text,
                                         "metadata": {"kind": kind, "state_index": si, "outcome": t.get("outcome")},
                                         **base}) + "\n")
                    n_notes += 1
                    kinds[kind] += 1
                    toks[d] += len(enc.encode(text, disallowed_special=()))
    with open(out / "questions.jsonl", "w") as fq:
        for q in qs:
            fq.write(json.dumps({"question_id": q["id"], "history_id": q["domain"], "question_type": q["question_type"],
                                 "question": q["question"], "question_date": "", "answer": str(q["answer"]),
                                 "gold_session_ids": [], "gold_note_ids": [], "split": "test",
                                 "domain": q["domain"], "environment": q.get("environment"),
                                 "eval_function": q["eval_function"]}) + "\n")
    stats = {"source": "HF xiaowu0162/longmemeval-v2 (Small tier, text-only)", "questions_total": len(qs_all),
             "questions_text_only": len(qs), "dropped_image_questions": len(qs_all) - len(qs),
             "types": dict(Counter(q["question_type"] for q in qs)), "domains": dict(Counter(q["domain"] for q in qs)),
             "trajectories": len(need), "states": n_states, "notes": n_notes, "note_kinds": dict(kinds),
             "tokens_per_domain_haystack": dict(toks), "chunk_tokens": a.chunk}
    write_json(out / "stats.json", stats)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
