"""LongMemEval-V2 Small, text-only: BM25 memory over trajectory notes + official prompts + official scorer.

Memory: per-domain BM25 (rank_bm25 Okapi, same tokenizer as V's reference retrievers) over the notes from
scripts/lme_v2_prepare.py; the top notes are packed into the memory context up to a token budget (cl100k).
Budget 0 = the official no-retrieval baseline. Reader: Qwen3-8B, non-thinking, greedy, with the official
domain system prompt and the official message layout ("### Memory context:" ... "### Question to answer:"),
both taken verbatim from raw/LongMemEval-V2/evaluation/harness.py (parsed with ast, not retyped). Scoring:
the official evaluation/qa_eval_metrics.py (eval_from_spec), with the official call convention (boxed answer
for deterministic matchers, full response for the LLM checkers, UNKNOWN = wrong). LLM checkers use the open
gpt-oss-20b judge (reasoning effort low) instead of the paper's API judge. memory_query_avg_seconds = BM25
query + packing time per question (index build time reported separately).

  python scripts/lme_v2_eval.py --data data/lme_v2_small --budgets 0 2048 8192 24576 \
      --reader-endpoint runs/vllm_endpoint.txt --judge-endpoint runs/vllm_judge_endpoint.txt
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]

from mem27.eval.common import n_tokens, write_json  # noqa: E402
from mem27.eval.llm import ChatLLM, resolve_endpoint  # noqa: E402
from mem27.eval.packing import pack  # noqa: E402
from mem27.eval.reader import QWEN3_NO_THINK  # noqa: E402
from mem27.vschema import iter_jsonl, write_jsonl  # noqa: E402
from scripts.vref_retrieve import tok  # noqa: E402

LLM_EVALS = {"llm_abstention_checker", "llm_gotchas_checker"}


def official_system_prompts(repo: Path) -> dict[str, str]:
    tree = ast.parse((repo / "evaluation" / "harness.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "DOMAIN_SYSTEM_PROMPTS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("DOMAIN_SYSTEM_PROMPTS not found")


def ci(x) -> tuple[float, float]:
    a = np.asarray(x, float)
    m = a[np.random.default_rng(0).integers(0, len(a), size=(10_000, len(a)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(argv=None):
    from rank_bm25 import BM25Okapi

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lme_v2_small")
    ap.add_argument("--repo", default="raw/LongMemEval-V2")
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 2048, 8192, 24576])
    ap.add_argument("--reader-endpoint", default="runs/vllm_endpoint.txt")
    ap.add_argument("--judge-endpoint", default="runs/vllm_judge_endpoint.txt")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--top", type=int, default=400)
    ap.add_argument("--rankings", default="runs/lme_v2_small_bm25_rankings.json",
                    help="BM25 rankings cache: written by --retrieve-only (CPU job), reused by the reading run (GPU job)")
    ap.add_argument("--retrieve-only", action="store_true")
    a = ap.parse_args(argv)
    data, repo = Path(a.data), Path(a.repo)
    sys.path.insert(0, str(repo / "evaluation"))
    import qa_eval_metrics as qem  # official scorer

    prompts = official_system_prompts(repo)
    qs = list(iter_jsonl(data / "questions.jsonl"))
    raw_q = {json.loads(l)["id"]: json.loads(l) for l in open(Path("raw/lme_v2") / "questions.jsonl")}

    # ---- retrieval (CPU)
    notes = defaultdict(list)
    for n in iter_jsonl(data / "notes.jsonl"):
        notes[n["history_id"]].append((n["note_id"], n["text"]))
    ranked, qsec, build_s = {}, {}, {}
    rk = Path(a.rankings)
    if rk.exists() and not a.retrieve_only:
        saved = json.loads(rk.read_text())
        ranked, qsec, build_s = saved["ranked"], saved["query_seconds"], saved["index_build_seconds"]
        print(f"[bm25] loaded rankings for {len(ranked)} questions from {rk}", flush=True)
    for dom, rows in ([] if ranked else notes.items()):
        t0 = time.perf_counter()
        bm = BM25Okapi([tok(t) or ["_"] for _, t in rows])
        build_s[dom] = time.perf_counter() - t0
        for q in qs:
            if q["history_id"] != dom:
                continue
            t0 = time.perf_counter()
            s = bm.get_scores(tok(q["question"]) or ["_"])
            order = np.argsort(-s, kind="stable")[: a.top]
            ranked[q["question_id"]] = ([rows[i][0] for i in order], [float(s[i]) for i in order])
            qsec[q["question_id"]] = time.perf_counter() - t0
        print(f"[bm25] {dom}: {len(rows)} notes, build {build_s[dom]:.1f}s", flush=True)
    if not rk.exists() or a.retrieve_only:
        write_json(rk, {"ranked": ranked, "query_seconds": qsec, "index_build_seconds": build_s})
    if a.retrieve_only:
        print(f"wrote {rk}; mean query {np.mean(list(qsec.values())):.3f}s", flush=True)
        return
    text = {nid: t for rows in notes.values() for nid, t in rows}

    reader = ChatLLM(a.reader_endpoint, max_workers=a.workers, extra_body=QWEN3_NO_THINK,
                     cache=Path("runs/_cache/lme_v2_reader.sqlite"))
    judge_url = resolve_endpoint(a.judge_endpoint)
    judge_model = ChatLLM(a.judge_endpoint).model
    summary = {"reader": reader.model, "judge": judge_model, "index_build_seconds": build_s,
               "n_questions": len(qs), "budgets": {}}
    for B in a.budgets:
        run = Path("runs") / f"lme_v2_small_bm25_b{B}"
        items, ctx_tok, pack_s = [], [], []
        for q in qs:
            t0 = time.perf_counter()
            if B:
                ids, sc = ranked[q["question_id"]]
                b = pack(ids, text, sc, token_cap=B, max_results=None)
                mem = "\n\n".join(b["passages"])
                ctx_tok.append(b["total_tokens"])
            else:
                b, mem = {"note_ids": [], "passages": [], "scores": [], "total_tokens": 0}, "(empty)"
                ctx_tok.append(0)
            pack_s.append(time.perf_counter() - t0 + (qsec[q["question_id"]] if B else 0.0))
            user = "### Memory context:\n" + mem + f"\n\n### Question to answer:\n{q['question']}"
            items.append((q, b, [{"role": "system", "content": prompts[q["domain"]]}, {"role": "user", "content": user}]))
        outs = reader.map([m for _, _, m in items], max_tokens=1024,
                          progress=lambda d, n: print(f"[read b{B}] {d}/{n}", flush=True))
        rows = []
        for (q, b, _), o in zip(items, outs):
            raw = o.get("text", "") or ""
            parsed = qem.extract_boxed_answer(raw)
            rows.append({"question_id": q["question_id"], "domain": q["domain"], "question_type": q["question_type"],
                         "eval_function": q["eval_function"], "eval_name": qem.eval_name(q["eval_function"]),
                         "answer_gold": q["answer"], "response_raw": raw, "response_parsed_boxed": parsed,
                         "is_unknown": qem.is_unknown(parsed), "context_tokens": b["total_tokens"],
                         "note_ids": b["note_ids"], "reader_error": o.get("error")})

        def score(r):
            kw = {}
            if r["eval_name"] in LLM_EVALS:
                kw = {"evaluator_model": judge_model, "evaluator_base_url": judge_url, "evaluator_api_key": "EMPTY",
                      "evaluator_reasoning_effort": "low", "evaluator_max_completion_tokens": 2048,
                      "evaluator_timeout_seconds": 600.0, "question_item": raw_q[r["question_id"]],
                      "parsed_prediction": r["response_parsed_boxed"], "model_response": r["response_raw"]}
            pred = r["response_raw"] if r["eval_name"] in LLM_EVALS else r["response_parsed_boxed"]
            try:
                ok = qem.score_to_bool(qem.eval_from_spec(r["eval_function"], pred, r["answer_gold"], **kw))
                err = None
            except Exception as e:  # noqa: BLE001
                ok, err = False, repr(e)[:200]
            return (False if r["is_unknown"] else ok), err

        with ThreadPoolExecutor(32) as ex:
            res = list(ex.map(score, rows))
        for r, (ok, err) in zip(rows, res):
            r["correct"], r["score_error"] = float(ok), err
        write_jsonl(run / "answers.jsonl", rows)
        acc = [r["correct"] for r in rows]
        by_t, by_d = defaultdict(list), defaultdict(list)
        for r in rows:
            by_t[r["question_type"]].append(r["correct"])
            by_d[r["domain"]].append(r["correct"])
        s = {"accuracy": float(np.mean(acc)), "accuracy_ci": ci(acc), "n": len(rows),
             "unknown_rate": float(np.mean([r["is_unknown"] for r in rows])),
             "reader_errors": sum(1 for r in rows if r["reader_error"]),
             "score_errors": sum(1 for r in rows if r["score_error"]),
             "context_tokens_mean": float(np.mean(ctx_tok)),
             "memory_query_avg_seconds": float(np.mean(pack_s)),
             "by_type": {t: {"n": len(v), "acc": float(np.mean(v))} for t, v in sorted(by_t.items())},
             "by_domain": {d: {"n": len(v), "acc": float(np.mean(v))} for d, v in sorted(by_d.items())}}
        summary["budgets"][str(B)] = s
        write_json(run / "results.json", s)
        print(f"b{B}: acc={s['accuracy']:.3f} {tuple(round(x, 3) for x in s['accuracy_ci'])} unknown={s['unknown_rate']:.3f} "
              f"ctx={s['context_tokens_mean']:.0f} q_s={s['memory_query_avg_seconds']:.3f} errors r/s={s['reader_errors']}/{s['score_errors']}",
              flush=True)
    write_json(Path("runs/_tables/lme_v2_small_text.json"), summary)


if __name__ == "__main__":
    main()
