"""Parallel Graphiti run over the full 100-question LongMemEval_S dev subset.

This scales the serial feasibility runner (``graphiti/run_graphiti.py``) to all
100 dev-split questions by running ``--concurrency`` questions at once in a
process pool. Different histories are independent, each worker builds its own
in-memory Kuzu graph (``group_id = history_id``), and all workers hit the same
local vLLM endpoint (:8000) -- vLLM batches the concurrent extraction calls, so
wall-clock drops roughly linearly in the worker count.

Per question we do exactly what the serial runner does:
  1. select a bounded set of episodes (gold session turns + chronological
     fillers up to ``--max-episodes``; keeps every gold turn),
  2. ingest them as Graphiti episodes (entity/edge extraction + dedup LLM calls),
  3. combined edge+node+episode hybrid search with the question,
  4. local reader answers from retrieved facts/nodes,
  5. local judge grades the answer vs gold.

Outputs (new files only):
  * runs/analysis/graphiti_answers_100.jsonl  -- shared Answer schema, one row/q,
    checkpointed as each question completes.
  * runs/analysis/graphiti_100_detail.jsonl   -- full per-question record,
    checkpointed as each question completes (source for aggregates).
  * runs/analysis/graphiti_100.json           -- aggregates over the 100.

Usage:
    .venv_graphiti/bin/python graphiti/run_graphiti_100.py \
        --split dev --max-episodes 10 --num-results 10 \
        --concurrency 8 --base-url http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("EMBEDDING_DIM", "768")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path("/fs/gamma-projects/audio/kajayaku/agents")
DATA = ROOT / "data" / "lme_s_k"
ANS_OUT = ROOT / "runs" / "analysis" / "graphiti_answers_100.jsonl"
DETAIL_OUT = ROOT / "runs" / "analysis" / "graphiti_100_detail.jsonl"
AGG_OUT = ROOT / "runs" / "analysis" / "graphiti_100.json"

LME_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]

logging.basicConfig(level=logging.ERROR)
for noisy in ("graphiti_core", "httpx", "kuzu", "sentence_transformers", "openai", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Worker globals (one embedder/reranker per worker process, reused across the
# questions that worker handles; cheap to build an LLM client per question).
# ---------------------------------------------------------------------------
_EMBEDDER = None
_RERANKER = None
_HCLIENT = None
_CFG: dict = {}


def _worker_init(cfg: dict, counter, lock) -> None:
    import sys
    sys.path.insert(0, str(ROOT / "graphiti"))
    import httpx
    from local_clients import LocalSTEmbedder, LocalCosineReranker, SentenceTransformer, EMBED_MODEL, EMBED_DIM

    global _EMBEDDER, _RERANKER, _HCLIENT, _CFG
    _CFG = cfg

    # Assign this worker a free GPU (physical 2 or 3, visible as cuda:2/cuda:3
    # since CUDA_VISIBLE_DEVICES covers 0-3). Round-robin across workers.
    with lock:
        idx = counter.value
        counter.value += 1
    gpus = cfg.get("embed_gpus", [2, 3])
    device = f"cuda:{gpus[idx % len(gpus)]}" if gpus else "cpu"

    class PinnedEmbedder(LocalSTEmbedder):
        def __init__(self, dev):
            self.model = SentenceTransformer(EMBED_MODEL, device=dev)
            self.dim = EMBED_DIM

    try:
        _EMBEDDER = PinnedEmbedder(device)
    except Exception:  # noqa: BLE001 -- fall back to CPU if the GPU is unusable
        _EMBEDDER = PinnedEmbedder("cpu")
    _RERANKER = LocalCosineReranker(_EMBEDDER)
    _HCLIENT = httpx.Client()


def _run_one(payload: tuple) -> dict:
    """Pool entry point: run a single question, return its full record."""
    q, turns = payload
    import sys
    sys.path.insert(0, str(ROOT / "graphiti"))
    from run_graphiti import run_question
    from types import SimpleNamespace

    args = SimpleNamespace(**_CFG)
    try:
        rec = asyncio.run(_run_question_pooled(q, turns, args))
    except Exception as e:  # noqa: BLE001 -- one bad question must not kill the run
        rec = {
            "question_id": q["question_id"],
            "history_id": q["history_id"],
            "question_type": q.get("question_type"),
            "fatal_error": f"{type(e).__name__}: {e}",
        }
    return rec


def select_episodes_full_gold(turns: list, gold_sessions: set, budget: int) -> list:
    """Include EVERY turn of the gold session(s), then pad with the earliest
    non-gold turns up to ``budget`` total (never dropping a gold turn).

    The feasibility runner's flat cap (gold turns first, truncate at cap) drops
    the gold turn whenever it sits past position ``cap`` inside a long gold
    session -- on the dev split that truncation pushed gold_in_selection to 0.33.
    Guaranteeing the whole gold session keeps gold_in_selection ~= 1.0 (the only
    misses are questions whose gold note is absent from notes.jsonl), which is
    the precondition for retrieval to be able to hit. Episodes are returned in
    chronological (session_idx, turn_idx) order.
    """
    def order_key(r):
        return (r.get("metadata", {}).get("session_idx", 0), r["turn_idx"])

    gold = sorted((r for r in turns if r["session_id"] in gold_sessions), key=order_key)
    other = sorted((r for r in turns if r["session_id"] not in gold_sessions), key=order_key)
    n_fill = max(0, budget - len(gold))
    chosen = gold + other[:n_fill]
    return sorted(chosen, key=order_key)


async def _run_question_pooled(q: dict, turns: list, args) -> dict:
    """Mirror of run_graphiti.run_question but reuses the worker's embedder and
    builds a fresh LLM client + in-memory Kuzu graph per question."""
    import time as _time
    from run_graphiti import (
        parse_ts, chat,
        ensure_fts_loaded, create_fts_indices, rebuild_fts_indices,
    )
    from local_clients import make_llm_client
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.nodes import EpisodeType
    from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

    hid = q["history_id"]
    gold_sessions = set(q["gold_session_ids"])
    gold_answer = q["answer"]
    eps = select_episodes_full_gold(turns, gold_sessions, args.max_episodes)
    gold_in_selection = any(r["note_id"] in set(q["gold_note_ids"]) for r in eps)

    llm = make_llm_client(args.base_url, args.api_key, args.model, args.max_tokens)
    driver = KuzuDriver(db=":memory:")
    driver._database = None  # 0.31.2 Kuzu work-around (see run_graphiti.py)
    graphiti = Graphiti(llm_client=llm, embedder=_EMBEDDER,
                        cross_encoder=_RERANKER, graph_driver=driver)

    rec: dict = {
        "question_id": q["question_id"],
        "history_id": hid,
        "question_type": q["question_type"],
        "n_turns_available": len(turns),
        "n_sessions_available": len({r["session_id"] for r in turns}),
        "n_episodes_selected": len(eps),
        "gold_in_selection": gold_in_selection,
    }
    try:
        await graphiti.build_indices_and_constraints()
        await ensure_fts_loaded(driver)
        await create_fts_indices(driver)
        llm.n_calls = 0
        n_ok = 0
        t0 = _time.time()
        for r in eps:
            body = f"{r['role']}: {r['text']}"
            if len(body) > args.text_cap:
                body = body[: args.text_cap] + " ..."
            try:
                await graphiti.add_episode(
                    name=f"{hid}:{r['session_id']}:{r['turn_idx']}",
                    episode_body=body,
                    source_description="LongMemEval conversation turn",
                    reference_time=parse_ts(r["ts"]),
                    source=EpisodeType.message,
                    group_id=hid,
                )
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                rec.setdefault("ingest_errors", []).append(
                    f"{r['turn_idx']}: {type(e).__name__}: {e}")
        rec["ingest_seconds"] = round(_time.time() - t0, 2)
        rec["n_episodes_ingested"] = n_ok
        rec["n_llm_calls_ingest"] = llm.n_calls
        rec["ingest_sec_per_episode"] = round(rec["ingest_seconds"] / max(n_ok, 1), 2)

        # ---- search ----
        await rebuild_fts_indices(driver)
        t1 = _time.time()
        try:
            config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
            config.limit = args.num_results
            sr = await graphiti._search(q["question"], config=config, group_ids=[hid])
            facts = [e.fact for e in sr.edges]
            node_summaries = [f"{n.name}: {n.summary}".strip() for n in sr.nodes]
            episode_texts = [ep.content for ep in sr.episodes]
            rec["search_error"] = None
        except Exception as e:  # noqa: BLE001
            facts, node_summaries, episode_texts = [], [], []
            rec["search_error"] = f"{type(e).__name__}: {e}"
        rec["search_seconds"] = round(_time.time() - t1, 2)
        rec["n_facts"] = len(facts)
        rec["n_nodes"] = len(node_summaries)
        rec["n_episodes_retrieved"] = len(episode_texts)
        rec["facts_sample"] = facts[:8]
        rec["nodes_sample"] = node_summaries[:8]
        ans_l = str(gold_answer).lower()
        retrieved_all = facts + node_summaries + episode_texts
        rec["search_hit"] = any(ans_l in str(t).lower() for t in retrieved_all)

        # ---- reader ----
        ctx_parts = []
        if facts:
            ctx_parts.append("Facts:\n" + "\n".join(f"- {f}" for f in facts))
        if node_summaries:
            ctx_parts.append("Entities:\n" + "\n".join(f"- {s}" for s in node_summaries))
        ctx = "\n\n".join(ctx_parts) or "(no memory retrieved)"
        reader_prompt = (
            "You are answering a question about a user's past conversations using "
            "the retrieved memory below.\n\n" + ctx + "\n\nQuestion: " + q["question"]
            + "\n\nAnswer concisely with just the answer. If the memory does not "
            "contain the answer, say you don't know."
        )
        pred = chat(_HCLIENT, args.base_url, args.model, reader_prompt, max_tokens=128)
        rec["predicted_answer"] = pred

        # ---- judge ----
        judge_prompt = (
            "You are grading an answer. The gold answer is the reference.\n"
            f"Question: {q['question']}\nGold answer: {gold_answer}\n"
            f"Model answer: {pred}\n\n"
            "Does the model answer convey the same information as the gold answer? "
            "Reply with exactly one word: CORRECT or WRONG."
        )
        verdict = chat(_HCLIENT, args.base_url, args.model, judge_prompt, max_tokens=5)
        rec["judge_raw"] = verdict
        rec["qa_correct"] = verdict.strip().upper().startswith("CORRECT")
    finally:
        try:
            await graphiti.close()
        except Exception:  # noqa: BLE001
            pass
    return rec


def to_answer_row(rec: dict, model: str) -> dict:
    return {
        "question_id": rec["question_id"],
        "plan": "graphiti",
        "answer": rec.get("predicted_answer", ""),
        "judge": {
            "correct": int(bool(rec.get("qa_correct"))),
            "judge_model": model,
            "raw": rec.get("judge_raw", rec.get("fatal_error", "")),
        },
    }


def aggregate(results: list[dict], wall_clock: float, cfg: dict) -> dict:
    ok = [r for r in results if r.get("n_episodes_ingested")]
    done = [r for r in results if not r.get("fatal_error")]

    def rate(pred, pool):
        return round(sum(bool(pred(r)) for r in pool) / len(pool), 4) if pool else None

    agg = {
        "n_questions": len(results),
        "n_completed": len(done),
        "n_fatal_errors": len(results) - len(done),
        "n_questions_ingested": len(ok),
        "wall_clock_seconds": round(wall_clock, 1),
        "wall_clock_hours": round(wall_clock / 3600, 2),
        "concurrency": cfg.get("concurrency"),
        "max_episodes": cfg.get("max_episodes"),
        "num_results": cfg.get("num_results"),
    }
    if ok:
        tot_eps = sum(r["n_episodes_ingested"] for r in ok)
        tot_calls = sum(r.get("n_llm_calls_ingest", 0) for r in ok)
        tot_ing = sum(r["ingest_seconds"] for r in ok)
        agg.update({
            "total_episodes": tot_eps,
            "total_ingest_llm_calls": tot_calls,
            "mean_llm_calls_per_episode": round(tot_calls / tot_eps, 2),
            "mean_ingest_sec_per_episode": round(tot_ing / tot_eps, 2),
            "mean_ingest_seconds_per_question": round(tot_ing / len(ok), 1),
            "search_hit_rate": rate(lambda r: r.get("search_hit"), ok),
            "qa_accuracy": rate(lambda r: r.get("qa_correct"), done),
            "qa_accuracy_over_ingested": rate(lambda r: r.get("qa_correct"), ok),
            "gold_in_selection_rate": rate(lambda r: r.get("gold_in_selection"), ok),
            "mean_facts_retrieved": round(sum(r.get("n_facts", 0) for r in ok) / len(ok), 2),
            "mean_nodes_retrieved": round(sum(r.get("n_nodes", 0) for r in ok) / len(ok), 2),
        })
    # Per-type breakdown over the 6 LongMemEval types.
    by_type = {}
    for t in LME_TYPES:
        sub = [r for r in done if r.get("question_type") == t]
        if not sub:
            continue
        by_type[t] = {
            "n": len(sub),
            "qa_accuracy": rate(lambda r: r.get("qa_correct"), sub),
            "search_hit_rate": rate(lambda r: r.get("search_hit"), sub),
            "gold_in_selection_rate": rate(lambda r: r.get("gold_in_selection"), sub),
        }
    agg["per_type"] = by_type
    return agg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="dev")
    p.add_argument("--n", type=int, default=0, help="0 = all questions in split")
    p.add_argument("--max-episodes", type=int, default=10)
    p.add_argument("--num-results", type=int, default=10)
    p.add_argument("--text-cap", type=int, default=1200)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="sk-local")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--embed-gpus", default="2,3")
    p.add_argument("--resume", action="store_true",
                   help="skip question_ids already present in the detail jsonl")
    p.add_argument("--tag", default="100",
                   help="output file tag; outputs are graphiti_{answers_,,_detail}{tag}. "
                        "Use a distinct tag (e.g. test100) to avoid clobbering the dev run.")
    p.add_argument("--qids-file", default=None,
                   help="JSON list of question_ids to run (overrides --split/--n), e.g. a "
                        "stratified type-representative sample.")
    args = p.parse_args()

    global ANS_OUT, DETAIL_OUT, AGG_OUT
    if args.tag != "100":
        base = ROOT / "runs" / "analysis"
        ANS_OUT = base / f"graphiti_answers_{args.tag}.jsonl"
        DETAIL_OUT = base / f"graphiti_{args.tag}_detail.jsonl"
        AGG_OUT = base / f"graphiti_{args.tag}.json"

    cfg = {
        "split": args.split, "max_episodes": args.max_episodes,
        "num_results": args.num_results, "text_cap": args.text_cap,
        "concurrency": args.concurrency, "base_url": args.base_url,
        "api_key": args.api_key, "model": args.model, "max_tokens": args.max_tokens,
        "embed_gpus": [int(x) for x in args.embed_gpus.split(",") if x.strip() != ""],
    }

    questions = [json.loads(l) for l in (DATA / "questions.jsonl").open()]
    if args.qids_file:
        import json as _json
        keep = set(_json.load(open(args.qids_file)))
        questions = [q for q in questions if q["question_id"] in keep]
    else:
        questions = [q for q in questions if q["split"] == args.split]
        if args.n:
            questions = questions[: args.n]

    turns_by_hid = defaultdict(list)
    wanted = {q["history_id"] for q in questions}
    for line in (DATA / "notes.jsonl").open():
        r = json.loads(line)
        if r["history_id"] in wanted:
            turns_by_hid[r["history_id"]].append(r)

    done_ids = set()
    if args.resume and DETAIL_OUT.exists():
        for line in DETAIL_OUT.open():
            try:
                done_ids.add(json.loads(line)["question_id"])
            except Exception:  # noqa: BLE001
                pass
        print(f"resume: {len(done_ids)} questions already done, skipping them")

    pending = [q for q in questions if q["question_id"] not in done_ids]
    payloads = [(q, turns_by_hid[q["history_id"]]) for q in pending]

    ANS_OUT.parent.mkdir(parents=True, exist_ok=True)
    ans_mode = "a" if (args.resume and ANS_OUT.exists()) else "w"
    detail_mode = "a" if (args.resume and DETAIL_OUT.exists()) else "w"
    ans_f = ANS_OUT.open(ans_mode)
    detail_f = DETAIL_OUT.open(detail_mode)

    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    lock = ctx.Lock()

    print(f"running {len(payloads)} questions, concurrency={args.concurrency}, "
          f"max_episodes={args.max_episodes}, endpoint={args.base_url}", flush=True)
    t_start = time.time()
    results = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.concurrency, mp_context=ctx,
                             initializer=_worker_init,
                             initargs=(cfg, counter, lock)) as ex:
        futs = {ex.submit(_run_one, pl): pl[0]["question_id"] for pl in payloads}
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            detail_f.write(json.dumps(rec) + "\n"); detail_f.flush()
            ans_f.write(json.dumps(to_answer_row(rec, args.model)) + "\n"); ans_f.flush()
            n_done += 1
            el = time.time() - t_start
            print(f"[{n_done}/{len(payloads)}] {rec['question_id']} "
                  f"({rec.get('question_type')}) ing={rec.get('n_episodes_ingested')} "
                  f"llm={rec.get('n_llm_calls_ingest')} ing_s={rec.get('ingest_seconds')} "
                  f"hit={rec.get('search_hit')} corr={rec.get('qa_correct')} "
                  f"err={rec.get('fatal_error')} | elapsed={el:.0f}s", flush=True)
    ans_f.close(); detail_f.close()
    wall = time.time() - t_start

    # Final aggregates read from the detail jsonl (covers resumed rows too).
    all_recs = [json.loads(l) for l in DETAIL_OUT.open()]
    out = {"config": cfg, "aggregates": aggregate(all_recs, wall, cfg),
           "per_question": all_recs}
    AGG_OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {AGG_OUT}\n{json.dumps(out['aggregates'], indent=2)}")


if __name__ == "__main__":
    main()
