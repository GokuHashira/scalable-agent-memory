"""Scoped Graphiti feasibility run on a small LongMemEval_S subset.

For the first ``--n`` questions of a split we:
  1. build a fresh in-process Kuzu knowledge graph (group_id = history_id),
  2. ingest a bounded set of conversation turns as Graphiti episodes
     (the gold session's turns + chronological fillers up to ``--max-episodes``),
  3. run Graphiti hybrid search with the question text,
  4. feed the retrieved facts to the local vLLM endpoint as a reader,
  5. judge the answer against the gold answer with the same endpoint.

Outputs per-question cost/retrieval/QA stats plus an extrapolation to the full
100-question subset. See docs/midterm/graphiti_note.md.

Usage:
    python graphiti/run_graphiti.py --n 5 --split test --max-episodes 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("EMBEDDING_DIM", "768")

import httpx

from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.graph_queries import get_fulltext_indices
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF

from local_clients import LocalCosineReranker, LocalSTEmbedder, make_llm_client

# ---------------------------------------------------------------------------
# Kuzu backend work-arounds (graphiti-core 0.31.2 ships a deprecated, partly
# broken Kuzu driver). Two defects we patch at runtime:
#   1. KuzuDriver never sets ``_database``; graphiti.py:1327 reads it directly
#      (not via getattr) and raises AttributeError on the first add_episode.
#   2. KuzuDriver.build_indices_and_constraints is a no-op, so the FTS indices
#      that hybrid search (and add_episode's dedup) rely on are never created.
# Kuzu FTS indices are static snapshots, so we create them before ingestion
# (empty, so dedup searches don't crash) and rebuild them after ingestion so
# the final search sees every row. The FTS extension loads fully offline.
# ---------------------------------------------------------------------------
FTS_TABLES = [
    ("Episodic", "episode_content"),
    ("Entity", "node_name_and_summary"),
    ("Community", "community_name"),
    ("RelatesToNode_", "edge_name_and_fact"),
]


async def _kuzu_exec(driver: KuzuDriver, stmt: str) -> None:
    await driver.client.execute(stmt)


async def ensure_fts_loaded(driver: KuzuDriver) -> None:
    for stmt in ("INSTALL fts;", "LOAD fts;"):
        try:
            await _kuzu_exec(driver, stmt)
        except Exception:  # noqa: BLE001 -- already installed/loaded is fine
            pass


async def create_fts_indices(driver: KuzuDriver) -> None:
    for stmt in get_fulltext_indices(GraphProvider.KUZU):
        try:
            await _kuzu_exec(driver, stmt)
        except Exception:  # noqa: BLE001 -- "already exists" is fine
            pass


async def rebuild_fts_indices(driver: KuzuDriver) -> None:
    """Drop and recreate FTS indices so they index all post-ingestion rows."""
    for table, name in FTS_TABLES:
        try:
            await _kuzu_exec(driver, f"CALL DROP_FTS_INDEX('{table}', '{name}');")
        except Exception:  # noqa: BLE001
            pass
    await create_fts_indices(driver)

ROOT = Path("/fs/gamma-projects/audio/kajayaku/agents")
DATA = ROOT / "data" / "lme_s_k"
DEFAULT_OUT = ROOT / "runs" / "analysis" / "graphiti_feasibility.json"

# A full LongMemEval_S history is this many turns on average (measured on the
# first 5 test histories: 469-550 turns). Used for the 100-question extrapolation.
FULL_HISTORY_TURNS = 510

logging.basicConfig(level=logging.WARNING)
for noisy in ("graphiti_core", "httpx", "kuzu", "sentence_transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

TS_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2}).*?(\d{2}):(\d{2})")


def parse_ts(ts: str) -> datetime:
    m = TS_RE.search(ts or "")
    if not m:
        return datetime.now(timezone.utc)
    y, mo, d, hh, mm = (int(x) for x in m.groups())
    return datetime(y, mo, d, hh, mm, tzinfo=timezone.utc)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def chat(client: httpx.Client, base_url: str, model: str, prompt: str,
         max_tokens: int = 256, temperature: float = 0.0) -> str:
    """One blocking chat completion against the local endpoint."""
    r = client.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def select_episodes(turns: list[dict], gold_sessions: set[str], cap: int) -> list[dict]:
    """Gold-session turns first, then earliest other turns, capped at ``cap``.

    Returned list is re-sorted into chronological (session_idx, turn_idx) order so
    episodes are ingested in the order they occurred.
    """
    def order_key(r: dict) -> tuple[int, int]:
        return (r.get("metadata", {}).get("session_idx", 0), r["turn_idx"])

    gold = sorted((r for r in turns if r["session_id"] in gold_sessions), key=order_key)
    other = sorted((r for r in turns if r["session_id"] not in gold_sessions), key=order_key)
    chosen = (gold + other)[:cap]
    return sorted(chosen, key=order_key)


async def run_question(q: dict, turns: list[dict], args, hclient: httpx.Client) -> dict:
    hid = q["history_id"]
    gold_sessions = set(q["gold_session_ids"])
    gold_answer = q["answer"]
    eps = select_episodes(turns, gold_sessions, args.max_episodes)
    gold_in_selection = any(r["note_id"] in set(q["gold_note_ids"]) for r in eps)

    embedder = LocalSTEmbedder()
    reranker = LocalCosineReranker(embedder)
    llm = make_llm_client(args.base_url, args.api_key, args.model, args.max_tokens)
    driver = KuzuDriver(db=":memory:")
    driver._database = None  # work-around #1 (see top of file)
    graphiti = Graphiti(llm_client=llm, embedder=embedder,
                        cross_encoder=reranker, graph_driver=driver)

    rec: dict = {
        "question_id": q["question_id"],
        "history_id": hid,
        "question_type": q["question_type"],
        "n_turns_available": len(turns),
        "n_sessions_available": len({r["session_id"] for r in turns}),
        "gold_in_selection": gold_in_selection,
    }
    try:
        await graphiti.build_indices_and_constraints()
        await ensure_fts_loaded(driver)
        await create_fts_indices(driver)  # empty indices so dedup search works
        llm.n_calls = 0
        n_ok = 0
        t0 = time.time()
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
            except Exception as e:  # noqa: BLE001 -- record and continue
                rec.setdefault("ingest_errors", []).append(f"{r['turn_idx']}: {type(e).__name__}: {e}")
        rec["ingest_seconds"] = round(time.time() - t0, 2)
        rec["n_episodes_ingested"] = n_ok
        rec["n_llm_calls_ingest"] = llm.n_calls
        rec["llm_calls_per_episode"] = round(llm.n_calls / max(n_ok, 1), 2)
        rec["ingest_sec_per_episode"] = round(rec["ingest_seconds"] / max(n_ok, 1), 2)

        # ---- search ----
        # Kuzu FTS indices are static snapshots, so rebuild them to cover all
        # ingested rows, then run a combined edge + node + episode hybrid search.
        await rebuild_fts_indices(driver)
        t1 = time.time()
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
        rec["search_seconds"] = round(time.time() - t1, 2)
        rec["n_facts"] = len(facts)
        rec["n_nodes"] = len(node_summaries)
        rec["n_episodes_retrieved"] = len(episode_texts)
        rec["facts_sample"] = facts[:8]
        rec["nodes_sample"] = node_summaries[:8]
        # search_hit: gold answer surfaced anywhere in the retrieved memory.
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
        pred = chat(hclient, args.base_url, args.model, reader_prompt, max_tokens=128)
        rec["predicted_answer"] = pred

        # ---- judge ----
        judge_prompt = (
            "You are grading an answer. The gold answer is the reference.\n"
            f"Question: {q['question']}\nGold answer: {gold_answer}\n"
            f"Model answer: {pred}\n\n"
            "Does the model answer convey the same information as the gold answer? "
            "Reply with exactly one word: CORRECT or WRONG."
        )
        verdict = chat(hclient, args.base_url, args.model, judge_prompt, max_tokens=5)
        rec["judge_raw"] = verdict
        rec["qa_correct"] = verdict.strip().upper().startswith("CORRECT")
    finally:
        try:
            await graphiti.close()
        except Exception:  # noqa: BLE001
            pass
    return rec


async def main_async(args) -> None:
    questions = [q for q in load_rows(DATA / "questions.jsonl") if q["split"] == args.split]
    questions = questions[: args.n]

    turns_by_hid: dict[str, list[dict]] = defaultdict(list)
    wanted = {q["history_id"] for q in questions}
    for line in (DATA / "notes.jsonl").open():
        r = json.loads(line)
        if r["history_id"] in wanted:
            turns_by_hid[r["history_id"]].append(r)

    hclient = httpx.Client()
    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['question_id']} ({q['question_type']}) ...", flush=True)
        try:
            rec = await run_question(q, turns_by_hid[q["history_id"]], args, hclient)
        except Exception as e:  # noqa: BLE001 -- one bad question must not kill the run
            rec = {"question_id": q["question_id"], "history_id": q["history_id"],
                   "fatal_error": f"{type(e).__name__}: {e}"}
        print(f"    -> ingested={rec.get('n_episodes_ingested')} "
              f"llm_calls={rec.get('n_llm_calls_ingest')} "
              f"ingest_s={rec.get('ingest_seconds')} "
              f"hit={rec.get('search_hit')} correct={rec.get('qa_correct')}", flush=True)
        results.append(rec)
    hclient.close()

    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    out = {"config": cfg, "per_question": results,
           "aggregates": aggregate(results)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    agg = out["aggregates"]
    print(json.dumps(agg, indent=2))


def aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("n_episodes_ingested")]
    if not ok:
        return {"n_questions": len(results), "note": "no question ingested any episode"}
    tot_eps = sum(r["n_episodes_ingested"] for r in ok)
    tot_calls = sum(r["n_llm_calls_ingest"] for r in ok)
    tot_sec = sum(r["ingest_seconds"] for r in ok)
    sec_per_ep = tot_sec / tot_eps
    calls_per_ep = tot_calls / tot_eps
    agg = {
        "n_questions": len(results),
        "n_questions_ingested": len(ok),
        "total_episodes": tot_eps,
        "total_ingest_llm_calls": tot_calls,
        "total_ingest_seconds": round(tot_sec, 1),
        "mean_llm_calls_per_episode": round(calls_per_ep, 2),
        "mean_ingest_sec_per_episode": round(sec_per_ep, 2),
        "search_hit_rate": round(sum(bool(r.get("search_hit")) for r in ok) / len(ok), 3),
        "qa_accuracy": round(sum(bool(r.get("qa_correct")) for r in ok) / len(ok), 3),
        "gold_in_selection_rate": round(sum(bool(r.get("gold_in_selection")) for r in ok) / len(ok), 3),
    }
    # Extrapolation: full ingestion of 100 histories (avg FULL_HISTORY_TURNS turns each).
    full_eps = FULL_HISTORY_TURNS * 100
    agg["extrapolation_100q_full_history"] = {
        "assumed_turns_per_history": FULL_HISTORY_TURNS,
        "episodes_total": full_eps,
        "est_llm_calls": int(full_eps * calls_per_ep),
        "est_ingest_hours_serial": round(full_eps * sec_per_ep / 3600, 1),
    }
    return agg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--split", default="test")
    p.add_argument("--max-episodes", type=int, default=12,
                   help="cap on turns ingested per question (gold session + fillers)")
    p.add_argument("--num-results", type=int, default=10, help="search top-k facts")
    p.add_argument("--text-cap", type=int, default=1200, help="max chars per episode body")
    p.add_argument("--base-url", default="http://localhost:8001/v1")
    p.add_argument("--api-key", default="sk-local")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
