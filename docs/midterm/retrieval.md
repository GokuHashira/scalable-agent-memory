# Midterm — Retrieval substrate

**Status (2026-10-15):** draft. Five of six retrieval methods are measured on LongMemEval_S. one_hop is being re-run
after an environment failure. Cold-vs-warm and the real-data determinism re-run are not done yet (see §6).

## 1. Setup

- **Data:** LongMemEval_S (`longmemeval_s_cleaned.json`), 500 questions, 30 of them abstention. The conversion gives
  246,607 turn notes and 23,854 session-parent notes, about 493 turns and 48 sessions per history. 13 sessions that
  were exact duplicates within a haystack were dropped. Each question has its own index over its own haystack.
- **Scoring:** evidence-level only, with no reader or judge. Recall is computed over the 470 non-abstention
  questions:
  - session recall@k: the share of gold sessions with any note in the top k
  - turn recall@k: the share of `has_answer` turns in the top k
- **Budget:** at most 20 results and 2,048 retrieved tokens (tiktoken `cl100k_base`) per question.
- **Embeddings:** `Qwen/Qwen3-Embedding-0.6B` (1024-d, `max_seq_length` 2048, task instruction on queries).
  We switched from bge-base, which truncates at 512 tokens and so cuts long turns.
- **Retrieval methods:** bm25 · dense (exact inner product) · hybrid (RRF k=60 over the top 50 of each) · summary_first
  (extractive summaries, top 5 sessions expanded) · one_hop · bounded (anchors = hybrid top 5,
  BFS with H=4, L=50, re-ranked by cosine). Links are deterministic: temporal, semantic top-5, version, sibling.

## 2. Retrieval table (LongMemEval_S, 470 scored questions, seed 0, warm)

| Method | Session recall@5 | Session recall@20 | Turn recall@20 | Mean tokens | Mean results | p50 ms | p95 ms | Run |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.838 | 0.904 | 0.753 | 1,845 | 10.2 | **2.1** | **3.6** | `bm25-lme_s-seed0-warm` |
| dense | **0.906** | **0.988** | **0.947** | 1,570 | 17.1 | 347.9 | 497.3 | `dense-lme_s-seed0-warm` |
| hybrid | 0.893 | 0.966 | 0.885 | 1,808 | 12.9 | 343.8 | 534.7 | `hybrid-lme_s-seed0-warm` |
| summary_first | 0.841 | 0.879 | 0.862 | 1,730 | 15.7 | 346.6 | 509.7 | `summary_first-lme_s-seed0-warm` |
| one_hop | — | — | — | — | — | — | — | re-running |
| bounded | 0.893 | 0.947 | 0.924 | 1,759 | 15.7 | 337.6 | 526.5 | `bounded-lme_s-seed0-warm` |

All five measured methods return exactly 500 bundles. No bundle is empty and none exceeds 2,048 tokens.

### Per question type: turn recall@20

| Method | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|
| bm25 | 0.910 | 0.604 | 0.875 | 0.372 | 0.930 | 0.754 |
| dense | 0.986 | 0.925 | 0.982 | 0.906 | 0.977 | 0.924 |
| hybrid | 0.975 | 0.811 | 0.982 | 0.733 | 0.977 | 0.852 |
| summary_first | 0.931 | 0.840 | 0.964 | 0.722 | 0.859 | 0.831 |
| bounded | 0.986 | 0.888 | 1.000 | 0.756 | 0.992 | 0.895 |

Session recall@20 per type is in `retrieval_table.md`.

## 3. Latency breakdown (p50 ms per stage)

| Method | Query encoding | Search | Fusion and expansion | Packing | Total |
|---|---|---|---|---|---|
| bm25 | — | 1.3 (bm25) | — | 0.9 | 2.1 |
| dense | 346.4 | 0.2 | — | 1.0 | 347.9 |
| hybrid | 339.3 | 2.2 bm25 + 0.3 dense | 0.1 fuse | 1.2 | 343.8 |
| summary_first | 344.3 | 0.3 bm25 + 0.1 dense over summaries | 0.2 rank turns | 1.0 | 346.6 |
| bounded | 334.4 | 1.9 bm25 + 0.3 dense | 0.0 fuse, 0.0 expand, 0.2 rerank | 1.2 | 337.6 |

About 99% of every dense-based method's latency is **encoding the query** with a 0.6B model on CPU (16 threads). The
search, fusion, traversal and re-ranking together take under 3 ms. A first dense run with one thread had a p50 of
3,396 ms. We discarded it because it measured thread starvation, not retrieval cost.

## 4. Index build cost

| Component | Histories | Build s | Disk MB | Projected @100K turns | Projected @1M turns |
|---|---|---|---|---|---|
| bm25 | 500 | 25.5 | 281.5 | 10 s / 114 MB | 104 s / 1.1 GB |
| dense (Qwen3-0.6B, A100-40GB) | 500 | 1,009.0 | 512.4 | 409 s / 208 MB | 4,092 s / 2.1 GB |
| summary_first (summaries + their index) | 500 | 507.2 | 181.5 | 206 s / 74 MB | 2,057 s / 736 MB |
| links (deterministic) | 500 | 46.7 | 450.8 | 19 s / 183 MB | 190 s / 1.8 GB |

The projection is linear in turn-note count, for per-history flat indexes. It does not cover approximate (HNSW) or
Lucene indexes at 1M scale. Peak RSS was about 3.5 GB, measured as the process high-water mark. Links take nearly as
much disk as the embeddings, because sibling links are complete within each session.

## 5. Findings

1. **Dense alone beats hybrid.** Fusing with BM25 lowers turn recall from 0.947 to 0.885. RRF gives the weaker lexical
   list equal weight, and BM25 is far weaker on preference (0.37) and multi-session (0.60) questions. Hybrid fusion
   is therefore not the strongest single method on this benchmark.
2. **The cost split is binary.** bm25 costs about 2 ms. Every method that embeds the query costs about 340 ms, and the
   methods' structural differences add under 3 ms. Latency is therefore driven by whether a query is embedded at all, not by
   the retrieval structure built on top of it.
3. **Bounded traversal recovers part of hybrid's loss** (turn recall 0.924 vs 0.885), but it is limited by the cap,
   not the hop budget. It hits L=50 on all 500 questions, after a mean of 2.28 hops.
4. **Extractive summaries are the weak link.** summary_first has the lowest session recall@20 (0.879). Its failures
   cluster on single-session-user (9), preference (8) and temporal (8) questions. Abstractive (LLM) summaries are a
   likely improvement.

## 6. Determinism, cold vs warm, hardware

- **Determinism:** on the fixture, every method reproduces identical note ids and scores across runs, and save/load
  round-trips are identical (`tests/test_retrievers.py::test_deterministic`, `tests/test_persistence.py`).
  The real-data re-run (hybrid on 50 questions, compared with `scripts/compare_bundles.py`) has not been run yet.
- **Cold vs warm:** not measured yet. All numbers above are warm: indexes loaded, then 20 warm-up queries.
- **Hardware:** the runs used Zaratan `standard` nodes, AMD EPYC 7763 (128 cores), with 16 threads for query encoding
  and `--exclusive` nodes. The builds used Zaratan `gpu` nodes, NVIDIA A100-SXM4-40GB. Software: Python 3.12.15,
  torch 2.14.1, sentence-transformers 6.1.0, faiss-cpu 1.15.1, rank-bm25 0.2.2. Every run directory holds its own
  `env.json`.

## 7. Failure traces

1. **Lexical miss, preference (bm25), question `32260d93`.** "Can you recommend a show or movie for me to watch
   tonight?" Gold: the session where the user said they like stand-up comedy specials on Netflix. The question shares
   no words with that evidence. BM25 ranks long, loosely related turns first (true-crime and self-improvement
   recommendations), and fills 2,022 of its 2,048 tokens with only 10 turns. Dense, hybrid and summary_first all find
   the gold session.
2. **Multi-event temporal (bm25), question `gpt4_e061b84f`.** "What is the order of the three sports events I
   participated in during the past month?" There are three gold sessions (a triathlon, a 5K and a soccer tournament).
   BM25 retrieves exercise-routine chatter from six other sessions and misses all three. Dense puts a gold session
   first ("…after my volleyball league game") and covers all three in its top 20.
3. **Traversal limited by the cap (bounded), question `32260d93`.** The five hybrid anchors all come from wrong
   sessions (the bm25-favoured `573dc24b_1`, `04bef53b` and others). Expansion reaches the 50-candidate cap after
   **1 hop**, mostly on siblings of those wrong anchors (`stop_reason=cap`, `hops_used=1`), so it never reaches the
   gold session that dense alone ranks within its top 20. Wrong anchors plus a sibling-heavy expansion consume the
   budget.

## 8. Reproduce

```bash
uv run python scripts/evaluate.py --runs 'runs/*-lme_s-seed0-warm' --questions data/lme_s/questions.jsonl
uv run python scripts/profile_indexes.py --dataset data/lme_s
uv run python scripts/failure_traces.py --runs runs/*-lme_s-seed0-warm --questions data/lme_s/questions.jsonl
```
