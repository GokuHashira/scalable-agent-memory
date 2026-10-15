# Retrieval results on LongMemEval_S

## Setup

- 500 questions (30 abstention); 246,607 turn notes and 23,854 session notes. Each question is answered from its
  own history. Recall is computed over the 470 non-abstention questions.
- Per question: at most 20 results and 2,048 retrieved tokens (`cl100k_base`).
- Embeddings: `Qwen/Qwen3-Embedding-0.6B` (1024-d, max 2,048 tokens per text).
- Methods: bm25; dense (exact inner product); hybrid (reciprocal-rank fusion, k=60, top 50 of each); summary_first
  (extractive session summaries, top 5 sessions expanded); one_hop and bounded (hybrid top-5 anchors expanded along
  temporal, semantic, version and sibling links; bounded uses at most 4 hops and 50 candidates).
- Metrics: session recall@k is the share of gold sessions with a note in the top k; turn recall@k is the share of
  gold turns in the top k.

## Results

| Method | Session recall@5 | Session recall@20 | Turn recall@20 | Mean tokens | Mean results | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| bm25 | 0.838 | 0.904 | 0.753 | 1,845 | 10.2 | 2.1 | 3.6 |
| dense | 0.906 | 0.988 | 0.947 | 1,570 | 17.1 | 347.9 | 497.3 |
| hybrid | 0.893 | 0.966 | 0.885 | 1,808 | 12.9 | 343.8 | 534.7 |
| summary_first | 0.841 | 0.879 | 0.862 | 1,730 | 15.7 | 346.6 | 509.7 |
| one_hop | — | — | — | — | — | — | — |
| bounded | 0.893 | 0.947 | 0.924 | 1,759 | 15.7 | 337.6 | 526.5 |

Every method returns 500 result sets; none is empty and none exceeds 2,048 tokens. one_hop is being re-run.

### Turn recall@20 by question type

| Method | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---|---|---|---|---|---|
| bm25 | 0.910 | 0.604 | 0.875 | 0.372 | 0.930 | 0.754 |
| dense | 0.986 | 0.925 | 0.982 | 0.906 | 0.977 | 0.924 |
| hybrid | 0.975 | 0.811 | 0.982 | 0.733 | 0.977 | 0.852 |
| summary_first | 0.931 | 0.840 | 0.964 | 0.722 | 0.859 | 0.831 |
| bounded | 0.986 | 0.888 | 1.000 | 0.756 | 0.992 | 0.895 |

### Latency by stage (p50 ms)

| Method | Query encoding | Search | Fusion / expansion / re-rank | Packing | Total |
|---|---|---|---|---|---|
| bm25 | — | 1.3 | — | 0.9 | 2.1 |
| dense | 346.4 | 0.2 | — | 1.0 | 347.9 |
| hybrid | 339.3 | 2.5 | 0.1 | 1.2 | 343.8 |
| summary_first | 344.3 | 0.4 | 0.2 | 1.0 | 346.6 |
| bounded | 334.4 | 2.2 | 0.2 | 1.2 | 337.6 |

### Index build cost

| Component | Build s | Disk MB | Projected @100K turns | Projected @1M turns |
|---|---|---|---|---|
| bm25 | 25.5 | 281.5 | 10 s / 114 MB | 104 s / 1.1 GB |
| dense | 1,009.0 | 512.4 | 409 s / 208 MB | 4,092 s / 2.1 GB |
| summary_first | 507.2 | 181.5 | 206 s / 74 MB | 2,057 s / 736 MB |
| links | 46.7 | 450.8 | 19 s / 183 MB | 190 s / 1.8 GB |

Projections are linear in turn-note count for per-history flat indexes. Peak memory was about 3.5 GB.

## Observations

- dense has the highest recall on every metric; hybrid is lower than dense on turn recall (0.885 vs 0.947).
- bm25 is weakest on single-session-preference (0.372) and multi-session (0.604) turn recall.
- Query encoding accounts for about 99% of latency in every method that embeds the query; all other stages
  together take under 3 ms.
- bounded reached the 50-candidate limit on all 500 questions, after a mean of 2.28 hops.
- summary_first has the lowest session recall@20 (0.879).

## Example failures

| Question | Type | Missed by | Found by | What was retrieved instead |
|---|---|---|---|---|
| "Can you recommend a show or movie for me to watch tonight?" (gold: user likes stand-up comedy specials on Netflix) | single-session-preference | bm25 | dense, hybrid, summary_first | 10 long, loosely related turns filling 2,022 of 2,048 tokens |
| "What is the order of the three sports events I participated in during the past month?" (3 gold sessions) | temporal-reasoning | bm25 | dense, hybrid | exercise-routine turns from 6 other sessions; dense ranks a gold session first |
| Same question as the first row | single-session-preference | bounded | dense | 5 anchors from wrong sessions; the 50-candidate limit was reached after 1 hop, mostly on their siblings |

## Environment

Retrieval runs: AMD EPYC 7763 (128 cores), 16 threads, exclusive node. Index builds: NVIDIA A100-SXM4-40GB.
Python 3.12.15, torch 2.14.1, sentence-transformers 6.1.0, faiss-cpu 1.15.1, rank-bm25 0.2.2. Results are warm
(indexes loaded, 20 warm-up queries). Fixture tests confirm identical rankings across repeated runs and after
save/load.

## Reproduce

```bash
uv run python scripts/evaluate.py --runs 'runs/*-lme_s-seed0-warm' --questions data/lme_s/questions.jsonl
uv run python scripts/profile_indexes.py --dataset data/lme_s
uv run python scripts/failure_traces.py --runs runs/*-lme_s-seed0-warm --questions data/lme_s/questions.jsonl
```
