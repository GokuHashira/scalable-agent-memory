# Pinned prompts

- `reader.txt` is the pinned reader prompt: chain-of-note (copied from `reader_con.txt`), with chronological
  session grouping and no date hints (`reader_flags.txt` is empty). It was chosen on the 100-question dev
  split, Oct 13. Dev accuracy, Qwen3-8B reader, Qwen3-32B judge:

  | prompt | gold sessions | BM25 bundles |
  |---|---|---|
  | `reader_con.txt` (chain-of-note) | 78.0 | 64.0 |
  | `reader_plain.txt` (official LongMemEval prompt) | 68.0 | 59.0 |
  | `reader_concise.txt` | 64.0 | 52.0 |

  Rank-order rendering scored 74.0 vs 78.0 for chronological. Date hints scored 74.0 vs 78.0 (no gain).
  The full sweep is in `runs/_tables/dev_prompt_sweep.md`.
- `judge.txt` holds the LongMemEval judge prompts, verbatim.
- `stale_judge.txt` and `ku_explicit.txt` are V's prompts for the freshness analysis on LongMemEval
  knowledge-update questions.
- Frozen with `freeze-v1`. Change only after a team message.
