# mem27: Scalable Agent Memory

Large memory with bounded retrieval and maintenance cost (CMSC 848N, Fall 2026, Project 27).
The memory representation is fixed, and retrieval and maintenance plans vary.

## Setup

```bash
uv sync                    # laptop: library + dev tools (tests use a stub encoder, no torch)
uv sync --extra encode     # cluster: adds sentence-transformers / torch for real embeddings
```

## Common commands

| Command | Where | What |
|---|---|---|
| `make fixture` | laptop | build the 5,500-note deterministic fixture in `data/fixture/` |
| `make check` | laptop | ruff format check, ruff lint, ty, pytest |
| `make baseline DS=lme_s_raw` | Zaratan | build indexes, run all six plans, evaluate |
| `make reproduce DS=lme_s_raw` | anywhere | rebuild the tables in `docs/midterm/` from `runs/` |
| `make planner-all` | GPU machine | convert, embed, tune the cascade, run all planners, oracle, timing, report (`run_all.sh`) |
| `make planner-run` / `planner-oracle` / `planner-report` | GPU machine | individual planner steps on `data/lme_s_k` (`source env.sh` first) |

## Layout

`src/mem27/` is the library: `schema.py` holds the shared contract, `retrieval/` the six retrieval plans, links and summaries,
`data/` the dataset converters and synthetic workload, `eval/` the reader, judge and metrics harness, `planner/` the
query planners and their reference retrievers, `reader/` the vLLM reader, and `oracle.py` / `analysis.py` the oracle and
headroom analysis. `allmem/` and `graphiti/` hold the external memory-system baselines.
`scripts/` holds thin CLIs and `scripts/slurm/` the Slurm wrappers. Experiment knobs live in `configs/`. Tests are in
`tests/`, and run outputs go to `runs/<method>-<dataset>-seed<N>-<cold|warm>/`, which is git-ignored.
