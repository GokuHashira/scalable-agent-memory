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
| `make fixture` | laptop | build the 1K-note deterministic fixture in `data/fixture/` |
| `make check` | laptop | ruff format check, ruff lint, ty, pytest |
| `make baseline DS=lme_s_g` | Zaratan | build indexes, run P0–P5, evaluate |
| `make reproduce DS=lme_s_g` | anywhere | rebuild the tables in `docs/midterm/` from `runs/` |

## Layout

`src/mem27/` is the library: `schema.py` holds the shared contract, and `retrieval/` holds P0–P5, links and summaries.
`scripts/` holds thin CLIs and `scripts/slurm/` the Slurm wrappers. Experiment knobs live in `configs/`. Tests are in
`tests/`, and run outputs go to `runs/<plan>-<dataset>-seed<N>-<cold|warm>/`, which is git-ignored.
