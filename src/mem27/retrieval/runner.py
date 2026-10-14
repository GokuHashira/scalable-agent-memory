"""Run one plan over a dataset and write a self-describing run directory (FR-012, FR-016, FR-017).

runs/<plan>-<dataset>-seed<N>-<cache>/ gets bundles.jsonl, config.resolved.yaml and env.json. Warm runs execute
`warmup_queries` throwaway queries first; indexes are always loaded from disk, never rebuilt here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from mem27.retrieval.base import BaseRetriever
from mem27.retrieval.encoder import Encoder, make_encoder
from mem27.retrieval.indexing import group_by_history
from mem27.retrieval.plans import PLAN_IDS, load_plan
from mem27.retrieval.profiling import capture_env
from mem27.schema import Question, RetrievalRequest, load_notes, load_questions, to_dict, write_jsonl


def default_run_dir(plan: str, dataset_dir: Path, seed: int, cache: str) -> Path:
    return Path("runs") / f"{plan}-{dataset_dir.name}-seed{seed}-{cache}"


def _request(question: Question, plan: str, config: dict[str, Any]) -> RetrievalRequest:
    return RetrievalRequest(
        query=question.question,
        budget_ms=config["budget_ms"],
        budget_tokens=config["budget_tokens"],
        max_results=config["max_results"],
        as_of_time=question.question_date,
        plan=plan,
        question_id=question.question_id,
    )


class PlanCache:
    """Loads each history's retriever once per process (warm behaviour)."""

    def __init__(self, plan: str, index_dir: Path, notes_by_history: dict, encoder: Encoder | None, config: dict):
        self.plan, self.index_dir, self.notes_by_history = plan, index_dir, notes_by_history
        self.encoder, self.config = encoder, config
        self._loaded: dict[str, BaseRetriever] = {}

    def get(self, history_id: str) -> BaseRetriever:
        if history_id not in self._loaded:
            self._loaded[history_id] = load_plan(
                self.plan, self.index_dir, history_id, self.notes_by_history[history_id], self.encoder, self.config
            )
        return self._loaded[history_id]


def run_plan(
    plan: str,
    dataset_dir: Path,
    index_dir: Path,
    out_dir: Path,
    config: dict[str, Any],
    encoder_kind: str,
    seed: int,
    cache: str,
    warmup: bool,
    limit: int | None,
) -> Path:
    if plan not in PLAN_IDS:
        raise ValueError(f"unknown plan {plan!r}; expected one of {', '.join(PLAN_IDS)}")
    questions = load_questions(dataset_dir / "questions.jsonl")[:limit]
    notes_by_history = group_by_history(load_notes(dataset_dir / "notes.jsonl"))
    encoder = None if plan == "p0" else make_encoder(encoder_kind, config)
    plans = PlanCache(plan, index_dir, notes_by_history, encoder, config)
    if warmup:
        for q in questions[: config["warmup_queries"]]:
            plans.get(q.history_id).retrieve(_request(q, plan, config))
    bundles = [plans.get(q.history_id).retrieve(_request(q, plan, config)) for q in questions]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "bundles.jsonl", (to_dict(b) for b in bundles))
    resolved = {
        **config,
        "plan": plan,
        "dataset": str(dataset_dir),
        "index_dir": str(index_dir),
        "encoder_kind": encoder_kind,
        "encoder_name": encoder.name if encoder else None,
        "seed": seed,
        "cache": cache,
        "warmup": warmup,
        "limit": limit,
    }
    (out_dir / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=True))
    env = capture_env(seed, cache)
    (out_dir / "env.json").write_text(json.dumps(env, indent=2))
    return out_dir


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Run one retrieval plan over a dataset.")
    parser.add_argument("--plan", required=True, choices=PLAN_IDS)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--index-dir", default=None, help="default: <dataset>/index")
    parser.add_argument("--config", default="configs/retrieval.yaml")
    parser.add_argument("--encoder", default="model", choices=["model", "stub"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", default="warm", choices=["warm", "cold"])
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    dataset = Path(args.dataset)
    index_dir = Path(args.index_dir) if args.index_dir else dataset / "index"
    out = Path(args.out) if args.out else default_run_dir(args.plan, dataset, args.seed, args.cache)
    config = yaml.safe_load(Path(args.config).read_text())
    warmup = args.cache == "warm" and not args.no_warmup
    run_dir = run_plan(
        args.plan, dataset, index_dir, out, config, args.encoder, args.seed, args.cache, warmup, args.limit
    )
    print(f"wrote {run_dir}")
    return run_dir
