"""Plan registry: plan id → a ready-to-build retriever wired from the pinned config.

Leaf components (P0 lexical, P1 dense) are the units persisted per history; composite plans reuse them, so the
expensive embeddings are computed once by build_indexes and only loaded by run_plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mem27.retrieval.base import BaseRetriever
from mem27.retrieval.encoder import Encoder
from mem27.retrieval.p0_bm25 import BM25Retriever
from mem27.retrieval.p1_dense import DenseRetriever
from mem27.retrieval.p2_hybrid import HybridRetriever
from mem27.schema import Note

PLAN_IDS = ("p0", "p1", "p2")


def make_bm25(config: dict[str, Any]) -> BM25Retriever:
    return BM25Retriever(k1=config["bm25"]["k1"], b=config["bm25"]["b"])


def make_hybrid(encoder: Encoder, config: dict[str, Any]) -> HybridRetriever:
    return HybridRetriever(
        make_bm25(config), DenseRetriever(encoder), rrf_k=config["hybrid"]["rrf_k"], depth=config["hybrid"]["depth"]
    )


def make_plan(plan: str, encoder: Encoder, config: dict[str, Any]) -> BaseRetriever:
    if plan == "p0":
        return make_bm25(config)
    if plan == "p1":
        return DenseRetriever(encoder)
    if plan == "p2":
        return make_hybrid(encoder, config)
    raise ValueError(f"unknown plan {plan!r}; expected one of {', '.join(PLAN_IDS)}")


def _load_bm25(index_dir: Path, history_id: str, notes: list[Note], config: dict[str, Any]) -> BM25Retriever:
    bm25 = make_bm25(config)
    bm25.load(index_dir / history_id / "bm25", notes)
    return bm25


def _load_dense(index_dir: Path, history_id: str, notes: list[Note], encoder: Encoder) -> DenseRetriever:
    dense = DenseRetriever(encoder)
    dense.load(index_dir / history_id / "dense", notes)
    return dense


def load_plan(
    plan: str, index_dir: Path, history_id: str, notes: list[Note], encoder: Encoder | None, config: dict[str, Any]
) -> BaseRetriever:
    """A ready retriever for one history, assembled from persisted components (nothing is re-embedded)."""
    if plan == "p0":
        return _load_bm25(index_dir, history_id, notes, config)
    if encoder is None:
        raise ValueError(f"plan {plan} needs an encoder to embed queries; pass --encoder model|stub")
    if plan == "p1":
        return _load_dense(index_dir, history_id, notes, encoder)
    if plan == "p2":
        hybrid = HybridRetriever(
            _load_bm25(index_dir, history_id, notes, config),
            _load_dense(index_dir, history_id, notes, encoder),
            rrf_k=config["hybrid"]["rrf_k"],
            depth=config["hybrid"]["depth"],
        )
        hybrid.build(notes, None)  # components are loaded, so this only registers the notes
        return hybrid
    raise ValueError(f"unknown plan {plan!r}; expected one of {', '.join(PLAN_IDS)}")
