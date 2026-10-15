"""Build and persist per-history indexes with build-cost profiling (FR-014, FR-018).

Layout per history: <index_dir>/<history_id>/{bm25/, dense/, summary/, links.jsonl}, plus
<index_dir>/build_profile.json and <index_dir>/summaries.jsonl (all sessions). Components: bm25, dense (turn
embeddings), summary_first (extractive summaries + their index), links (deterministic link graph). summary_first
and links reuse the persisted dense embeddings, so turns are embedded exactly once. Composite plans load these
instead of rebuilding.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from mem27.retrieval.dense import DenseRetriever
from mem27.retrieval.encoder import Encoder, make_encoder
from mem27.retrieval.links import build_links, write_links
from mem27.retrieval.plans import make_bm25, make_summary_first
from mem27.retrieval.profiling import capture_env, disk_bytes, measure_build
from mem27.retrieval.summaries import build_extractive_summaries
from mem27.schema import Note, Summary, load_notes, write_jsonl

COMPONENTS = ("bm25", "dense", "summary_first", "links")
COMPONENT_PATH = {"bm25": "bm25", "dense": "dense", "summary_first": "summary", "links": "links.jsonl"}
_COMPONENT_ALIASES = {"p0": "bm25", "p1": "dense", "p3": "summary_first"}


def group_by_history(notes: list[Note]) -> dict[str, list[Note]]:
    grouped: dict[str, list[Note]] = {}
    for note in notes:
        grouped.setdefault(note.history_id, []).append(note)
    return grouped


def _load_dense(history_dir: Path, notes: list[Note], encoder: Encoder | None) -> DenseRetriever:
    dense = DenseRetriever(_require(encoder))
    dense.load(history_dir / "dense", notes)
    return dense


def _build_component(
    component: str,
    notes: list[Note],
    history_dir: Path,
    encoder: Encoder | None,
    config: dict[str, Any],
    summaries_out: list[Summary],
) -> dict[str, float]:
    target = history_dir / COMPONENT_PATH[component]
    with measure_build() as profile:
        if component == "bm25":
            retriever = make_bm25(config)
            retriever.build(notes, None)
            retriever.save(target)
        elif component == "dense":
            retriever = DenseRetriever(_require(encoder))
            retriever.build(notes, None)
            retriever.save(target)
        elif component == "summary_first":
            summaries = build_extractive_summaries(notes, created_ts=datetime.now(UTC).isoformat())
            p3 = make_summary_first(_load_dense(history_dir, notes, encoder), _require(encoder), config)
            p3.build(notes, summaries)
            p3.save(target)
            summaries_out.extend(summaries)
        else:  # links
            graph = build_links(notes, _load_dense(history_dir, notes, encoder), config["links"]["semantic_k"])
            write_links(target, graph)
            profile["links"] = len(graph)
    profile["disk_bytes"] = disk_bytes(target)
    return profile


def _require(encoder: Encoder | None) -> Encoder:
    if encoder is None:
        raise ValueError("an encoder is required to build dense components; pass --encoder model|stub")
    return encoder


def _totals(per_history: dict[str, dict[str, dict[str, float]]], components: list[str]) -> dict[str, dict]:
    totals: dict[str, dict] = {}
    for component in components:
        profiles = [h[component] for h in per_history.values() if component in h]
        totals[component] = {
            "histories": len(profiles),
            "build_ms": sum(p["build_ms"] for p in profiles),
            "disk_bytes": sum(p["disk_bytes"] for p in profiles),
            "peak_rss_mb": max((p["peak_rss_mb"] for p in profiles), default=0.0),
        }
    return totals


def build_indexes(
    dataset_dir: Path, index_dir: Path, components: list[str], encoder_kind: str, config: dict[str, Any]
) -> dict[str, Any]:
    components = [_COMPONENT_ALIASES.get(c, c) for c in components]
    unknown = [c for c in components if c not in COMPONENTS]
    if unknown:
        raise ValueError(f"unknown components {unknown}; buildable here: {', '.join(COMPONENTS)}")
    notes = load_notes(dataset_dir / "notes.jsonl")
    needs_encoder = any(c in components for c in ("dense", "summary_first", "links"))
    encoder = make_encoder(encoder_kind, config) if needs_encoder else None
    per_history: dict[str, dict[str, dict[str, float]]] = {}
    summaries: list[Summary] = []
    for history_id, history_notes in sorted(group_by_history(notes).items()):
        per_history[history_id] = {
            c: _build_component(c, history_notes, index_dir / history_id, encoder, config, summaries)
            for c in components
        }
    if summaries:
        write_jsonl(index_dir / "summaries.jsonl", sorted(summaries, key=lambda s: s.summary_id))
    profile = {
        "dataset": str(dataset_dir),
        "n_notes": len(notes),
        "n_turn_notes": sum(1 for n in notes if not n.is_session_parent),
        "encoder": encoder.name if encoder else None,
        "note": "peak_rss_mb is the process high-water mark at the end of each build, not a per-build delta",
        "totals": _totals(per_history, components),
        "histories": per_history,
        "env": capture_env(seed=0, cache="build"),
    }
    _merge_profile(index_dir / "build_profile.json", profile)
    return profile


def _merge_profile(path: Path, profile: dict[str, Any]) -> None:
    """Keeps totals from earlier builds of other components (e.g. bm25,dense first, then summary_first,links)."""
    if path.exists():
        previous = json.loads(path.read_text())
        profile["totals"] = {**previous.get("totals", {}), **profile["totals"]}
        for history_id, comps in previous.get("histories", {}).items():
            profile["histories"].setdefault(history_id, {})
            profile["histories"][history_id] = {**comps, **profile["histories"][history_id]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Build and persist per-history retrieval indexes.")
    parser.add_argument("--dataset", required=True, help="dir with notes.jsonl")
    parser.add_argument("--index-dir", default=None, help="default: <dataset>/index")
    parser.add_argument("--plans", default="bm25,dense", help="components: bm25,dense,summary_first,links")
    parser.add_argument("--encoder", default="model", choices=["model", "stub"])
    parser.add_argument("--config", default="configs/retrieval.yaml")
    args = parser.parse_args(argv)
    dataset = Path(args.dataset)
    index_dir = Path(args.index_dir) if args.index_dir else dataset / "index"
    config = yaml.safe_load(Path(args.config).read_text())
    profile = build_indexes(dataset, index_dir, args.plans.split(","), args.encoder, config)
    for component, total in profile["totals"].items():
        print(f"{component}: {total['histories']} histories, {total['build_ms'] / 1000:.1f}s, {total['disk_bytes']} B")
    return profile
