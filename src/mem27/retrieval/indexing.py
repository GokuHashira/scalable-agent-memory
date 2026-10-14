"""Build and persist per-history indexes with build-cost profiling (FR-014, FR-018).

Layout: <index_dir>/<history_id>/<component>/..., plus <index_dir>/build_profile.json. Components: bm25 (P0),
dense (P1). Composite plans load these instead of rebuilding, so embeddings are computed exactly once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from mem27.retrieval.encoder import Encoder, make_encoder
from mem27.retrieval.p1_dense import DenseRetriever
from mem27.retrieval.plans import make_bm25
from mem27.retrieval.profiling import capture_env, measure_build
from mem27.schema import Note, load_notes

COMPONENTS = ("p0", "p1")
COMPONENT_DIR = {"p0": "bm25", "p1": "dense"}


def group_by_history(notes: list[Note]) -> dict[str, list[Note]]:
    grouped: dict[str, list[Note]] = {}
    for note in notes:
        grouped.setdefault(note.history_id, []).append(note)
    return grouped


def _build_component(
    component: str, notes: list[Note], out_dir: Path, encoder: Encoder | None, config: dict[str, Any]
) -> dict[str, float]:
    retriever = make_bm25(config) if component == "p0" else DenseRetriever(_require(encoder))
    with measure_build(out_dir) as profile:
        retriever.build(notes, None)
        retriever.save(out_dir)
    profile["disk_bytes"] = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
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
    unknown = [c for c in components if c not in COMPONENTS]
    if unknown:
        raise ValueError(f"unknown components {unknown}; buildable here: {', '.join(COMPONENTS)}")
    notes = load_notes(dataset_dir / "notes.jsonl")
    encoder = make_encoder(encoder_kind, config) if "p1" in components else None
    per_history: dict[str, dict[str, dict[str, float]]] = {}
    for history_id, history_notes in sorted(group_by_history(notes).items()):
        per_history[history_id] = {
            c: _build_component(c, history_notes, index_dir / history_id / COMPONENT_DIR[c], encoder, config)
            for c in components
        }
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
    """Keeps totals from earlier builds of other components (e.g. p0,p1 first, then p3,links)."""
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
    parser.add_argument("--plans", default="p0,p1", help="comma-separated components to build")
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
