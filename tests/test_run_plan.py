"""End-to-end smoke test of build → run on the fixture (FR-016)."""

import json

import yaml

from mem27.retrieval import indexing, runner
from mem27.schema import read_jsonl


def test_build_then_run_writes_run_directory(tmp_path, fixture_dir):
    index_dir = tmp_path / "index"
    indexing.main(
        ["--dataset", str(fixture_dir), "--index-dir", str(index_dir), "--plans", "bm25,dense", "--encoder", "stub"]
    )
    profile = json.loads((index_dir / "build_profile.json").read_text())
    assert {"bm25", "dense"} <= set(profile["totals"])
    out = tmp_path / "runs" / "hybrid-fixture-seed0-warm"
    runner.main(
        [
            "--plan", "hybrid",
            "--dataset", str(fixture_dir),
            "--index-dir", str(index_dir),
            "--encoder", "stub",
            "--seed", "0",
            "--cache", "warm",
            "--limit", "5",
            "--out", str(out),
        ]
    )  # fmt: skip
    bundles = list(read_jsonl(out / "bundles.jsonl"))
    assert len(bundles) == 5
    assert all(b["plan"] == "p2" for b in bundles)
    env = json.loads((out / "env.json").read_text())
    assert env["cache"] == "warm" and env["seed"] == 0
    resolved = yaml.safe_load((out / "config.resolved.yaml").read_text())
    assert resolved["plan"] == "p2" and resolved["budget_tokens"] == 2048


def test_structure_components_and_structured_plan_runs(tmp_path, fixture_dir):
    index_dir = tmp_path / "index"
    base = ["--dataset", str(fixture_dir), "--index-dir", str(index_dir), "--encoder", "stub"]
    indexing.main([*base, "--plans", "bm25,dense"])
    indexing.main([*base, "--plans", "summary_first,links"])
    profile = json.loads((index_dir / "build_profile.json").read_text())
    assert {"bm25", "dense", "summary_first", "links"} <= set(profile["totals"])
    assert (index_dir / "fx000" / "links.jsonl").exists()
    assert (index_dir / "fx000" / "summary" / "summaries.jsonl").exists()
    assert len(list(read_jsonl(fixture_dir.parent / "fixture" / "notes.jsonl"))) > 0
    for plan, pid in (("summary_first", "p3"), ("one_hop", "p4"), ("bounded", "p5")):
        out = tmp_path / "runs" / f"{plan}-fixture-seed0-warm"
        runner.main(["--plan", plan, *base[:4], "--encoder", "stub", "--limit", "3", "--no-warmup", "--out", str(out)])
        bundles = list(read_jsonl(out / "bundles.jsonl"))
        assert len(bundles) == 3 and all(b["plan"] == pid for b in bundles)


def test_links_require_dense_component(tmp_path, fixture_dir):
    import pytest

    with pytest.raises(FileNotFoundError, match="--plans dense"):
        indexing.main(
            [
                "--dataset",
                str(fixture_dir),
                "--index-dir",
                str(tmp_path / "ix"),
                "--encoder",
                "stub",
                "--plans",
                "links",
            ]
        )
