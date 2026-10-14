"""End-to-end smoke test of build → run on the fixture (FR-016)."""

import json

import yaml

from mem27.retrieval import indexing, runner
from mem27.schema import read_jsonl


def test_build_then_run_writes_run_directory(tmp_path, fixture_dir):
    index_dir = tmp_path / "index"
    indexing.main(
        ["--dataset", str(fixture_dir), "--index-dir", str(index_dir), "--plans", "p0,p1", "--encoder", "stub"]
    )
    profile = json.loads((index_dir / "build_profile.json").read_text())
    assert {"p0", "p1"} <= set(profile["totals"])
    out = tmp_path / "runs" / "p2-fixture-seed0-warm"
    runner.main(
        [
            "--plan", "p2",
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
