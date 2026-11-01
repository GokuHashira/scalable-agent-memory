"""Plan names (used for CLI, folders) ↔ research plan ids (used inside records)."""

from pathlib import Path

import pytest

from mem27.retrieval.plans import PLAN_NAMES, plan_id, plan_name
from mem27.retrieval.runner import default_run_dir


def test_names_and_ids_roundtrip():
    assert PLAN_NAMES == {
        "p0": "bm25",
        "p1": "dense",
        "p2": "hybrid",
        "p3": "summary_first",
        "p4": "one_hop",
        "p5": "bounded",
    }
    for pid, name in PLAN_NAMES.items():
        assert plan_id(name) == pid and plan_id(pid) == pid
        assert plan_name(pid) == name and plan_name(name) == name


def test_unknown_plan_is_rejected_with_choices():
    with pytest.raises(ValueError, match="bm25, dense, hybrid"):
        plan_id("p9")


def test_run_dir_uses_descriptive_name():
    assert default_run_dir("p2", Path("data/lme_s_raw"), 0, "warm") == Path("runs/hybrid-lme_s_raw-seed0-warm")
