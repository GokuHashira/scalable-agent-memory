"""Reader wrapper, judge parsing and leakage helpers."""
import pytest

from mem27.eval.common import days_before, format_history
from mem27.eval.freshness import contains, synth_correct
from mem27.eval.judge import judge_prompt, load_judge_prompts, parse_label
from mem27.eval.leakage import binom_sf
from mem27.eval.reader import extract_final
from mem27.vschema import Question


def test_days_before():
    assert days_before("2023-05-01T10:00:00", "2023-05-24T09:00:00") == ", 23 days before the current date"
    assert days_before("2023-05-23T10:00:00", "2023-05-24T09:00:00") == ", 1 day before the current date"
    assert days_before("2023-05-24T01:00:00", "2023-05-24T09:00:00") == ", same day as the current date"
    assert days_before("", "2023-05-24") == ""


def test_format_history_groups_and_orders():
    h = format_history(["q:s2:0", "q:s1:3", "q:s1:1"], ["c", "b", "a"], None, "chrono")
    assert h.index("user: a") < h.index("user: b")  # turns in order inside a session
    assert "...\nuser: b" in h  # gap marker between non-adjacent turns
    r = format_history(["q:s2:0", "q:s1:3"], ["c", "b"], None, "rank")
    assert r.index("Excerpt 1") < r.index("Excerpt 2") and r.index("c") < r.index("b")


def test_extract_final():
    assert extract_final("step 1...\nFinal answer: 42 days") == "42 days"
    assert extract_final("**Final Answer:** Paris") == "Paris"
    assert extract_final("just text") == "just text"


def test_judge_prompt_selection():
    p = load_judge_prompts("configs/prompts/judge.txt")
    assert set(p) >= {"default", "temporal-reasoning", "knowledge-update", "single-session-preference", "abstention"}
    q = Question("x_abs", "x", "multi-session", "Q?", "", "A", [], [], "test")
    assert "unanswerable" in judge_prompt(p, q, "resp")
    q2 = Question("y", "y", "temporal-reasoning", "Q?", "", "A", [], [], "test")
    assert "off-by-one" in judge_prompt(p, q2, "resp")
    assert parse_label("Yes.") and not parse_label("No") and parse_label("<think>no</think> yes")


def test_string_judge_and_binom():
    assert contains("She works at Korvath Labs now.", "Korvath Labs")
    assert not contains("inactive", "active")
    q = {"question_type": "what_changed", "answer": "from A1 to B2", "fact": {"old_value": "Velmora", "new_value": "Quandt"}}
    assert synth_correct("It went from Velmora to Quandt.", q) and not synth_correct("Quandt", q)
    assert binom_sf(0, 10, 0.1) == 1.0
    assert binom_sf(10, 10, 0.5) == pytest.approx(0.5 ** 10)
