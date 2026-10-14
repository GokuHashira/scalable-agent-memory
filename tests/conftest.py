"""Shared fixtures: the offline fixture dataset, a stub encoder, and the pinned config."""

from pathlib import Path

import pytest
import yaml

from mem27.retrieval.encoder import StubEncoder
from mem27.retrieval.fixture import write_fixture
from mem27.schema import load_notes, load_questions

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "data" / "fixture"


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    if not (FIXTURE_DIR / "notes.jsonl").exists():
        write_fixture(FIXTURE_DIR, n_turn_notes=5000, seed=0)
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def notes(fixture_dir):
    return load_notes(fixture_dir / "notes.jsonl")


@pytest.fixture(scope="session")
def questions(fixture_dir):
    return load_questions(fixture_dir / "questions.jsonl")


@pytest.fixture(scope="session")
def notes_by_history(notes):
    grouped: dict[str, list] = {}
    for note in notes:
        grouped.setdefault(note.history_id, []).append(note)
    return grouped


@pytest.fixture(scope="session")
def config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "retrieval.yaml").read_text())


@pytest.fixture(scope="session")
def encoder() -> StubEncoder:
    return StubEncoder()
