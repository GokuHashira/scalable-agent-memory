"""Pins the shared contract (slice §0, data-model.md) so drift from K's schema is caught immediately."""

from dataclasses import fields

from mem27 import schema


def _names(cls) -> list[str]:
    return [f.name for f in fields(cls)]


def test_note_fields():
    assert _names(schema.Note) == [
        "note_id",
        "history_id",
        "session_id",
        "turn_idx",
        "role",
        "text",
        "ts",
        "valid_to",
        "version",
        "supersedes_note_id",
        "metadata",
        "entity_tags",
        "summary_id",
        "links",
    ]


def test_summary_fields():
    assert _names(schema.Summary) == [
        "summary_id",
        "history_id",
        "session_id",
        "text",
        "source_note_ids",
        "created_ts",
        "stale",
    ]


def test_question_fields():
    assert _names(schema.Question) == [
        "question_id",
        "history_id",
        "question_type",
        "question",
        "question_date",
        "answer",
        "gold_session_ids",
        "gold_note_ids",
        "split",
    ]


def test_update_fields():
    assert _names(schema.Update) == ["update_id", "ts", "op", "note", "supersedes_note_id", "explicit"]


def test_request_fields_and_defaults():
    assert _names(schema.RetrievalRequest) == [
        "query",
        "budget_ms",
        "budget_tokens",
        "max_results",
        "as_of_time",
        "plan",
        "question_id",
    ]
    req = schema.RetrievalRequest(query="q")
    assert req.budget_tokens == 2048
    assert req.max_results == 20


def test_bundle_fields():
    assert _names(schema.EvidenceBundle)[:7] == [
        "note_ids",
        "passages",
        "scores",
        "total_tokens",
        "latency_ms",
        "stale_flags",
        "trace",
    ]
    assert {"question_id", "plan"} <= set(_names(schema.EvidenceBundle))


def test_retriever_protocol_methods():
    for method in ("build", "insert", "retrieve"):
        assert hasattr(schema.Retriever, method)


def test_note_roundtrip_through_dict():
    note = schema.Note(
        note_id="h:s:0",
        history_id="h",
        session_id="s",
        turn_idx=0,
        role="user",
        text="hello",
        ts="2023-05-20T02:21:00",
        links=[schema.Link(type="temporal", target_note_id="h:s:1", weight=1.0)],
    )
    again = schema.Note.from_dict(schema.to_dict(note))
    assert again == note


def test_token_helpers():
    assert schema.count_tokens("hello world") == 2
    kept = schema.truncate_to_budget(["hello world", "hello world", "hello world"], budget_tokens=5)
    assert kept == ["hello world", "hello world"]
