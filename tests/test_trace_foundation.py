"""Foundation checks for local semantic tracing and safe logging."""

from __future__ import annotations

import json
import sqlite3

import pytest

from utils.logger import mask_sensitive
from utils.trace import configure_trace, disable_trace, emit_trace, flush_traces


def test_trace_migrates_legacy_schema_flushes_and_redacts(tmp_path) -> None:
    database = tmp_path / "legacy-traces.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                branch_id TEXT,
                iteration INTEGER,
                node TEXT,
                type TEXT,
                content TEXT,
                ts REAL
            )
            """
        )
        connection.commit()

    streamed = []
    configure_trace(database)
    try:
        emit_trace(
            node="planner",
            event_type="reasoning_summary",
            content={
                "summary": "Проверены варианты",
                "api_key": "must-not-survive",
                "chain_of_thought": "must-not-survive",
            },
            thread_id="thread-1",
            turn_id="turn-1",
            research_id="research-1",
            run_id="run-1",
            branch_id="branch-1",
            batch_id="batch-1",
            action_id="action-1",
            iteration=3,
            public=True,
            stream_writer=streamed.append,
        )
        assert flush_traces(timeout=2.0)

        with sqlite3.connect(database) as connection:
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(traces)")
            }
            row = connection.execute(
                """
                SELECT thread_id, turn_id, research_id, run_id, branch_id,
                       batch_id, action_id, iteration, event_type, content
                FROM traces ORDER BY id DESC LIMIT 1
                """
            ).fetchone()

        assert {
            "thread_id",
            "turn_id",
            "research_id",
            "batch_id",
            "action_id",
            "event_type",
            "timestamp",
        } <= columns
        assert row[:9] == (
            "thread-1",
            "turn-1",
            "research-1",
            "run-1",
            "branch-1",
            "batch-1",
            "action-1",
            3,
            "reasoning_summary",
        )
        persisted = json.loads(row[9])
        assert persisted["summary"] == "Проверены варианты"
        assert persisted["api_key"] == "[REDACTED]"
        assert persisted["chain_of_thought"] == "[REDACTED]"
        assert streamed[0]["content"]["api_key"] == "[REDACTED]"
    finally:
        assert disable_trace(timeout=2.0)


def test_trace_rejects_legacy_thought_event() -> None:
    with pytest.raises(ValueError, match="unsupported trace event type"):
        emit_trace(
            node="planner",
            event_type="thought",  # type: ignore[arg-type]
            content="raw reasoning must not be accepted",
            run_id="run-1",
        )


def test_logger_masks_credentials_in_text_and_context() -> None:
    assert mask_sensitive("Authorization: Bearer very-secret") == (
        "Authorization: Bearer [REDACTED]"
    )
    assert mask_sensitive({"token": "secret", "run_id": "run-1"}) == {
        "token": "[REDACTED]",
        "run_id": "run-1",
    }


def test_terminal_node_persists_completion_event(tmp_path) -> None:
    from research.assistant import assistant_node

    database = tmp_path / "completion.db"
    configure_trace(database)
    try:
        result = assistant_node(
            {
                "thread_id": "thread-1",
                "turn_id": "turn-1",
                "research_id": "research-1",
                "run_id": "run-1",
                "branch_id": "branch-1",
                "batch_id": "batch-1",
                "iteration_count": 2,
                "research_status": "error",
                "termination_reason": "error",
                "error": {
                    "kind": "model",
                    "message": "Безопасная терминальная ошибка",
                },
            },
            config={},
        )
        assert flush_traces(timeout=2.0)
        with sqlite3.connect(database) as connection:
            event = connection.execute(
                """
                SELECT event_type, research_id, run_id, iteration, content
                FROM traces ORDER BY id DESC LIMIT 1
                """
            ).fetchone()

        assert result["research_status"] == "error"
        assert event[:4] == ("completion", "research-1", "run-1", 2)
        assert json.loads(event[4]) == {
            "status": "error",
            "termination_reason": "error",
        }
    finally:
        assert disable_trace(timeout=2.0)
