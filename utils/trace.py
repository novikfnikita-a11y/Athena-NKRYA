"""Local semantic tracing with lazy SQLite persistence and safe streaming."""

from __future__ import annotations

import atexit
import json
import os
import queue
import re
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal


TraceEventType = Literal[
    "planning",
    "action",
    "observation",
    "decision",
    "completion",
    "final_answer",
    "error",
    "reasoning_summary",
]
StreamWriter = Callable[[dict[str, Any]], Any]
_ALLOWED_EVENT_TYPES = {
    "planning",
    "action",
    "observation",
    "decision",
    "completion",
    "final_answer",
    "error",
    "reasoning_summary",
}

_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|token|chain[_-]?of[_-]?thought)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(
    r"(?i)(bearer\s+)[^\s,;]+|"
    r"((?:api[_-]?key|password|secret|token)\s*[:=]\s*)[^\s,;]+",
)


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _SENSITIVE_VALUE.sub(lambda match: (match.group(1) or match.group(2)) + "[REDACTED]", value)
    return value


def _serialize_content(content: Any) -> str:
    safe = _redact(content)
    if isinstance(safe, str):
        return safe
    return json.dumps(safe, ensure_ascii=False, sort_keys=True, default=str)


class TraceStore:
    """A lazily started, flushable SQLite trace writer."""

    def __init__(self, db_path: str | os.PathLike[str], queue_size: int = 1000) -> None:
        self.db_path = Path(db_path)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._start_lock = threading.Lock()
        self._ready = threading.Event()
        self._closed = False
        self._last_error: BaseException | None = None

    @property
    def last_error(self) -> BaseException | None:
        return self._last_error

    def _start(self) -> None:
        if self._thread is not None:
            return
        with self._start_lock:
            if self._thread is not None:
                return
            if self._closed:
                raise RuntimeError("TraceStore is already closed")
            self._thread = threading.Thread(
                target=self._worker,
                name="athena-trace-writer",
                daemon=True,
            )
            self._thread.start()
        self._ready.wait(timeout=5.0)
        if self._last_error is not None:
            raise RuntimeError("Trace storage initialization failed") from self._last_error

    def _migrate(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                turn_id TEXT,
                research_id TEXT,
                run_id TEXT NOT NULL,
                branch_id TEXT,
                batch_id TEXT,
                action_id TEXT,
                iteration INTEGER NOT NULL DEFAULT 0,
                node TEXT NOT NULL,
                event_type TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                public INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(traces)").fetchall()
        }
        additions = {
            "thread_id": "TEXT",
            "turn_id": "TEXT",
            "research_id": "TEXT",
            "run_id": "TEXT",
            "branch_id": "TEXT",
            "batch_id": "TEXT",
            "action_id": "TEXT",
            "iteration": "INTEGER NOT NULL DEFAULT 0",
            "node": "TEXT",
            "event_type": "TEXT",
            "content": "TEXT",
            "timestamp": "REAL",
            "public": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, declaration in additions.items():
            if name not in columns:
                connection.execute(
                    f'ALTER TABLE traces ADD COLUMN "{name}" {declaration}'
                )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(traces)").fetchall()
        }
        if "type" in columns:
            connection.execute(
                "UPDATE traces SET event_type = COALESCE(event_type, type)"
            )
        if "ts" in columns:
            connection.execute(
                "UPDATE traces SET timestamp = COALESCE(timestamp, ts)"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_research ON traces(research_id, run_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_traces_thread_time ON traces(thread_id, timestamp)"
        )
        connection.commit()

    def _worker(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as connection:
                self._migrate(connection)
                self._ready.set()
                while True:
                    event = self._queue.get()
                    try:
                        if event is None:
                            return
                        connection.execute(
                            """
                            INSERT INTO traces (
                                thread_id, turn_id, research_id, run_id,
                                branch_id, batch_id, action_id, iteration,
                                node, event_type, content, timestamp, public
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                event.get("thread_id"),
                                event.get("turn_id"),
                                event.get("research_id"),
                                event["run_id"],
                                event.get("branch_id"),
                                event.get("batch_id"),
                                event.get("action_id"),
                                event["iteration"],
                                event["node"],
                                event["event_type"],
                                event["content"],
                                event["timestamp"],
                                int(event["public"]),
                            ),
                        )
                        connection.commit()
                    except BaseException as error:  # keep the worker alive for later events
                        self._last_error = error
                    finally:
                        self._queue.task_done()
        except BaseException as error:
            self._last_error = error
            self._ready.set()

    def emit(self, event: dict[str, Any]) -> None:
        self._start()
        if self._last_error is not None:
            raise RuntimeError("Trace storage is unavailable") from self._last_error
        try:
            self._queue.put_nowait(event)
        except queue.Full as error:
            raise RuntimeError("Trace queue is full; event was not persisted") from error

    def flush(self, timeout: float = 5.0) -> bool:
        if self._thread is None:
            return True
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return self._last_error is None

    def close(self, timeout: float = 5.0) -> bool:
        if self._closed:
            return self._last_error is None
        flushed = self.flush(timeout=timeout)
        self._closed = True
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=timeout)
        return flushed and self._last_error is None


_store_lock = threading.Lock()
_store: TraceStore | None = None
_enabled = os.environ.get("ATHENA_TRACE_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _default_db_path() -> Path:
    configured = os.environ.get("ATHENA_TRACE_DB")
    if configured:
        return Path(configured).expanduser()
    return Path.cwd() / "research_traces.db"


def get_trace_store() -> TraceStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = TraceStore(_default_db_path())
    return _store


def configure_trace(db_path: str | os.PathLike[str]) -> TraceStore:
    """Replace the lazy store, primarily for application startup and tests."""

    global _enabled, _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = TraceStore(db_path)
        _enabled = True
        return _store


def emit_trace(
    node: str,
    event_type: TraceEventType,
    content: Any,
    *,
    run_id: str,
    thread_id: str | None = None,
    turn_id: str | None = None,
    research_id: str | None = None,
    branch_id: str | None = None,
    batch_id: str | None = None,
    action_id: str | None = None,
    iteration: int = 0,
    public: bool = False,
    stream_writer: StreamWriter | None = None,
) -> dict[str, Any]:
    """Persist one semantic event and optionally expose its safe public subset."""

    if not run_id or not str(run_id).strip():
        raise ValueError("run_id is required for every trace event")
    if event_type not in _ALLOWED_EVENT_TYPES:
        raise ValueError(f"unsupported trace event type: {event_type!r}")
    if event_type == "reasoning_summary":
        if not isinstance(content, Mapping):
            raise ValueError("reasoning_summary content must be a mapping")
        summary = content.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("reasoning_summary requires a non-empty summary")
    event = {
        "thread_id": thread_id,
        "turn_id": turn_id,
        "research_id": research_id,
        "run_id": str(run_id),
        "branch_id": branch_id,
        "batch_id": batch_id,
        "action_id": action_id,
        "iteration": max(0, int(iteration)),
        "node": str(node),
        "event_type": event_type,
        "content": _serialize_content(content),
        "timestamp": time.time(),
        "public": bool(public),
    }
    if _enabled or _store is not None:
        get_trace_store().emit(event)
    if public and stream_writer is not None:
        stream_writer(
            {
                "type": event_type,
                "node": event["node"],
                "research_id": research_id,
                "run_id": event["run_id"],
                "iteration": event["iteration"],
                "content": _redact(content),
            }
        )
    return event


def flush_traces(timeout: float = 5.0) -> bool:
    return True if _store is None else _store.flush(timeout=timeout)


def close_trace(timeout: float = 5.0) -> bool:
    return True if _store is None else _store.close(timeout=timeout)


def disable_trace(timeout: float = 5.0) -> bool:
    """Flush and detach the process-wide store without creating a replacement."""

    global _enabled, _store
    with _store_lock:
        result = True if _store is None else _store.close(timeout=timeout)
        _store = None
        _enabled = False
        return result


def _shutdown() -> None:
    close_trace(timeout=5.0)


atexit.register(_shutdown)


__all__ = [
    "TraceEventType",
    "TraceStore",
    "close_trace",
    "configure_trace",
    "disable_trace",
    "emit_trace",
    "flush_traces",
    "get_trace_store",
]
