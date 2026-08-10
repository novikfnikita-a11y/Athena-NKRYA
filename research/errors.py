"""Deterministic terminal rendering for safe graph errors."""

from __future__ import annotations

import re
from typing import Any

from research.assistant import format_facts
from state.models import ErrorKind
from state.schema import ResearchState


_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)(?:api[_-]?key|token|secret)\s*[:=]\s*\S+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)


def _sanitize(message: str) -> str:
    if "traceback (most recent call last)" in message.lower():
        return "Внутренняя ошибка выполнения без доступных безопасных деталей."
    safe = message.replace("\n", " ").replace("\r", " ").strip()
    for pattern in _SECRET_PATTERNS:
        safe = pattern.sub("[REDACTED]", safe)
    return safe[:1_000] or "Исследование завершилось с контролируемой ошибкой."


def error_node(state: ResearchState) -> dict[str, Any]:
    raw = state.get("error") or {}
    kind_value = raw.get("kind") or ErrorKind.INTERNAL.value
    raw_kind = (
        kind_value.value if isinstance(kind_value, ErrorKind) else str(kind_value)
    )
    kind = raw_kind if raw_kind in {item.value for item in ErrorKind} else "internal"
    message = _sanitize(str(raw.get("message") or "Контролируемая ошибка."))
    run_id = str(state.get("run_id") or "не назначен")
    facts = list(state.get("facts", []))

    parts = [
        f"Исследование завершилось с ошибкой типа `{kind}`: {message}",
        f"Идентификатор запуска для диагностики: `{run_id}`.",
    ]
    if facts:
        parts.extend(
            [
                "До ошибки были получены частичные проверяемые результаты:",
                format_facts(facts),
            ]
        )
    return {
        "final_response": "\n\n".join(parts),
        "research_status": "error",
        "termination_reason": "error",
        "error": {
            "kind": kind,
            "message": message,
            "node": raw.get("node", "unknown"),
            "retryable": raw.get("retryable") is True,
        },
    }


def budget_terminal_node(state: ResearchState) -> dict[str, Any]:
    """Record deterministic partial completion when iteration budget ends."""

    return {
        "research_status": "partial",
        "termination_reason": "budget_exhausted",
        "is_goal_reached": False,
        "needs_replanning": False,
        "aggregator_status": "complete",
        "aggregator_reasoning": (
            f"Исчерпан лимит итераций: {state.get('iteration_count', 0)}."
        ),
        "error": None,
    }


__all__ = ["budget_terminal_node", "error_node"]
