"""Deterministic, idempotent reducers for LangGraph state channels.

Reducers in this module accept both the typed contracts introduced by the
refactor and legacy dictionaries still produced by the current graph.  New
records replace records with the same stable identifier.  The final ordering is
derived from identifiers rather than task completion order, which makes merges
from parallel branches reproducible.

An empty list cannot reset a reducer-backed LangGraph channel.  The explicit
``reset_accumulator`` update is therefore used by the lifecycle node when a new
research task starts in an existing conversation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, Generic, TypeVar

from pydantic import BaseModel


RecordT = TypeVar("RecordT")
RESET_MARKER = "__athena_reset_accumulator__"


class AccumulatorReset(dict[str, Any], Generic[RecordT]):
    """Serializable reducer update that replaces, rather than appends."""


def reset_accumulator(
    items: Iterable[RecordT] = (),
) -> AccumulatorReset[RecordT]:
    """Return an explicit, checkpoint-safe reset update for a channel."""

    return AccumulatorReset({RESET_MARKER: True, "items": list(items)})


def _is_reset(value: object) -> bool:
    return isinstance(value, Mapping) and value.get(RESET_MARKER) is True


def _records(value: object) -> list[Any]:
    if value is None:
        return []
    if _is_reset(value):
        items = value.get("items", [])  # type: ignore[union-attr]
        return list(items) if isinstance(items, Iterable) else []
    if isinstance(value, (str, bytes, bytearray)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _field(record: object, name: str) -> object | None:
    if isinstance(record, BaseModel):
        return getattr(record, name, None)
    if isinstance(record, Mapping):
        return record.get(name)
    return None


def _canonical_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_canonical_value(item) for item in value]
    return value


def _fallback_identifier(record: object) -> str:
    """Give legacy records a stable content identity during the transition."""

    canonical = json.dumps(
        _canonical_value(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "legacy:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_string(record: object) -> str:
    return json.dumps(
        _canonical_value(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _numeric_field(record: object, name: str) -> int:
    value = _field(record, name)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return 0


def _record_rank(record: object) -> tuple[int, int, int, str]:
    """Resolve same-ID conflicts without depending on delivery order.

    Producers may use ``revision`` for explicit replacement.  ``iteration`` and
    action status provide useful monotonic fallbacks for current contracts.  A
    canonical representation is the final deterministic tie-breaker for
    immutable evidence/facts that should not normally conflict.
    """

    status_order = {
        "planned": 0,
        "running": 1,
        "skipped": 2,
        "unsupported": 2,
        "failed": 3,
        "succeeded": 3,
    }
    status = str(_field(record, "status") or "")
    return (
        _numeric_field(record, "revision"),
        _numeric_field(record, "iteration"),
        status_order.get(status, 0),
        _canonical_string(record),
    )


def _identifier(record: object, id_field: str) -> str:
    value = _field(record, id_field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return _fallback_identifier(record)


def merge_records_by_id(
    current: Iterable[RecordT] | None,
    update: Iterable[RecordT] | AccumulatorReset[RecordT] | None,
    *,
    id_field: str,
) -> list[RecordT]:
    """Merge records by ID, replacing duplicates and sorting deterministically."""

    base = [] if _is_reset(update) else _records(current)
    incoming = _records(update)
    merged: dict[str, RecordT] = {}
    for record in [*base, *incoming]:
        identifier = _identifier(record, id_field)
        previous = merged.get(identifier)
        if previous is None or _record_rank(record) > _record_rank(previous):
            merged[identifier] = record
    return [merged[key] for key in sorted(merged)]


def merge_actions(current: object, update: object) -> list[Any]:
    return merge_records_by_id(current, update, id_field="action_id")


def merge_evidence(current: object, update: object) -> list[Any]:
    return merge_records_by_id(current, update, id_field="evidence_id")


def merge_facts(current: object, update: object) -> list[Any]:
    return merge_records_by_id(current, update, id_field="fact_id")


def merge_branch_results(current: object, update: object) -> list[Any]:
    return merge_records_by_id(current, update, id_field="branch_id")


def merge_unique_text(current: object, update: object) -> list[str]:
    """Merge textual observations without duplicates or completion-order drift."""

    base = [] if _is_reset(update) else _records(current)
    incoming = _records(update)
    values = {str(item).strip() for item in [*base, *incoming] if str(item).strip()}
    return sorted(values)


__all__ = [
    "AccumulatorReset",
    "RESET_MARKER",
    "merge_actions",
    "merge_branch_results",
    "merge_evidence",
    "merge_facts",
    "merge_records_by_id",
    "merge_unique_text",
    "reset_accumulator",
]
