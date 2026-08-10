"""Concurrency-safe research budgets and canonical action signatures."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel


class BudgetExceeded(RuntimeError):
    def __init__(self, limit: str, message: str) -> None:
        super().__init__(message)
        self.limit = limit


def _canonical(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        canonical_items = [_canonical(item) for item in value]
        if all(not isinstance(item, (dict, list)) for item in canonical_items):
            return sorted(canonical_items, key=lambda item: json.dumps(item, default=str))
        return canonical_items
    if isinstance(value, (set, frozenset)):
        return sorted((_canonical(item) for item in value), key=lambda item: str(item))
    if isinstance(value, str):
        return value.strip()
    return value


def action_signature(tool: str, params: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"tool": tool.strip(), "params": _canonical(params)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    max_branches: int = 4
    max_actions_per_branch: int = 16
    max_requests_per_iteration: int = 4
    max_external_calls: int = 32
    max_wall_time_seconds: float = 120.0
    max_evidence_items: int = 100
    max_context_chars: int = 300_000

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "BudgetLimits":
        source = values or {}
        defaults = cls()
        legacy_actions = int(
            source.get("max_actions", defaults.max_actions_per_branch)
        )
        return cls(
            max_branches=int(source.get("max_branches", defaults.max_branches)),
            max_actions_per_branch=int(
                source.get("max_actions_per_branch", legacy_actions)
            ),
            max_requests_per_iteration=int(
                source.get(
                    "max_requests_per_iteration",
                    defaults.max_requests_per_iteration,
                )
            ),
            max_external_calls=int(
                source.get("max_external_calls", defaults.max_external_calls)
            ),
            max_wall_time_seconds=float(
                source.get(
                    "max_wall_time_seconds", defaults.max_wall_time_seconds
                )
            ),
            max_evidence_items=int(
                source.get("max_evidence_items", defaults.max_evidence_items)
            ),
            max_context_chars=int(
                source.get("max_context_chars", defaults.max_context_chars)
            ),
        )

    def __post_init__(self) -> None:
        for name in (
            "max_branches",
            "max_actions_per_branch",
            "max_requests_per_iteration",
            "max_external_calls",
            "max_evidence_items",
            "max_context_chars",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_wall_time_seconds <= 0:
            raise ValueError("max_wall_time_seconds must be positive")


class BudgetTracker:
    """Reserve scarce resources atomically across concurrent actions."""

    def __init__(
        self,
        limits: BudgetLimits,
        *,
        existing_external_calls: int = 0,
        existing_branch_actions: int = 0,
        existing_evidence_items: int = 0,
        existing_branches: int = 1,
        existing_context_chars: int = 0,
        started_at: float | None = None,
        clock: Any = time.time,
    ) -> None:
        self.limits = limits
        self._external_calls = existing_external_calls
        self._branch_actions = existing_branch_actions
        self._branches = existing_branches
        self._iteration_requests = 0
        self._evidence_items = existing_evidence_items
        self._context_chars = existing_context_chars
        self._clock = clock
        self._started_at = started_at if started_at is not None else clock()
        self._lock = asyncio.Lock()

    @property
    def external_calls(self) -> int:
        return self._external_calls

    @property
    def usage(self) -> dict[str, int]:
        return {
            "branches": self._branches,
            "actions": self._branch_actions,
            "external_calls": self._external_calls,
            "evidence_items": self._evidence_items,
            "context_chars": self._context_chars,
        }

    def remaining_wall_time(self) -> float:
        return self.limits.max_wall_time_seconds - (self._clock() - self._started_at)

    async def reserve_action(self, *, external: bool = True) -> None:
        async with self._lock:
            if self.remaining_wall_time() <= 0:
                raise BudgetExceeded("max_wall_time_seconds", "Research wall-time budget exhausted.")
            if self._branch_actions >= self.limits.max_actions_per_branch:
                raise BudgetExceeded("max_actions_per_branch", "Branch action budget exhausted.")
            if external and self._iteration_requests >= self.limits.max_requests_per_iteration:
                raise BudgetExceeded("max_requests_per_iteration", "Iteration request budget exhausted.")
            if external and self._external_calls >= self.limits.max_external_calls:
                raise BudgetExceeded("max_external_calls", "External-call budget exhausted.")
            self._branch_actions += 1
            if external:
                self._iteration_requests += 1
                self._external_calls += 1

    async def reserve_branch(self, count: int = 1) -> None:
        async with self._lock:
            if count < 0 or self._branches + count > self.limits.max_branches:
                raise BudgetExceeded("max_branches", "Research branch budget exhausted.")
            self._branches += count

    async def reserve_context(self, chars: int) -> None:
        async with self._lock:
            if chars < 0 or self._context_chars + chars > self.limits.max_context_chars:
                raise BudgetExceeded("max_context_chars", "Research context budget exhausted.")
            self._context_chars += chars

    async def reserve_evidence(self, count: int = 1, *, context_chars: int = 0) -> None:
        async with self._lock:
            if self._evidence_items + count > self.limits.max_evidence_items:
                raise BudgetExceeded("max_evidence_items", "Evidence item budget exhausted.")
            if self._context_chars + context_chars > self.limits.max_context_chars:
                raise BudgetExceeded("max_context_chars", "Research context budget exhausted.")
            self._evidence_items += count
            self._context_chars += context_chars


__all__ = [
    "BudgetExceeded",
    "BudgetLimits",
    "BudgetTracker",
    "action_signature",
]
