"""Typed aggregation of exactly one explicit, non-empty evidence batch."""

import asyncio
import json
from typing import Any, Mapping

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from LLM.client import get_llm
from LLM.prompts import EVIDENCE_AGGREGATOR_SYSTEM_PROMPT
from LLM.structured import (
    AggregatorOutput,
    StructuredOutputError,
    ainvoke_structured,
)
from state.models import (
    AggregatorDecision,
    AggregatorStatus,
    ErrorKind,
    TerminationReason,
)
from state.schema import ResearchState
from tools.budgets import action_signature


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value if isinstance(value, Mapping) else {}


def _evidence_id(item: Mapping[str, Any], index: int) -> str:
    del index
    evidence_id = item.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        raise ValueError("every current artifact requires evidence_id")
    return evidence_id


def _format_artifact(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    return {
        "evidence_id": _evidence_id(item, index),
        "research_id": item.get("research_id"),
        "run_id": item.get("run_id"),
        "branch_id": item.get("branch_id"),
        "batch_id": item.get("batch_id"),
        "action_id": item.get("action_id"),
        "tool": item.get("tool") or item.get("action") or "unknown_action",
        "params": item.get("params", {}),
        "status": item.get("status", "success"),
        "message": item.get("message"),
        "payload": item.get("payload", item.get("response")),
        "raw_response_hash": item.get("raw_response_hash"),
    }


def _completed_by_id(state: ResearchState) -> dict[str, Mapping[str, Any]]:
    completed: dict[str, Mapping[str, Any]] = {}
    for record in state.get("completed_actions", []):
        action = _as_mapping(record)
        action_id = action.get("action_id")
        if isinstance(action_id, str) and action_id:
            if action_id in completed:
                raise ValueError("completed actions contain duplicate action_id")
            completed[action_id] = action
    return completed


def _validate_artifact_link(
    state: ResearchState,
    item: Mapping[str, Any],
    completed: Mapping[str, Mapping[str, Any]],
    *,
    expected_batch_id: str | None = None,
) -> None:
    evidence_id = _evidence_id(item, 0)
    for name in ("research_id", "run_id", "branch_id"):
        expected_value = state.get(name)
        if not isinstance(expected_value, str) or not expected_value:
            raise ValueError("current research scope is incomplete")
        if item.get(name) != expected_value:
            raise ValueError(f"artifact {evidence_id} has foreign {name}")
    artifact_batch_id = item.get("batch_id")
    if not isinstance(artifact_batch_id, str) or not artifact_batch_id:
        raise ValueError(f"artifact {evidence_id} requires batch_id")
    if expected_batch_id is not None and artifact_batch_id != expected_batch_id:
        raise ValueError(f"artifact {evidence_id} has foreign batch_id")

    action_id = item.get("action_id")
    if not isinstance(action_id, str) or not action_id:
        raise ValueError(f"artifact {evidence_id} requires action_id")
    action = completed.get(action_id)
    if action is None:
        raise ValueError(f"artifact {evidence_id} is not linked to a completed action")
    for name in ("research_id", "run_id", "branch_id", "batch_id"):
        if action.get(name) != item.get(name):
            raise ValueError(f"completed action {action_id} has foreign {name}")
    if action.get("status") not in {"succeeded", "failed", "skipped", "unsupported"}:
        raise ValueError(f"action {action_id} is not terminal")

    artifact_tool = item.get("tool") or item.get("action")
    action_tool = action.get("tool") or action.get("action")
    if not isinstance(artifact_tool, str) or artifact_tool != action_tool:
        raise ValueError(f"artifact {evidence_id} tool does not match its action")
    artifact_params = item.get("params")
    action_params = action.get("params")
    if not isinstance(artifact_params, Mapping):
        raise ValueError(f"artifact {evidence_id} params must be an object")
    if not isinstance(action_params, Mapping):
        raise ValueError(f"action {action_id} params must be an object")

    artifact_signature = action_signature(artifact_tool, artifact_params)
    accepted_signatures = {action_signature(artifact_tool, action_params)}
    original_signature = action.get("signature")
    if isinstance(original_signature, str) and original_signature:
        accepted_signatures.add(original_signature)
    if artifact_signature not in accepted_signatures:
        raise ValueError(f"artifact {evidence_id} params do not match its action")

    pinned_corpus = state.get("recommended_corpus")
    if pinned_corpus:
        if artifact_params.get("corpus") not in (None, pinned_corpus):
            raise ValueError(f"artifact {evidence_id} escaped the pinned corpus")
        if action_params.get("corpus") not in (None, pinned_corpus):
            raise ValueError(f"action {action_id} escaped the pinned corpus")


def _validate_current_batch(
    state: ResearchState,
    batch: list[Mapping[str, Any]],
) -> None:
    expected_batch_id = state.get("batch_id")
    if not isinstance(expected_batch_id, str) or not expected_batch_id:
        raise ValueError("current research scope is incomplete")
    completed = _completed_by_id(state)

    seen_evidence_ids: set[str] = set()
    for index, item in enumerate(batch, 1):
        evidence_id = _evidence_id(item, index)
        if evidence_id in seen_evidence_ids:
            raise ValueError("current batch contains duplicate evidence_id")
        seen_evidence_ids.add(evidence_id)
        _validate_artifact_link(
            state,
            item,
            completed,
            expected_batch_id=expected_batch_id,
        )


def _select_batch(
    batch: list[Mapping[str, Any]],
    max_chars: int,
) -> tuple[list[dict[str, Any]], bool]:
    selected: list[dict[str, Any]] = []
    used = 0
    for index, item in enumerate(batch, 1):
        formatted = _format_artifact(item, index)
        encoded = json.dumps(formatted, ensure_ascii=False, default=str)
        if selected and used + len(encoded) > max_chars:
            return selected, True
        if not selected and len(encoded) > max_chars:
            # Preserve identity/status/message even when the first payload alone
            # exceeds the context budget.
            formatted["payload"] = "[PAYLOAD_TRUNCATED_BY_CONTEXT_BUDGET]"
            selected.append(formatted)
            return selected, True
        selected.append(formatted)
        used += len(encoded)
    return selected, False


def _validated_facts(
    state: ResearchState,
    current_batch: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    completed = _completed_by_id(state)
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for record in [*state.get("evidence", []), *current_batch]:
        artifact = _as_mapping(record)
        evidence_id = artifact.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            continue
        existing = evidence_by_id.get(evidence_id)
        if existing is not None and dict(existing) != dict(artifact):
            raise ValueError("evidence_id resolves to conflicting artifacts")
        evidence_by_id[evidence_id] = artifact

    facts: list[dict[str, Any]] = []
    seen_fact_ids: set[str] = set()
    for record in state.get("facts", []):
        item = dict(_as_mapping(record))
        fact_id = item.get("fact_id")
        if not isinstance(fact_id, str) or not fact_id or fact_id in seen_fact_ids:
            raise ValueError("facts require unique fact_id values")
        seen_fact_ids.add(fact_id)
        for name in ("research_id", "run_id", "branch_id"):
            if item.get(name) != state.get(name):
                raise ValueError(f"fact {fact_id} has foreign {name}")
        evidence_id = item.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError(f"fact {fact_id} requires evidence_id")
        artifact = evidence_by_id.get(evidence_id)
        if artifact is None:
            raise ValueError(f"fact {fact_id} is not linked to evidence")
        _validate_artifact_link(state, artifact, completed)
        if item.get("batch_id") != artifact.get("batch_id"):
            raise ValueError(f"fact {fact_id} batch_id does not match its evidence")
        if item.get("action_id") != artifact.get("action_id"):
            raise ValueError(f"fact {fact_id} action_id does not match its evidence")
        if item.get("tool") != (artifact.get("tool") or artifact.get("action")):
            raise ValueError(f"fact {fact_id} tool does not match its evidence")
        if (
            state.get("recommended_corpus")
            and item.get("corpus") != state.get("recommended_corpus")
        ):
            raise ValueError(f"fact {fact_id} escaped the pinned corpus")
        facts.append(item)
    return facts


def _facts_for_batch(
    facts: list[dict[str, Any]],
    batch: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evidence_ids = {
        _evidence_id(item, index) for index, item in enumerate(batch, 1)
    }
    return [item for item in facts if item.get("evidence_id") in evidence_ids]


def _completed_actions(state: ResearchState) -> list[dict[str, Any]]:
    return [dict(_as_mapping(item)) for item in state.get("completed_actions", [])]


def _decision_from_output(
    output: AggregatorOutput,
    state: ResearchState,
    batch: list[Mapping[str, Any]],
    batch_facts: list[dict[str, Any]],
) -> AggregatorDecision:
    evidence_ids = tuple(
        _evidence_id(item, index) for index, item in enumerate(batch, 1)
    )
    status = output.status
    termination_reason: TerminationReason | None = None
    error_kind = output.error_kind
    error_message = output.error_message

    if status is AggregatorStatus.CONTINUE and not output.progress_made:
        status = AggregatorStatus.COMPLETE
        termination_reason = TerminationReason.NO_PROGRESS
    elif status is AggregatorStatus.COMPLETE:
        termination_reason = TerminationReason.GOAL_REACHED
    elif status is AggregatorStatus.ERROR:
        termination_reason = TerminationReason.ERROR

    return AggregatorDecision(
        research_id=str(state.get("research_id") or "research-unassigned"),
        run_id=str(state.get("run_id") or "run-unassigned"),
        branch_id=str(state.get("branch_id") or "branch-unassigned"),
        batch_id=str(state.get("batch_id") or "batch-unassigned"),
        status=status,
        reasoning=output.reasoning,
        iteration=int(state.get("iteration_count", 0)) + 1,
        progress_made=output.progress_made,
        evidence_ids=evidence_ids,
        fact_ids=tuple(
            str(item["fact_id"]) for item in batch_facts if item.get("fact_id")
        ),
        covered_plan_steps=output.covered_plan_steps,
        missing_plan_steps=output.missing_plan_steps,
        missing_information=output.missing_information,
        termination_reason=termination_reason,
        error_kind=error_kind,
        error_message=error_message,
    )


def _decision_update(
    decision: AggregatorDecision,
    deductions: tuple[str, ...],
) -> dict[str, Any]:
    status = decision.status
    research_status = "running"
    if status is AggregatorStatus.COMPLETE:
        research_status = (
            "complete"
            if decision.termination_reason is TerminationReason.GOAL_REACHED
            else "partial"
        )
    elif status is AggregatorStatus.ERROR:
        research_status = "error"

    update: dict[str, Any] = {
        "deductions": list(deductions),
        "is_goal_reached": (
            status is AggregatorStatus.COMPLETE
            and decision.termination_reason is TerminationReason.GOAL_REACHED
        ),
        "needs_replanning": status is AggregatorStatus.REPLAN,
        "aggregator_status": status.value,
        "aggregator_reasoning": decision.reasoning,
        "missing_information": list(decision.missing_information),
        "iteration_count": decision.iteration,
        "termination_reason": (
            decision.termination_reason.value
            if decision.termination_reason is not None
            else None
        ),
        "research_status": research_status,
        "last_evidence_batch": [],
        "planned_actions": [],
    }
    if status is AggregatorStatus.ERROR:
        update["error"] = {
            "kind": (decision.error_kind or ErrorKind.VALIDATION).value,
            "message": decision.error_message or "Не удалось проверить evidence.",
            "node": "evidence_aggregator",
            "retryable": False,
        }
    else:
        update["error"] = None
    return update


async def evidence_aggregator_node_async(
    state: ResearchState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    current_batch = [
        _as_mapping(item) for item in state.get("last_evidence_batch", [])
    ]
    current_iteration = int(state.get("iteration_count", 0))
    if not current_batch:
        return {
            "is_goal_reached": False,
            "needs_replanning": False,
            "aggregator_status": "complete",
            "aggregator_reasoning": "Новый пакет evidence пуст; повторный анализ старых данных запрещён.",
            "termination_reason": "no_actions",
            "research_status": "partial",
            "last_evidence_batch": [],
            "planned_actions": [],
        }

    try:
        _validate_current_batch(state, current_batch)
        all_current_facts = _validated_facts(state, current_batch)
        current_observable_facts = _facts_for_batch(
            all_current_facts,
            current_batch,
        )
    except Exception:
        return {
            "is_goal_reached": False,
            "needs_replanning": False,
            "aggregator_status": "error",
            "aggregator_reasoning": "Новый пакет evidence нарушает границы текущей ветки.",
            "iteration_count": current_iteration + 1,
            "termination_reason": "error",
            "research_status": "error",
            "error": {
                "kind": "validation",
                "message": "Некорректное происхождение текущего пакета evidence.",
                "node": "evidence_aggregator",
                "retryable": False,
            },
            "last_evidence_batch": [],
            "planned_actions": [],
        }

    max_chars = int(state.get("budgets", {}).get("max_context_chars", 300_000))
    selected, truncated = _select_batch(current_batch, max_chars)
    if truncated:
        return {
            "is_goal_reached": False,
            "needs_replanning": False,
            "aggregator_status": "error",
            "aggregator_reasoning": (
                "Новый пакет превышает бюджет контекста и не может быть "
                "проанализирован полностью без потери происхождения."
            ),
            "iteration_count": current_iteration + 1,
            "termination_reason": "budget_exhausted",
            "research_status": "partial",
            "error": {
                "kind": "budget",
                "message": "Пакет evidence превышает бюджет контекста.",
                "node": "evidence_aggregator",
                "retryable": False,
            },
            "last_evidence_batch": [],
            "planned_actions": [],
        }
    context = {
        "question": state.get("research_question", ""),
        "goal": state.get("goal", ""),
        "full_research_plan": state.get("research_plan", []),
        "current_batch": selected,
        "batch_truncated": truncated,
        "current_observable_facts": current_observable_facts,
        "all_current_facts": all_current_facts,
        "completed_actions": _completed_actions(state),
        "previous_missing_information": state.get("missing_information", []),
    }
    messages = [
        SystemMessage(content=EVIDENCE_AGGREGATOR_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(context, ensure_ascii=False, default=str)),
    ]

    try:
        output = await ainvoke_structured(
            get_llm(),
            messages,
            AggregatorOutput,
            node="evidence_aggregator",
            config=config,
        )
        plan_size = len(state.get("research_plan", []))
        referenced_steps = {
            *output.covered_plan_steps,
            *output.missing_plan_steps,
        }
        if any(step > plan_size for step in referenced_steps):
            raise ValueError("aggregator referenced a step outside the current plan")
        if (
            output.status is AggregatorStatus.COMPLETE
            and plan_size
            and set(output.covered_plan_steps) != set(range(1, plan_size + 1))
        ):
            raise ValueError("complete aggregation must cover every plan step")
        decision = _decision_from_output(
            output,
            state,
            selected,
            current_observable_facts,
        )
        return _decision_update(decision, output.deductions)
    except StructuredOutputError as error:
        return {
            "is_goal_reached": False,
            "needs_replanning": False,
            "aggregator_status": "error",
            "aggregator_reasoning": error.result.message,
            "iteration_count": current_iteration + 1,
            "termination_reason": "error",
            "research_status": "error",
            "error": error.result.as_safe_error(),
            "last_evidence_batch": [],
            "planned_actions": [],
        }
    except Exception:
        return {
            "is_goal_reached": False,
            "needs_replanning": False,
            "aggregator_status": "error",
            "aggregator_reasoning": "Решение агрегатора не прошло доменную проверку.",
            "iteration_count": current_iteration + 1,
            "termination_reason": "error",
            "research_status": "error",
            "error": {
                "kind": "validation",
                "message": "Некорректный ответ агрегатора evidence.",
                "node": "evidence_aggregator",
                "retryable": False,
            },
            "last_evidence_batch": [],
            "planned_actions": [],
        }


def evidence_aggregator_node(
    state: ResearchState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Synchronous Stage-3 adapter; Stage 4 wires the async node directly."""

    return asyncio.run(evidence_aggregator_node_async(state, config))


__all__ = ["evidence_aggregator_node", "evidence_aggregator_node_async"]
