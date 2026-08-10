"""Typed asynchronous execution planning for one corpus-bound branch."""

import asyncio
import json
import uuid
from typing import Any, Mapping

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from LLM.client import get_llm
from LLM.prompts import EXECUTION_PLANNER_SYSTEM_PROMPT
from LLM.structured import (
    ExecutionOutput,
    StructuredOutputError,
    ainvoke_structured,
)
from state.models import (
    ActionCall,
    ErrorKind,
    ExecutionPlan,
    ExecutionStatus,
)
from state.schema import ResearchState
from tools.budgets import action_signature
from tools.registry import TOOL_CAPABILITIES, get_registry_description
from utils.trace import emit_trace


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value if isinstance(value, Mapping) else {}


def _past_signatures(state: ResearchState) -> set[str]:
    signatures: set[str] = set()
    for record in state.get("completed_actions", []):
        item = _as_mapping(record)
        signature = item.get("signature")
        tool = item.get("tool") or item.get("action")
        params = item.get("params", {})
        if signature:
            signatures.add(str(signature))
        elif tool and isinstance(params, Mapping):
            signatures.add(action_signature(str(tool), params))
    return signatures


def _remaining_budgets(state: ResearchState) -> dict[str, int]:
    limits = state.get("budgets", {})
    usage = state.get("budget_usage", {})
    completed = len(state.get("completed_actions", []))
    return {
        "actions": max(
            0,
            int(limits.get("max_actions_per_branch", 16))
            - max(completed, int(usage.get("actions", 0))),
        ),
        "requests_this_iteration": int(
            limits.get("max_requests_per_iteration", 4)
        ),
        "external_calls": max(
            0,
            int(limits.get("max_external_calls", 32))
            - int(usage.get("external_calls", 0)),
        ),
    }


def _safe_error_update(
    *,
    batch_id: str,
    kind: ErrorKind | str,
    message: str,
    retryable: bool,
) -> dict[str, Any]:
    kind_value = kind.value if isinstance(kind, ErrorKind) else str(kind)
    return {
        "planned_actions": [],
        "batch_id": batch_id,
        "execution_status": "error",
        "research_status": "error",
        "termination_reason": "error",
        "error": {
            "kind": kind_value,
            "message": message,
            "node": "execution_planner",
            "retryable": retryable,
        },
    }


def _build_plan(
    output: ExecutionOutput,
    state: ResearchState,
    batch_id: str,
) -> ExecutionPlan:
    research_id = str(state.get("research_id") or "research-unassigned")
    run_id = str(state.get("run_id") or "run-unassigned")
    branch_id = str(state.get("branch_id") or "branch-unassigned")
    iteration = int(state.get("iteration_count", 0))
    pinned_corpus = str(state.get("recommended_corpus") or "MAIN")
    remaining = _remaining_budgets(state)

    if output.status is ExecutionStatus.EXECUTE:
        if len(output.actions) > remaining["actions"]:
            raise RuntimeError("max_actions_per_branch budget exhausted")
        if len(output.actions) > remaining["requests_this_iteration"]:
            raise RuntimeError("max_requests_per_iteration budget exhausted")
        if len(output.actions) > remaining["external_calls"]:
            raise RuntimeError("max_external_calls budget exhausted")

    past = _past_signatures(state)
    current: set[str] = set()
    actions: list[ActionCall] = []
    for index, proposed in enumerate(output.actions):
        capability = TOOL_CAPABILITIES.get(proposed.tool)
        if capability is None:
            raise ValueError(f"unknown tool: {proposed.tool}")
        params = dict(proposed.params)
        if "corpus" in capability.required_params or "corpus" in params:
            if params.get("corpus") != pinned_corpus:
                raise ValueError(
                    f"tool {proposed.tool} must use pinned corpus {pinned_corpus}"
                )
        violations = capability.validate_params(params)
        if violations:
            raise ValueError("; ".join(violations))
        if any(dependency >= index for dependency in proposed.depends_on):
            raise ValueError("dependencies must refer only to earlier actions")

        signature = action_signature(proposed.tool, params)
        if signature in past or signature in current:
            raise ValueError("execution plan repeats an existing action signature")
        current.add(signature)
        action_id = _new_id("action")
        dependencies = tuple(actions[item].action_id for item in proposed.depends_on)
        actions.append(
            ActionCall(
                research_id=research_id,
                run_id=run_id,
                branch_id=branch_id,
                batch_id=batch_id,
                action_id=action_id,
                tool=proposed.tool,
                params=params,
                iteration=iteration,
                depends_on_action_ids=dependencies,
                signature=signature,
            )
        )

    return ExecutionPlan(
        research_id=research_id,
        run_id=run_id,
        branch_id=branch_id,
        batch_id=batch_id,
        status=output.status,
        reasoning=output.reasoning,
        iteration=iteration,
        actions=tuple(actions),
        error_kind=output.error_kind,
        error_message=output.error_message,
    )


async def execution_planner_node_async(
    state: ResearchState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    batch_id = _new_id("batch")
    remaining = _remaining_budgets(state)
    if min(remaining["actions"], remaining["external_calls"]) <= 0:
        return _safe_error_update(
            batch_id=batch_id,
            kind=ErrorKind.BUDGET,
            message="Бюджет действий исследовательской ветки исчерпан.",
            retryable=False,
        )

    completed = [_as_mapping(item) for item in state.get("completed_actions", [])]
    facts = [_as_mapping(item) for item in state.get("facts", [])]
    evidence_warnings = [
        _as_mapping(item)
        for item in state.get("evidence", [])
        if str(_as_mapping(item).get("status"))
        in {"warning", "error", "unsupported"}
    ]
    context = {
        "goal": state.get("goal", ""),
        "research_plan": state.get("research_plan", []),
        "pinned_corpus": state.get("recommended_corpus", "MAIN"),
        "current_facts_with_provenance": facts,
        "completed_actions_with_exact_params": completed,
        "completed_signatures": sorted(_past_signatures(state)),
        "warnings_and_limitations": evidence_warnings,
        "remaining_budgets": remaining,
    }
    messages = [
        SystemMessage(
            content=EXECUTION_PLANNER_SYSTEM_PROMPT.format(
                registry_description=get_registry_description()
            )
        ),
        HumanMessage(content=json.dumps(context, ensure_ascii=False, default=str)),
    ]

    try:
        output = await ainvoke_structured(
            get_llm(),
            messages,
            ExecutionOutput,
            node="execution_planner",
            config=config,
        )
        plan = _build_plan(output, state, batch_id)
        emit_trace(
            node="execution_planner",
            event_type="planning",
            content={
                "status": plan.status.value,
                "reasoning_summary": plan.reasoning,
                "action_ids": [action.action_id for action in plan.actions],
            },
            run_id=plan.run_id,
            thread_id=state.get("thread_id"),
            turn_id=state.get("turn_id"),
            research_id=plan.research_id,
            branch_id=plan.branch_id,
            batch_id=plan.batch_id,
            iteration=plan.iteration,
        )
        if plan.status is ExecutionStatus.ERROR:
            return _safe_error_update(
                batch_id=batch_id,
                kind=plan.error_kind or ErrorKind.MODEL,
                message=plan.error_message or "Не удалось составить исполнительный план.",
                retryable=False,
            )
        if plan.status is ExecutionStatus.COMPLETE:
            return {
                "planned_actions": [],
                "batch_id": batch_id,
                "execution_status": "complete",
                # The execution planner can only say that it has no justified
                # actions left.  It cannot prove plan coverage; only the
                # evidence aggregator may declare goal_reached.
                "research_status": "partial",
                "termination_reason": "no_actions",
                "error": None,
            }
        return {
            "planned_actions": [
                action.model_dump(mode="json") for action in plan.actions
            ],
            "batch_id": batch_id,
            "execution_status": plan.status.value,
            "error": None,
        }
    except StructuredOutputError as error:
        return _safe_error_update(
            batch_id=batch_id,
            kind=error.result.kind,
            message=error.result.message,
            retryable=error.result.retryable,
        )
    except RuntimeError:
        return _safe_error_update(
            batch_id=batch_id,
            kind=ErrorKind.BUDGET,
            message="Предложенный план превышает бюджет исследовательской ветки.",
            retryable=False,
        )
    except Exception:
        return _safe_error_update(
            batch_id=batch_id,
            kind=ErrorKind.VALIDATION,
            message="Исполнительный план не соответствует контракту.",
            retryable=False,
        )


def execution_planner_node(
    state: ResearchState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Synchronous Stage-3 adapter; Stage 4 wires the async node directly."""

    return asyncio.run(execution_planner_node_async(state, config))


__all__ = ["execution_planner_node", "execution_planner_node_async"]
