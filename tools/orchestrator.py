"""Asynchronous, bounded executor for deterministic NKRJA tool actions."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from research.fact_extractor import extract_facts
from state.models import (
    ActionCall,
    ActionStatus,
    EvidenceArtifact,
    EvidenceStatus,
    Fact,
)
from state.schema import ResearchState
from tools.budgets import (
    BudgetExceeded,
    BudgetLimits,
    BudgetTracker,
    action_signature,
)
from tools.compatibility import CompatibilityStatus, check_compatibility
from tools.evidence_compressor import CompressionLimits, compress_response
from tools.nkrja_client import NKRJAClient
from tools.registry import ResultType, TOOL_CAPABILITIES, normalize_corpus
from utils.trace import emit_trace


@dataclass(frozen=True, slots=True)
class _ActionOutcome:
    action: ActionCall
    evidence: tuple[EvidenceArtifact, ...] = ()
    facts: tuple[Fact, ...] = ()


def _new_identifier(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("planned action must be a mapping or Pydantic model")


def _message(error: BaseException | str) -> str:
    value = str(error).strip() or type(error).__name__
    return value[:500]


def _action_from_item(
    item: Any,
    *,
    research_id: str,
    run_id: str,
    branch_id: str,
    batch_id: str,
    iteration: int,
) -> ActionCall | None:
    values = _as_mapping(item)
    tool = values.get("tool") or values.get("action")
    if tool == "finish":
        return None
    params = copy.deepcopy(values.get("params") or {})
    if not isinstance(params, dict):
        raise TypeError("action params must be an object")
    signature = action_signature(str(tool or "unknown_action"), params)
    return ActionCall(
        research_id=research_id,
        run_id=run_id,
        branch_id=branch_id,
        batch_id=batch_id,
        action_id=values.get("action_id") or _new_identifier("action"),
        tool=str(tool or "unknown_action"),
        params=params,
        status=ActionStatus.PLANNED,
        revision=int(values.get("revision", 0)),
        iteration=iteration,
        depends_on_action_ids=tuple(values.get("depends_on_action_ids", ())),
        signature=signature,
    )


def _artifact(
    action: ActionCall,
    *,
    status: EvidenceStatus,
    source: str,
    params: Mapping[str, Any],
    payload: Any = None,
    message: str | None = None,
    raw_response_hash: str | None = None,
) -> EvidenceArtifact:
    return EvidenceArtifact(
        research_id=action.research_id,
        run_id=action.run_id,
        branch_id=action.branch_id,
        batch_id=action.batch_id,
        action_id=action.action_id,
        evidence_id=_new_identifier("evidence"),
        tool=action.tool,
        source=source,
        status=status,
        params=copy.deepcopy(dict(params)),
        payload=payload,
        message=message,
        raw_response_hash=raw_response_hash,
    )


def _completed(
    action: ActionCall,
    status: ActionStatus,
    *,
    params: Mapping[str, Any] | None = None,
    message: str | None = None,
) -> ActionCall:
    return action.model_copy(
        update={
            "params": copy.deepcopy(dict(params if params is not None else action.params)),
            "status": status,
            "message": message,
        }
    )


def _budget_outcome(
    action: ActionCall,
    error: BudgetExceeded,
    *,
    params: Mapping[str, Any] | None = None,
    warnings: tuple[EvidenceArtifact, ...] = (),
) -> _ActionOutcome:
    effective_params = params if params is not None else action.params
    message = _message(error)
    evidence = _artifact(
        action,
        status=EvidenceStatus.ERROR,
        source="System_Budget",
        params=effective_params,
        message=message,
    )
    return _ActionOutcome(
        _completed(
            action,
            ActionStatus.SKIPPED,
            params=effective_params,
            message=message,
        ),
        (*warnings, evidence),
    )


def _safe_trace(state: ResearchState, event: Mapping[str, Any], action: ActionCall) -> None:
    try:
        emit_trace(
            node="api_orchestrator",
            event_type="observation",
            content=dict(event),
            run_id=action.run_id,
            thread_id=state.get("thread_id"),
            turn_id=state.get("turn_id"),
            research_id=action.research_id,
            branch_id=action.branch_id,
            batch_id=action.batch_id,
            action_id=action.action_id,
            iteration=action.iteration,
        )
    except Exception:
        # Trace is a side channel and must not alter research semantics.
        return


def _prepare_word_portrait(
    action: ActionCall,
    *,
    create_warnings: bool = True,
) -> tuple[dict[str, Any], tuple[EvidenceArtifact, ...]]:
    params = copy.deepcopy(action.params)
    corpus = normalize_corpus(str(params.get("corpus", "MAIN"))).value
    params["corpus"] = corpus
    requested = params.get("resultType")
    if not isinstance(requested, list) or not requested:
        raise ValueError("resultType must be a non-empty list")

    allowed: list[str] = []
    warnings: list[EvidenceArtifact] = []
    for raw_result_type in requested:
        try:
            result_type = ResultType(str(raw_result_type).strip().upper())
        except ValueError as error:
            raise ValueError(f"unknown resultType: {raw_result_type!r}") from error
        decision = check_compatibility(corpus, result_type)
        if decision.status is CompatibilityStatus.UNSUPPORTED:
            if create_warnings:
                warnings.append(
                    _artifact(
                        action,
                        status=EvidenceStatus.WARNING,
                        source="System_Safeguard",
                        params=action.params,
                        message=(
                            f"{result_type.value} недоступен для {corpus}: "
                            f"{decision.reason}"
                        ),
                    )
                )
        else:
            # UNKNOWN is intentionally attempted: empty/error probe outcomes
            # are not permanent backend limitations.
            allowed.append(result_type.value)
    params["resultType"] = list(dict.fromkeys(allowed))
    return params, tuple(warnings)


async def _invoke_handler(handler: Any, params: Mapping[str, Any]) -> Any:
    result = handler(**copy.deepcopy(dict(params)))
    return await result if inspect.isawaitable(result) else result


async def _execute_one(
    state: ResearchState,
    action: ActionCall,
    *,
    client: Any,
    tracker: BudgetTracker,
    compression_limits: CompressionLimits,
) -> _ActionOutcome:
    capability = TOOL_CAPABILITIES.get(action.tool)
    if capability is None:
        try:
            await tracker.reserve_action(external=False)
        except BudgetExceeded as error:
            return _budget_outcome(action, error)
        message = f"Инструмент {action.tool!r} отсутствует в реестре."
        evidence = _artifact(
            action,
            status=EvidenceStatus.ERROR,
            source="System",
            params=action.params,
            message=message,
        )
        return _ActionOutcome(
            _completed(action, ActionStatus.FAILED, message=message),
            (evidence,),
        )

    try:
        validation_errors = capability.validate_params(action.params)
    except Exception as error:
        validation_errors = (_message(error),)
    if validation_errors:
        try:
            await tracker.reserve_action(external=False)
        except BudgetExceeded as error:
            return _budget_outcome(action, error)
        message = "; ".join(validation_errors)
        evidence = _artifact(
            action,
            status=EvidenceStatus.ERROR,
            source="System_Validation",
            params=action.params,
            message=message,
        )
        return _ActionOutcome(
            _completed(action, ActionStatus.FAILED, message=message),
            (evidence,),
        )

    effective_params = copy.deepcopy(action.params)
    warnings: tuple[EvidenceArtifact, ...] = ()
    try:
        if action.tool == "get_word_portrait":
            effective_params, warnings = _prepare_word_portrait(action)
    except (TypeError, ValueError) as error:
        try:
            await tracker.reserve_action(external=False)
        except BudgetExceeded as budget_error:
            return _budget_outcome(action, budget_error)
        message = _message(error)
        evidence = _artifact(
            action,
            status=EvidenceStatus.ERROR,
            source="System_Validation",
            params=action.params,
            message=message,
        )
        return _ActionOutcome(
            _completed(action, ActionStatus.FAILED, message=message),
            (evidence,),
        )

    if action.tool == "get_word_portrait" and not effective_params["resultType"]:
        try:
            await tracker.reserve_action(external=False)
        except BudgetExceeded as error:
            return _budget_outcome(
                action,
                error,
                params=effective_params,
                warnings=warnings,
            )
        message = "Все запрошенные типы результата подтверждённо недоступны."
        unsupported = _artifact(
            action,
            status=EvidenceStatus.UNSUPPORTED,
            source="System_Safeguard",
            params=action.params,
            message=message,
        )
        evidence = (*warnings, unsupported)
        return _ActionOutcome(
            _completed(action, ActionStatus.UNSUPPORTED, message=message),
            evidence,
        )

    handler = getattr(client, action.tool, None)
    if handler is None:
        try:
            await tracker.reserve_action(external=False)
        except BudgetExceeded as error:
            return _budget_outcome(
                action,
                error,
                params=effective_params,
                warnings=warnings,
            )
        message = f"Метод {action.tool!r} не реализован клиентом НКРЯ."
        evidence = _artifact(
            action,
            status=EvidenceStatus.ERROR,
            source="System",
            params=effective_params,
            message=message,
        )
        return _ActionOutcome(
            _completed(action, ActionStatus.FAILED, params=effective_params, message=message),
            (*warnings, evidence),
        )

    try:
        await tracker.reserve_action()
        remaining = tracker.remaining_wall_time()
        if remaining <= 0:
            raise BudgetExceeded(
                "max_wall_time_seconds", "Research wall-time budget exhausted."
            )
        try:
            async with asyncio.timeout(remaining):
                raw_response = await _invoke_handler(handler, effective_params)
                compressed = compress_response(
                    action.tool,
                    raw_response,
                    params=effective_params,
                    limits=compression_limits,
                )
        except TimeoutError as error:
            raise BudgetExceeded(
                "max_wall_time_seconds", "Research wall-time budget exhausted."
            ) from error
        success = _artifact(
            action,
            status=EvidenceStatus.SUCCESS,
            source="NKRJA",
            params=effective_params,
            payload=compressed,
            raw_response_hash=compressed["raw_response_hash"],
        )
        facts = extract_facts(success).facts
        _safe_trace(
            state,
            {
                "status": "success",
                "tool": action.tool,
                "raw_response_hash": success.raw_response_hash,
            },
            action,
        )
        return _ActionOutcome(
            _completed(action, ActionStatus.SUCCEEDED, params=effective_params),
            (*warnings, success),
            facts,
        )
    except BudgetExceeded as error:
        return _budget_outcome(
            action,
            error,
            params=effective_params,
            warnings=warnings,
        )
    except Exception as error:
        message = _message(error)
        evidence = _artifact(
            action,
            status=EvidenceStatus.ERROR,
            source="NKRJA",
            params=effective_params,
            message=message,
        )
        _safe_trace(
            state,
            {"status": "error", "tool": action.tool, "message": message},
            action,
        )
        return _ActionOutcome(
            _completed(action, ActionStatus.FAILED, params=effective_params, message=message),
            (*warnings, evidence),
        )


def _previous_signatures(state: ResearchState) -> set[str]:
    signatures: set[str] = set()
    for item in state.get("completed_actions", []):
        try:
            values = _as_mapping(item)
        except TypeError:
            continue
        signature = values.get("signature")
        if isinstance(signature, str) and signature:
            signatures.add(signature)
            continue
        tool = values.get("tool") or values.get("action")
        params = values.get("params")
        if isinstance(tool, str) and isinstance(params, Mapping):
            signatures.add(action_signature(tool, params))
    return signatures


def _previous_action_ids(state: ResearchState) -> dict[str, str]:
    identifiers: dict[str, str] = {}
    for item in state.get("completed_actions", []):
        try:
            values = _as_mapping(item)
        except TypeError:
            continue
        action_id = values.get("action_id")
        tool = values.get("tool") or values.get("action")
        params = values.get("params")
        signature = values.get("signature")
        if not isinstance(signature, str) and isinstance(tool, str) and isinstance(params, Mapping):
            signature = action_signature(tool, params)
        if isinstance(action_id, str) and isinstance(signature, str):
            identifiers[action_id] = signature
    return identifiers


def _requires_external_client(action: ActionCall) -> bool:
    """Return false only when execution is guaranteed to terminate locally."""

    capability = TOOL_CAPABILITIES.get(action.tool)
    if capability is None:
        return False
    try:
        if capability.validate_params(action.params):
            return False
        if action.tool == "get_word_portrait":
            effective_params, _warnings = _prepare_word_portrait(
                action, create_warnings=False
            )
            return bool(effective_params["resultType"])
    except Exception:
        return False
    return True


def _conflicting_id_outcome(action: ActionCall) -> _ActionOutcome:
    conflicting_id = action.action_id
    diagnostic = action.model_copy(
        update={"action_id": _new_identifier("action-conflict")}
    )
    message = (
        f"action_id {conflicting_id!r} уже связан с другим вызовом инструмента."
    )
    return _ActionOutcome(
        _completed(diagnostic, ActionStatus.FAILED, message=message),
        (
            _artifact(
                diagnostic,
                status=EvidenceStatus.ERROR,
                source="System_Validation",
                params=action.params,
                message=message,
            ),
        ),
    )


async def api_orchestrator_node_async(
    state: ResearchState,
    *,
    client: Any | None = None,
    budget_tracker: BudgetTracker | None = None,
    compression_limits: CompressionLimits | None = None,
) -> dict[str, Any]:
    """Execute only the current batch and return only newly created records."""

    planned_items = list(state.get("planned_actions", []))
    if not planned_items:
        return {
            "planned_actions": [],
            "last_evidence_batch": [],
            "execution_status": "complete",
        }

    research_id = state.get("research_id") or _new_identifier("research")
    run_id = state.get("run_id") or _new_identifier("run")
    branch_id = state.get("branch_id") or _new_identifier("branch")
    batch_id = state.get("batch_id") or _new_identifier("batch")
    iteration = int(state.get("iteration_count", 0))
    limits = BudgetLimits.from_mapping(state.get("budgets"))
    previous_actions = list(state.get("completed_actions", []))
    usage = dict(state.get("budget_usage", {}))
    tracker = budget_tracker or BudgetTracker(
        limits,
        existing_external_calls=int(
            usage.get(
                "external_calls",
                sum(
                    1
                    for item in previous_actions
                    if str(_as_mapping(item).get("status", ""))
                    in {
                        "succeeded",
                        "failed",
                        "ActionStatus.SUCCEEDED",
                        "ActionStatus.FAILED",
                    }
                ),
            )
        ),
        existing_branch_actions=int(usage.get("actions", len(previous_actions))),
        existing_evidence_items=int(
            usage.get("evidence_items", len(state.get("evidence", [])))
        ),
        existing_branches=int(usage.get("branches", 1)),
        existing_context_chars=int(usage.get("context_chars", 0)),
        started_at=state.get("research_started_at"),
    )
    compressor_limits = compression_limits or CompressionLimits(
        max_payload_chars=min(50_000, limits.max_context_chars)
    )

    known_signatures = _previous_signatures(state)
    known_action_ids = _previous_action_ids(state)
    current_signatures: set[str] = set()
    current_actions_by_id: dict[str, ActionCall] = {}
    conflicted_action_ids: set[str] = set()
    executable: list[ActionCall] = []
    immediate: list[_ActionOutcome] = []
    for item in planned_items:
        try:
            action = _action_from_item(
                item,
                research_id=research_id,
                run_id=run_id,
                branch_id=branch_id,
                batch_id=batch_id,
                iteration=iteration,
            )
        except Exception as error:
            # A malformed record still receives stable scope and a terminal
            # action/evidence pair instead of destroying sibling actions.
            fallback = ActionCall(
                research_id=research_id,
                run_id=run_id,
                branch_id=branch_id,
                batch_id=batch_id,
                action_id=_new_identifier("action"),
                tool="unknown_action",
                params={},
                iteration=iteration,
                signature=action_signature("unknown_action", {}),
            )
            message = _message(error)
            immediate.append(
                _ActionOutcome(
                    _completed(fallback, ActionStatus.FAILED, message=message),
                    (
                        _artifact(
                            fallback,
                            status=EvidenceStatus.ERROR,
                            source="System_Validation",
                            params={},
                            message=message,
                        ),
                    ),
                )
            )
            continue
        if action is None:
            continue
        assert action.signature is not None
        if action.action_id in conflicted_action_ids:
            continue
        previous_id_signature = known_action_ids.get(action.action_id)
        if (
            previous_id_signature is not None
            and previous_id_signature != action.signature
        ):
            immediate.append(_conflicting_id_outcome(action))
            continue
        current_same_id = current_actions_by_id.get(action.action_id)
        if (
            current_same_id is not None
            and current_same_id.signature != action.signature
        ):
            if current_same_id.signature is not None:
                current_signatures.discard(current_same_id.signature)
            executable = [
                candidate
                for candidate in executable
                if candidate.action_id != action.action_id
            ]
            immediate = [
                outcome
                for outcome in immediate
                if outcome.action.action_id != action.action_id
            ]
            immediate.append(_conflicting_id_outcome(action))
            conflicted_action_ids.add(action.action_id)
            continue
        current_actions_by_id[action.action_id] = action
        if action.signature in known_signatures or action.signature in current_signatures:
            message = "Идентичный вызов инструмента уже выполнен или запланирован."
            duplicate = _artifact(
                action,
                status=EvidenceStatus.WARNING,
                source="System_Safeguard",
                params=action.params,
                message=message,
            )
            immediate.append(
                _ActionOutcome(
                    _completed(action, ActionStatus.SKIPPED, message=message),
                    (duplicate,),
                )
            )
            continue
        current_signatures.add(action.signature)
        executable.append(action)

    local_actions: list[ActionCall] = []
    external_actions: list[ActionCall] = []
    for action in executable:
        target = (
            external_actions
            if _requires_external_client(action)
            else local_actions
        )
        target.append(action)
    outcomes: list[_ActionOutcome] = list(
        await asyncio.gather(
            *(
                _execute_one(
                    state,
                    action,
                    client=object(),
                    tracker=tracker,
                    compression_limits=compressor_limits,
                )
                for action in local_actions
            )
        )
    )
    if external_actions:
        owns_client = client is None
        try:
            active_client = client or NKRJAClient()
        except Exception as error:
            message = _message(error)
            outcomes.extend(
                _ActionOutcome(
                    _completed(action, ActionStatus.FAILED, message=message),
                    (
                        _artifact(
                            action,
                            status=EvidenceStatus.ERROR,
                            source="System_Configuration",
                            params=action.params,
                            message=message,
                        ),
                    ),
                )
                for action in external_actions
            )
        else:
            try:
                outcomes.extend(
                    await asyncio.gather(
                        *(
                            _execute_one(
                                state,
                                action,
                                client=active_client,
                                tracker=tracker,
                                compression_limits=compressor_limits,
                            )
                            for action in external_actions
                        )
                    )
                )
            finally:
                if owns_client:
                    close = getattr(active_client, "aclose", None)
                    if close is not None:
                        result = close()
                        if inspect.isawaitable(result):
                            await result

    all_outcomes = sorted(
        [*immediate, *outcomes], key=lambda outcome: outcome.action.action_id
    )
    retained_outcomes: list[_ActionOutcome] = []
    for outcome in all_outcomes:
        serialized = [item.model_dump(mode="json") for item in outcome.evidence]
        context_chars = len(
            json.dumps(
                serialized,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        )
        try:
            await tracker.reserve_evidence(
                len(outcome.evidence), context_chars=context_chars
            )
        except BudgetExceeded as error:
            budget_outcome = _budget_outcome(outcome.action, error)
            budget_serialized = [
                item.model_dump(mode="json") for item in budget_outcome.evidence
            ]
            budget_chars = len(
                json.dumps(
                    budget_serialized,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            )
            try:
                await tracker.reserve_evidence(
                    len(budget_outcome.evidence), context_chars=budget_chars
                )
            except BudgetExceeded:
                message = _message(error)
                budget_outcome = _ActionOutcome(
                    _completed(
                        outcome.action,
                        ActionStatus.SKIPPED,
                        message=message,
                    )
                )
            retained_outcomes.append(budget_outcome)
        else:
            retained_outcomes.append(outcome)

    new_evidence = [
        artifact
        for outcome in retained_outcomes
        for artifact in outcome.evidence
    ]
    retained_evidence_ids = {item.evidence_id for item in new_evidence}
    new_facts = sorted(
        (
            fact
            for outcome in retained_outcomes
            for fact in outcome.facts
            if fact.evidence_id in retained_evidence_ids
        ),
        key=lambda fact: fact.fact_id,
    )

    return {
        "evidence": [item.model_dump(mode="json") for item in new_evidence],
        "facts": [item.model_dump(mode="json") for item in new_facts],
        "last_evidence_batch": [
            item.model_dump(mode="json") for item in new_evidence
        ],
        "completed_actions": [
            outcome.action.model_dump(mode="json")
            for outcome in retained_outcomes
        ],
        "budget_usage": tracker.usage,
        "planned_actions": [],
        "batch_id": batch_id,
        "execution_status": "complete",
    }


def api_orchestrator_node(state: ResearchState) -> dict[str, Any]:
    """Temporary sync adapter for the stage-1 graph.

    Stage 4 will wire ``api_orchestrator_node_async`` directly into the fully
    asynchronous graph.  Calling this adapter from an existing event loop is a
    programming error rather than a reason to create a nested loop.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(api_orchestrator_node_async(state))
    raise RuntimeError(
        "api_orchestrator_node cannot run inside an event loop; use "
        "api_orchestrator_node_async"
    )


__all__ = ["api_orchestrator_node", "api_orchestrator_node_async"]
