"""Lifecycle boundary between a persistent conversation and one research task."""

from __future__ import annotations

import uuid
import time
from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.config import get_settings
from state.models import ResearchStatus
from state.reducers import reset_accumulator
from state.schema import ConversationTurn, ResearchSnapshot, ResearchState


def _new_identifier(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


def _thread_id(state: ResearchState, config: RunnableConfig | None) -> str:
    configurable: Mapping[str, Any] = {}
    if config:
        value = config.get("configurable", {})
        if isinstance(value, Mapping):
            configurable = value
    configured = configurable.get("thread_id")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    existing = state.get("thread_id")
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    return _new_identifier("thread")


def _snapshot_previous(state: ResearchState) -> ResearchSnapshot | None:
    research_id = state.get("research_id")
    question = state.get("active_research_question")
    if not research_id or not question:
        return None

    status = state.get("research_status", ResearchStatus.PARTIAL)
    if status not in {
        ResearchStatus.COMPLETE,
        ResearchStatus.PARTIAL,
        ResearchStatus.ERROR,
        ResearchStatus.CANCELLED,
        "complete",
        "partial",
        "error",
        "cancelled",
    }:
        status = ResearchStatus.PARTIAL

    return ResearchSnapshot(
        research_id=research_id,
        run_id=state.get("run_id", ""),
        turn_id=state.get("turn_id", ""),
        question=question,
        goal=state.get("goal", ""),
        status=status,
        final_response=state.get("final_response", ""),
        facts=list(state.get("facts", [])),
        evidence=list(state.get("evidence", [])),
        termination_reason=state.get("termination_reason"),
    )


def start_turn_node(
    state: ResearchState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Start exactly one user turn and isolate it from prior research data.

    This node is the graph entry point. Replanning edges target the planner
    directly, so an aggregator return cannot accidentally create new IDs.
    """

    question = str(state.get("research_question", "")).strip()
    if not question:
        return {
            "research_status": ResearchStatus.ERROR,
            "error": {
                "kind": "validation",
                "message": "Исследовательский вопрос не может быть пустым.",
                "node": "lifecycle",
                "retryable": False,
            },
        }

    previous = _snapshot_previous(state)
    archive = dict(state.get("research_archive", {}))
    history = list(state.get("conversation_history", []))
    selected_context_research_id: str | None = None
    context_facts: list[Any] = []
    context_evidence: list[Any] = []

    if previous is not None:
        archive[previous["research_id"]] = previous
        inherited_context_id = state.get("selected_context_research_id")
        inherited_facts = list(state.get("context_facts", []))
        inherited_evidence = list(state.get("context_evidence", []))
        if state.get("mode") == "chat" and inherited_context_id:
            selected_context_research_id = inherited_context_id
            context_facts = inherited_facts
            context_evidence = inherited_evidence
        else:
            selected_context_research_id = previous["research_id"]
            context_facts = list(previous.get("facts", []))
            context_evidence = list(previous.get("evidence", []))
        history.append(
            ConversationTurn(
                turn_id=previous.get("turn_id", ""),
                research_id=previous["research_id"],
                question=previous["question"],
                answer=previous.get("final_response", ""),
            )
        )

    settings = get_settings()
    turn_id = _new_identifier("turn")
    research_id = _new_identifier("research")
    run_id = _new_identifier("run")

    return {
        "thread_id": _thread_id(state, config),
        "conversation_history": history,
        "research_archive": archive,
        "selected_context_research_id": selected_context_research_id,
        "context_facts": context_facts,
        "context_evidence": context_evidence,
        "turn_id": turn_id,
        "research_id": research_id,
        "run_id": run_id,
        "branch_id": _new_identifier("branch"),
        "batch_id": _new_identifier("batch"),
        "active_research_question": question,
        "research_question": question,
        "final_response": "",
        "research_status": ResearchStatus.PLANNING,
        "mode": "research",
        "planner_route": "research",
        "planner_reasoning": "",
        "overall_goal": "",
        "goal": "",
        "research_plan": [],
        "current_step_index": 0,
        "recommended_corpus": "MAIN",
        "corpus_reasoning": "",
        "hypotheses": [],
        "open_questions": [],
        "sub_goals": [],
        "iteration_count": 0,
        "research_started_at": time.time(),
        "budgets": {
            "max_iterations": settings.max_research_iterations,
            "max_branches": settings.max_research_branches,
            "max_actions_per_branch": settings.max_actions_per_branch,
            "max_requests_per_iteration": settings.max_requests_per_iteration,
            "max_external_calls": settings.max_external_calls,
            "max_wall_time_seconds": settings.research_wall_time_seconds,
            "max_evidence_items": settings.max_evidence_items,
            "max_context_chars": settings.evidence_context_budget_chars,
        },
        "budget_usage": {
            "branches": 1,
            "actions": 0,
            "external_calls": 0,
            "evidence_items": 0,
            "context_chars": 0,
        },
        "evidence": reset_accumulator(),
        "facts": reset_accumulator(),
        "deductions": reset_accumulator(),
        "completed_actions": reset_accumulator(),
        "branch_results": reset_accumulator(),
        "planned_actions": [],
        "last_evidence_batch": [],
        "execution_status": None,
        "aggregator_status": None,
        "aggregator_reasoning": "",
        "missing_information": [],
        "needs_replanning": False,
        "is_goal_reached": False,
        "confidence": 0.0,
        "pagination_context": {},
        "termination_reason": None,
        "error": None,
    }


__all__ = ["start_turn_node"]
