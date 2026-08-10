"""State contracts, reducers, lifecycle, and research-isolation tests."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from research.lifecycle import start_turn_node
from state.models import (
    ActionCall,
    AggregatorDecision,
    AggregatorStatus,
    EvidenceArtifact,
    EvidenceStatus,
    ExecutionPlan,
    ExecutionStatus,
    Fact,
    PlannerDecision,
    ResearchSubGoal,
    TerminationReason,
)
from state.reducers import (
    merge_actions,
    merge_evidence,
    merge_facts,
    reset_accumulator,
)
from state.schema import ResearchState


SCOPE = {
    "research_id": "research-1",
    "run_id": "run-1",
    "branch_id": "branch-1",
    "batch_id": "batch-1",
}


def test_domain_contracts_validate_one_complete_scope_chain() -> None:
    sub_goal = ResearchSubGoal(
        research_id="research-1",
        run_id="run-1",
        branch_id="branch-1",
        goal="Сопоставить употребление",
        corpus="MAIN",
        corpus_reasoning="Основной корпус покрывает задачу",
    )
    decision = PlannerDecision(
        research_id="research-1",
        run_id="run-1",
        mode="research",
        reasoning="Нужны корпусные данные",
        goal="Сопоставить употребление",
        sub_goals=(sub_goal,),
    )
    action = ActionCall(
        **SCOPE,
        action_id="action-1",
        tool="get_corpus_stats",
        params={"corpus": "MAIN"},
        iteration=0,
    )
    plan = ExecutionPlan(
        **SCOPE,
        status=ExecutionStatus.EXECUTE,
        reasoning="Получить статистику",
        iteration=0,
        actions=(action,),
    )
    evidence = EvidenceArtifact(
        **SCOPE,
        action_id="action-1",
        evidence_id="evidence-1",
        tool="get_corpus_stats",
        source="NKRJA",
        status=EvidenceStatus.SUCCESS,
        payload={"documents": 10},
    )
    fact = Fact(
        **SCOPE,
        action_id="action-1",
        evidence_id="evidence-1",
        fact_id="fact-1",
        corpus="MAIN",
        metric="documents",
        value=10,
        tool="get_corpus_stats",
        unit="documents",
    )
    aggregation = AggregatorDecision(
        **SCOPE,
        status=AggregatorStatus.COMPLETE,
        reasoning="План покрыт",
        iteration=0,
        progress_made=True,
        evidence_ids=(evidence.evidence_id,),
        fact_ids=(fact.fact_id,),
        termination_reason=TerminationReason.GOAL_REACHED,
    )

    assert decision.route.value == "research"
    assert plan.actions == (action,)
    assert aggregation.route.value == "complete"


def test_contracts_reject_scope_mismatch_and_string_boolean() -> None:
    action = ActionCall(
        **SCOPE,
        action_id="action-1",
        tool="get_corpus_stats",
        iteration=1,
    )
    with pytest.raises(ValidationError, match="iteration"):
        ExecutionPlan(
            **SCOPE,
            status="execute",
            reasoning="Несогласованный пакет",
            iteration=0,
            actions=(action,),
        )

    with pytest.raises(ValidationError):
        AggregatorDecision(
            **SCOPE,
            status="complete",
            reasoning="Невалидный тип флага",
            iteration=0,
            progress_made="false",
            termination_reason="goal_reached",
        )


@pytest.mark.parametrize(
    ("reducer", "id_field"),
    [
        (merge_actions, "action_id"),
        (merge_evidence, "evidence_id"),
        (merge_facts, "fact_id"),
    ],
)
def test_reducers_are_idempotent_replace_by_id_and_order_independent(
    reducer: Any,
    id_field: str,
) -> None:
    first = {id_field: "id-b", "value": "old", "revision": 1}
    replacement = {id_field: "id-b", "value": "new", "revision": 2}
    other = {id_field: "id-a", "value": "other"}

    left = reducer(reducer([], [first, other]), [replacement, replacement])
    right = reducer(reducer([], [replacement]), [other])

    assert left == right
    assert [item[id_field] for item in left] == ["id-a", "id-b"]
    assert left[1]["value"] == "new"


def test_same_id_conflict_has_same_winner_in_both_delivery_orders() -> None:
    left_update = {"evidence_id": "evidence-1", "payload": {"marker": "a"}}
    right_update = {"evidence_id": "evidence-1", "payload": {"marker": "b"}}

    left_then_right = merge_evidence([left_update], [right_update])
    right_then_left = merge_evidence([right_update], [left_update])

    assert left_then_right == right_then_left


def test_reset_update_is_explicit_and_clears_accumulated_records() -> None:
    existing = [{"evidence_id": "evidence-old", "value": 1}]

    assert merge_evidence(existing, []) == existing
    assert merge_evidence(existing, reset_accumulator()) == []


def _apply_lifecycle_update(
    previous: dict[str, Any],
    update: dict[str, Any],
) -> dict[str, Any]:
    merged = {**previous, **update}
    for field, reducer in {
        "evidence": merge_evidence,
        "facts": merge_facts,
        "completed_actions": merge_actions,
    }.items():
        if field in update:
            merged[field] = reducer(previous.get(field, []), update[field])
    return merged


def test_lifecycle_archives_previous_research_and_isolates_new_turn() -> None:
    config = {"configurable": {"thread_id": "thread-shared"}}
    first_update = start_turn_node(
        {"research_question": "Первый вопрос"},
        config=config,
    )
    first = _apply_lifecycle_update({}, first_update)
    first.update(
        {
            "goal": "Первая цель",
            "research_status": "complete",
            "termination_reason": "goal_reached",
            "facts": ["FIRST_FACT"],
            "evidence": [
                {"evidence_id": "evidence-first", "payload": {"marker": "first"}}
            ],
            "final_response": "Первый ответ",
        }
    )

    second_update = start_turn_node(
        {**first, "research_question": "Второй вопрос"},
        config=config,
    )
    second = _apply_lifecycle_update(first, second_update)

    assert second["thread_id"] == "thread-shared"
    assert second["turn_id"] != first["turn_id"]
    assert second["research_id"] != first["research_id"]
    assert second["run_id"] != first["run_id"]
    assert second["facts"] == []
    assert second["evidence"] == []
    assert second["context_facts"] == ["FIRST_FACT"]
    assert second["selected_context_research_id"] == first["research_id"]
    assert second["research_archive"][first["research_id"]]["goal"] == "Первая цель"
    assert second["conversation_history"][-1]["answer"] == "Первый ответ"


def test_consecutive_chat_turns_keep_explicit_research_context() -> None:
    config = {"configurable": {"thread_id": "thread-chat"}}
    research_update = start_turn_node(
        {"research_question": "Исследовательский вопрос"},
        config=config,
    )
    research = _apply_lifecycle_update({}, research_update)
    research.update(
        {
            "research_status": "complete",
            "facts": ["RESEARCH_FACT"],
            "final_response": "Исследовательский ответ",
        }
    )

    chat_update = start_turn_node(
        {**research, "research_question": "Первое уточнение"},
        config=config,
    )
    chat = _apply_lifecycle_update(research, chat_update)
    original_context_id = chat["selected_context_research_id"]
    chat.update(
        {
            "mode": "chat",
            "research_status": "complete",
            "final_response": "Ответ на уточнение",
        }
    )

    next_update = start_turn_node(
        {**chat, "research_question": "Второе уточнение"},
        config=config,
    )
    next_turn = _apply_lifecycle_update(chat, next_update)

    assert next_turn["selected_context_research_id"] == original_context_id
    assert next_turn["context_facts"] == ["RESEARCH_FACT"]
    assert chat["research_id"] in next_turn["research_archive"]


def test_schema_has_no_legacy_empty_list_control_fields() -> None:
    fields = ResearchState.__annotations__

    assert "next_action" not in fields
    assert "action_params" not in fields
    assert "execution_status" in fields
    assert "aggregator_status" in fields
    assert "termination_reason" in fields
    assert "error" in fields
