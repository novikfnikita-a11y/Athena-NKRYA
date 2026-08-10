"""Stage-3 contracts for strict LLM decisions and safe terminal behavior."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from LLM.structured import (
    AggregatorOutput,
    ExecutionOutput,
    StructuredOutputError,
    ainvoke_structured,
)
from app.graph import evidence_router, execution_router
from planners.execution import execution_planner_node_async
from planners.planner import planner_node_async
from research.assistant import assistant_node_async
from research.evidence import evidence_aggregator_node_async
from research.errors import budget_terminal_node, error_node


def _valid_aggregation(**overrides: Any) -> dict[str, Any]:
    value = {
        "status": "continue",
        "reasoning": "Получен новый результат, нужен следующий шаг",
        "progress_made": True,
        "covered_plan_steps": [1],
        "missing_plan_steps": [2],
        "missing_information": ["Нужны примеры"],
        "deductions": [],
        "error_kind": None,
        "error_message": None,
    }
    value.update(overrides)
    return value


@pytest.mark.asyncio
async def test_invalid_json_gets_exactly_one_repair(fake_llm: Any) -> None:
    fake_llm.queue("not-json", _valid_aggregation())

    result = await ainvoke_structured(
        fake_llm,
        [HumanMessage(content="Проверь пакет")],
        AggregatorOutput,
        node="test_aggregator",
    )

    assert result.progress_made is True
    assert len(fake_llm.calls) == 2
    assert "JSON Schema" in fake_llm.calls[1].input[-1].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        _valid_aggregation(progress_made="false"),
        _valid_aggregation(unexpected="forbidden"),
        _valid_aggregation(covered_plan_steps=["1"]),
        "prefix {\"status\": \"continue\"} suffix",
    ],
)
async def test_coercions_extras_and_json_substrings_are_rejected(
    fake_llm: Any,
    invalid: Any,
) -> None:
    fake_llm.queue(invalid, _valid_aggregation(progress_made=False))

    result = await ainvoke_structured(
        fake_llm,
        [HumanMessage(content="Проверь пакет")],
        AggregatorOutput,
        node="test_aggregator",
    )

    assert result.progress_made is False
    assert len(fake_llm.calls) == 2


@pytest.mark.asyncio
async def test_second_invalid_response_is_typed_terminal_failure(
    fake_llm: Any,
) -> None:
    fake_llm.queue("not-json", {"progress_made": "false"})

    with pytest.raises(StructuredOutputError) as raised:
        await ainvoke_structured(
            fake_llm,
            [HumanMessage(content="Проверь пакет")],
            AggregatorOutput,
            node="evidence_aggregator",
        )

    assert raised.value.result.attempts == 2
    assert raised.value.result.kind.value == "validation"
    assert raised.value.result.retryable is False
    assert len(fake_llm.calls) == 2


@pytest.mark.asyncio
async def test_planner_builds_typed_multicorpus_subgoals(fake_llm: Any) -> None:
    fake_llm.queue(
        {
            "mode": "research",
            "reasoning": "Запрошено сравнение двух регистров",
            "goal": "Сравнить употребление леммы",
            "sub_goals": [
                {
                    "goal": "Исследовать основной корпус",
                    "corpus": "MAIN",
                    "corpus_reasoning": "Общая письменная норма",
                    "lemma": "слово",
                    "hypotheses": [],
                    "research_plan": ["Получить частотность"],
                },
                {
                    "goal": "Исследовать устный корпус",
                    "corpus": "SPOKEN",
                    "corpus_reasoning": "Разговорная речь",
                    "lemma": "слово",
                    "hypotheses": [],
                    "research_plan": ["Получить частотность"],
                },
            ],
            "error_kind": None,
            "error_message": None,
        }
    )

    result = await planner_node_async(
        {
            "research_question": "Сравни слово в основном и устном корпусе",
            "research_id": "research-1",
            "run_id": "run-1",
        }
    )

    assert [item["corpus"] for item in result["sub_goals"]] == ["MAIN", "SPOKEN"]
    assert len({item["branch_id"] for item in result["sub_goals"]}) == 2
    assert all(item["research_id"] == "research-1" for item in result["sub_goals"])
    assert result["overall_goal"] == "Сравнить употребление леммы"
    assert result["goal"] == "Исследовать основной корпус"


@pytest.mark.asyncio
async def test_execution_rejects_corpus_escape(fake_llm: Any) -> None:
    fake_llm.queue(
        {
            "status": "execute",
            "reasoning": "Попытка сменить корпус",
            "actions": [
                {
                    "tool": "get_corpus_stats",
                    "params": {"corpus": "SPOKEN"},
                    "depends_on": [],
                }
            ],
            "error_kind": None,
            "error_message": None,
        }
    )

    result = await execution_planner_node_async(
        {
            "research_id": "research-1",
            "run_id": "run-1",
            "branch_id": "branch-1",
            "recommended_corpus": "MAIN",
            "goal": "Проверить корпус",
            "research_plan": ["Получить статистику"],
            "completed_actions": [],
            "facts": [],
            "evidence": [],
        }
    )

    assert result["execution_status"] == "error"
    assert result["error"]["kind"] == "validation"
    assert result["planned_actions"] == []


@pytest.mark.asyncio
async def test_execution_complete_is_explicit_terminal_route(fake_llm: Any) -> None:
    fake_llm.queue(
        {
            "status": "complete",
            "reasoning": "Все шаги уже покрыты фактами",
            "actions": [],
            "error_kind": None,
            "error_message": None,
        }
    )

    result = await execution_planner_node_async(
        {
            "research_id": "research-1",
            "run_id": "run-1",
            "branch_id": "branch-1",
            "recommended_corpus": "MAIN",
            "goal": "Проверить корпус",
            "research_plan": ["Получить статистику"],
            "completed_actions": [],
            "facts": [],
            "evidence": [],
        }
    )

    assert result["execution_status"] == "complete"
    assert result["research_status"] == "partial"
    assert result["termination_reason"] == "no_actions"
    assert execution_router(result) == "assistant"


@pytest.mark.asyncio
async def test_assistant_uses_selected_chat_facts_with_evidence_ids(
    fake_llm: Any,
) -> None:
    fake_llm.queue("Ответ по выбранному факту [evidence-ctx].")
    context_fact = {
        "fact_id": "fact-ctx",
        "evidence_id": "evidence-ctx",
        "research_id": "research-old",
        "run_id": "run-old",
        "branch_id": "branch-old",
        "batch_id": "batch-old",
        "action_id": "action-old",
        "revision": 0,
        "corpus": "MAIN",
        "metric": "documents",
        "value": 10,
        "tool": "get_corpus_stats",
        "lemma": None,
        "unit": "documents",
    }

    result = await assistant_node_async(
        {
            "mode": "chat",
            "research_question": "Что означал прошлый результат?",
            "context_facts": [context_fact],
            "facts": [dict(context_fact, evidence_id="forbidden-current")],
            "run_id": "run-chat",
        }
    )

    prompt = fake_llm.calls[0].input[-1].content
    assert "evidence-ctx" in prompt
    assert "forbidden-current" not in prompt
    assert result["research_status"] == "complete"


@pytest.mark.asyncio
async def test_assistant_receives_qualitative_provenance_fact(fake_llm: Any) -> None:
    fake_llm.queue("Пример подтверждён [evidence-example].")
    qualitative_fact = {
        "fact_id": "fact-example",
        "evidence_id": "evidence-example",
        "research_id": "research-1",
        "run_id": "run-1",
        "branch_id": "branch-1",
        "batch_id": "batch-1",
        "action_id": "action-1",
        "corpus": "MAIN",
        "metric": "concordance.0.observation",
        "value": {"text": "Проверяемый пример", "doc_id": "doc-1"},
        "tool": "get_simple_concordance",
        "lemma": "пример",
        "unit": None,
    }

    result = await assistant_node_async(
        {
            "mode": "research",
            "research_question": "Приведи пример",
            "facts": [qualitative_fact],
            "research_status": "complete",
            "termination_reason": "goal_reached",
            "run_id": "run-1",
        }
    )

    prompt = fake_llm.calls[0].input[-1].content
    assert "Проверяемый пример" in prompt
    assert "evidence-example" in prompt
    assert result["research_status"] == "complete"


def test_error_node_redacts_secrets_and_keeps_run_id() -> None:
    result = error_node(
        {
            "run_id": "run-diagnostic",
            "error": {
                "kind": "api",
                "message": "Bearer super-secret-token api_key=another-secret",
                "node": "worker",
                "retryable": False,
            },
            "facts": [],
        }
    )

    assert "super-secret-token" not in result["final_response"]
    assert "another-secret" not in result["final_response"]
    assert "run-diagnostic" in result["final_response"]


@pytest.mark.asyncio
async def test_aggregator_never_claims_complete_for_uncovered_plan(
    fake_llm: Any,
) -> None:
    fake_llm.queue(
        _valid_aggregation(
            status="complete",
            covered_plan_steps=[1],
            missing_plan_steps=[],
            missing_information=[],
        )
    )
    result = await evidence_aggregator_node_async(
        {
            "research_id": "research-1",
            "run_id": "run-1",
            "branch_id": "branch-1",
            "batch_id": "batch-1",
            "recommended_corpus": "MAIN",
            "research_plan": ["Шаг 1", "Шаг 2"],
            "last_evidence_batch": [
                {
                    "research_id": "research-1",
                    "run_id": "run-1",
                    "branch_id": "branch-1",
                    "batch_id": "batch-1",
                    "evidence_id": "evidence-1",
                    "action_id": "action-1",
                    "tool": "get_corpus_stats",
                    "params": {"corpus": "MAIN"},
                    "status": "success",
                    "payload": {"documents": 10},
                }
            ],
            "facts": [],
            "completed_actions": [{
                "research_id": "research-1",
                "run_id": "run-1",
                "branch_id": "branch-1",
                "batch_id": "batch-1",
                "action_id": "action-1",
                "tool": "get_corpus_stats",
                "params": {"corpus": "MAIN"},
                "status": "succeeded",
            }],
        }
    )

    assert result["aggregator_status"] == "error"
    assert result["error"]["kind"] == "validation"


@pytest.mark.asyncio
async def test_context_truncation_is_explicit_budget_terminal(
    fake_llm: Any,
) -> None:
    result = await evidence_aggregator_node_async(
        {
            "research_id": "research-1",
            "run_id": "run-1",
            "branch_id": "branch-1",
            "batch_id": "batch-1",
            "recommended_corpus": "MAIN",
            "research_plan": ["Шаг 1"],
            "budgets": {"max_context_chars": 20},
            "last_evidence_batch": [
                {
                    "research_id": "research-1",
                    "run_id": "run-1",
                    "branch_id": "branch-1",
                    "batch_id": "batch-1",
                    "evidence_id": "evidence-1",
                    "action_id": "action-1",
                    "tool": "get_corpus_stats",
                    "params": {"corpus": "MAIN"},
                    "status": "success",
                    "payload": {"very_large": "x" * 1_000},
                }
            ],
            "facts": [],
            "completed_actions": [{
                "research_id": "research-1",
                "run_id": "run-1",
                "branch_id": "branch-1",
                "batch_id": "batch-1",
                "action_id": "action-1",
                "tool": "get_corpus_stats",
                "params": {"corpus": "MAIN"},
                "status": "succeeded",
            }],
        }
    )

    assert result["aggregator_status"] == "error"
    assert result["research_status"] == "partial"
    assert result["termination_reason"] == "budget_exhausted"
    assert result["error"]["kind"] == "budget"
    assert fake_llm.calls == []


def test_execution_dependency_indices_are_strict_integers() -> None:
    with pytest.raises(ValidationError):
        ExecutionOutput.model_validate_json(
            '{"status":"execute","reasoning":"test","actions":['
            '{"tool":"get_corpus_stats","params":{"corpus":"MAIN"},'
            '"depends_on":["0"]}],"error_kind":null,"error_message":null}'
        )


@pytest.mark.asyncio
async def test_planner_enforces_branch_budget(fake_llm: Any) -> None:
    fake_llm.queue(
        {
            "mode": "research",
            "reasoning": "Слишком много ветвей",
            "goal": "Сравнить корпуса",
            "sub_goals": [
                {
                    "goal": "MAIN",
                    "corpus": "MAIN",
                    "corpus_reasoning": "Первый",
                    "hypotheses": [],
                    "research_plan": [],
                },
                {
                    "goal": "SPOKEN",
                    "corpus": "SPOKEN",
                    "corpus_reasoning": "Второй",
                    "hypotheses": [],
                    "research_plan": [],
                },
            ],
        }
    )
    result = await planner_node_async(
        {
            "research_question": "Сравнить",
            "research_id": "research-1",
            "run_id": "run-1",
            "budgets": {"max_branches": 1},
        }
    )

    assert result["planner_route"] == "error"
    assert result["error"]["kind"] == "validation"


@pytest.mark.asyncio
async def test_foreign_or_unlinked_evidence_is_rejected(fake_llm: Any) -> None:
    base = {
        "research_id": "research-1",
        "run_id": "run-1",
        "branch_id": "branch-1",
        "batch_id": "batch-1",
        "recommended_corpus": "MAIN",
        "research_plan": ["Шаг 1"],
        "facts": [],
        "completed_actions": [{
            "research_id": "research-1",
            "run_id": "run-1",
            "branch_id": "branch-1",
            "batch_id": "batch-1",
            "action_id": "action-1",
            "tool": "get_corpus_stats",
            "params": {"corpus": "MAIN"},
            "status": "succeeded",
        }],
    }
    foreign = await evidence_aggregator_node_async(
        {
            **base,
            "last_evidence_batch": [{
                "research_id": "research-foreign",
                "run_id": "run-1",
                "branch_id": "branch-1",
                "batch_id": "batch-1",
                "action_id": "action-1",
                "evidence_id": "evidence-foreign",
                "tool": "get_corpus_stats",
                "params": {"corpus": "MAIN"},
                "status": "success",
                "payload": {"documents": 10},
            }],
        }
    )
    missing_id = await evidence_aggregator_node_async(
        {
            **base,
            "last_evidence_batch": [{
                "research_id": "research-1",
                "run_id": "run-1",
                "branch_id": "branch-1",
                "batch_id": "batch-1",
                "action_id": "action-1",
                "tool": "get_corpus_stats",
                "params": {"corpus": "MAIN"},
                "status": "success",
                "payload": {"documents": 10},
            }],
        }
    )

    assert foreign["error"]["kind"] == "validation"
    assert missing_id["error"]["kind"] == "validation"
    assert fake_llm.calls == []


@pytest.mark.asyncio
async def test_aggregator_rejects_artifact_param_substitution(fake_llm: Any) -> None:
    result = await evidence_aggregator_node_async(
        {
            "research_id": "research-1",
            "run_id": "run-1",
            "branch_id": "branch-1",
            "batch_id": "batch-1",
            "recommended_corpus": "MAIN",
            "research_plan": ["Шаг 1"],
            "last_evidence_batch": [{
                "research_id": "research-1",
                "run_id": "run-1",
                "branch_id": "branch-1",
                "batch_id": "batch-1",
                "action_id": "action-1",
                "evidence_id": "evidence-1",
                "tool": "get_simple_concordance",
                "params": {"corpus": "MAIN", "lemma": "ЧУЖАЯ"},
                "status": "success",
                "payload": {"examples": []},
            }],
            "facts": [],
            "completed_actions": [{
                "research_id": "research-1",
                "run_id": "run-1",
                "branch_id": "branch-1",
                "batch_id": "batch-1",
                "action_id": "action-1",
                "tool": "get_simple_concordance",
                "params": {"corpus": "MAIN", "lemma": "мир"},
                "status": "succeeded",
            }],
        }
    )

    assert result["aggregator_status"] == "error"
    assert result["error"]["kind"] == "validation"
    assert fake_llm.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fact_scope", ["foreign", "unlinked"])
async def test_aggregator_rejects_untrusted_all_current_facts(
    fake_llm: Any,
    fact_scope: str,
) -> None:
    artifact = {
        "research_id": "research-1",
        "run_id": "run-1",
        "branch_id": "branch-1",
        "batch_id": "batch-1",
        "action_id": "action-1",
        "evidence_id": "evidence-1",
        "tool": "get_simple_concordance",
        "params": {"corpus": "MAIN", "lemma": "мир"},
        "status": "success",
        "payload": {"examples": []},
    }
    fact = {
        "fact_id": f"fact-{fact_scope}",
        "evidence_id": (
            "evidence-unknown" if fact_scope == "unlinked" else "evidence-1"
        ),
        "research_id": (
            "research-foreign" if fact_scope == "foreign" else "research-1"
        ),
        "run_id": "run-1",
        "branch_id": "branch-1",
        "batch_id": "batch-1",
        "action_id": "action-1",
        "corpus": "MAIN",
        "metric": "concordance.0.observation",
        "value": {"text": "Чужое наблюдение"},
        "tool": "get_simple_concordance",
        "lemma": "мир",
        "unit": None,
    }
    result = await evidence_aggregator_node_async(
        {
            "research_id": "research-1",
            "run_id": "run-1",
            "branch_id": "branch-1",
            "batch_id": "batch-1",
            "recommended_corpus": "MAIN",
            "research_plan": ["Шаг 1"],
            "evidence": [artifact],
            "last_evidence_batch": [artifact],
            "facts": [fact],
            "completed_actions": [{
                "research_id": "research-1",
                "run_id": "run-1",
                "branch_id": "branch-1",
                "batch_id": "batch-1",
                "action_id": "action-1",
                "tool": "get_simple_concordance",
                "params": {"corpus": "MAIN", "lemma": "мир"},
                "status": "succeeded",
            }],
        }
    )

    assert result["aggregator_status"] == "error"
    assert result["error"]["kind"] == "validation"
    assert fake_llm.calls == []


def test_iteration_limit_routes_through_partial_budget_terminal() -> None:
    state = {
        "iteration_count": 4,
        "budgets": {"max_iterations": 4},
        "aggregator_status": "continue",
        "research_status": "running",
        "termination_reason": None,
        "is_goal_reached": False,
    }
    assert evidence_router(state) == "budget_terminal"
    update = budget_terminal_node(state)
    assert update["research_status"] == "partial"
    assert update["termination_reason"] == "budget_exhausted"
