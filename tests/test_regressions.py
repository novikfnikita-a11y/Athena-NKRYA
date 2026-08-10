"""Red regression tests for the defects recorded before the refactor.

These tests intentionally describe the required behaviour, not the current
implementation.  At plan step 1.2 they must fail on the baseline.  Later plan
steps turn them green one defect at a time; production code is deliberately not
changed here.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from app import graph as graph_module
from planners import execution as execution_module
from planners import planner as planner_module
from research import evidence as evidence_module
from tools import orchestrator as orchestrator_module


def _research_input(question: str) -> dict[str, Any]:
    """Return the same minimal per-question input currently used by the CLI."""

    return {
        "research_question": question,
        "iteration_count": 0,
        "is_goal_reached": False,
        "needs_replanning": False,
        "planned_actions": [],
        "next_action": "",
        "action_params": {},
    }


def _aggregator_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "research_question": "Почему меняется значение слова?",
        "goal": "Проверить изменение значения",
        "hypotheses": ["Контексты различаются"],
        "evidence": [
            {
                "source": "NKRJA",
                "action": "get_corpus_stats",
                "params": {"corpus": "MAIN"},
                "response": {"marker": "current"},
            }
        ],
        "last_evidence_batch": [],
        "planned_actions": [
            {"action": "get_corpus_stats", "params": {"corpus": "MAIN"}}
        ],
        "iteration_count": 0,
    }
    state.update(overrides)
    return state


def _disable_node_traces(monkeypatch: Any) -> None:
    for module in (planner_module, execution_module, orchestrator_module):
        monkeypatch.setattr(module, "emit_trace", lambda **_kwargs: None)


def test_orchestrator_returns_evidence_delta_without_doubling(
    fake_nkrja: Any,
    monkeypatch: Any,
) -> None:
    """A reducer-backed field must receive only the newly produced records."""

    monkeypatch.setattr(orchestrator_module, "emit_trace", lambda **_kwargs: None)
    fake_nkrja.queue("get_corpus_stats", {"marker": "new"})

    old_evidence = {
        "source": "NKRJA",
        "action": "get_corpus_stats",
        "params": {"corpus": "SPOKEN"},
        "response": {"marker": "old"},
    }
    state = {
        "research_question": "Одинакова ли частотность?",
        "evidence": [old_evidence],
        "planned_actions": [
            {"action": "get_corpus_stats", "params": {"corpus": "MAIN"}}
        ],
        "next_action": "",
        "action_params": {},
    }

    update = orchestrator_module.api_orchestrator_node(state)

    # This is how LangGraph applies ResearchState.evidence's operator.add
    # reducer.  Returning old + new from the node makes the old record appear
    # twice after the reducer is applied.
    merged_evidence = state["evidence"] + update["evidence"]

    assert [item["response"]["marker"] for item in merged_evidence] == [
        "old",
        "new",
    ]


def test_empty_plan_reaches_terminal_route_instead_of_looping() -> None:
    """No actions and no evidence must terminate with a controlled response."""

    state = _aggregator_state(
        evidence=[],
        last_evidence_batch=[],
        planned_actions=[],
        iteration_count=0,
        is_goal_reached=False,
        needs_replanning=False,
        next_action="",
    )

    update = evidence_module.evidence_aggregator_node(state, config={})
    route = graph_module.evidence_router({**state, **update})

    assert route == "assistant"


def test_empty_current_batch_does_not_reanalyse_previous_evidence(
    fake_llm: Any,
) -> None:
    """An empty batch is not permission to feed the last old record to the LLM."""

    fake_llm.queue(
        {
            "new_facts": ["STALE_FACT"],
            "is_goal_reached": True,
            "needs_replanning": False,
            "reasoning": "Старый пакет был ошибочно проанализирован повторно",
        }
    )
    state = _aggregator_state(
        evidence=[
            {
                "source": "NKRJA",
                "action": "get_corpus_stats",
                "params": {"corpus": "MAIN"},
                "response": {"marker": "previous-iteration"},
            }
        ],
        last_evidence_batch=[],
        planned_actions=[],
    )

    evidence_module.evidence_aggregator_node(state, config={})

    assert fake_llm.calls == []


def test_string_false_is_normalized_to_boolean_false(fake_llm: Any) -> None:
    """The string ``"false"`` must never become a truthy routing flag."""

    fake_llm.queue(
        {
            "new_facts": [],
            "is_goal_reached": "false",
            "needs_replanning": "false",
            "reasoning": "Данных пока недостаточно",
        }
    )

    update = evidence_module.evidence_aggregator_node(
        _aggregator_state(),
        config={},
    )

    assert update["is_goal_reached"] is False
    assert update["needs_replanning"] is False


def test_second_research_in_same_thread_does_not_mix_first_research(
    fake_llm: Any,
    fake_nkrja: Any,
    monkeypatch: Any,
) -> None:
    """A persistent conversation may keep dialogue, but not research-local data."""

    _disable_node_traces(monkeypatch)
    fake_llm.queue(
        {
            "mode": "research",
            "mode_reasoning": "Первая самостоятельная задача",
            "goal": "FIRST_GOAL",
            "hypotheses": ["FIRST_HYPOTHESIS"],
            "research_plan": ["FIRST_STEP"],
            "recommended_corpus": "MAIN",
            "corpus_reasoning": "Первый корпус",
        },
        {
            "planned_actions": [
                {"action": "get_corpus_stats", "params": {"corpus": "MAIN"}}
            ],
            "reasoning": "Первый вызов",
        },
        {
            "new_facts": ["FIRST_FACT"],
            "is_goal_reached": True,
            "needs_replanning": False,
            "reasoning": "Первая задача завершена",
        },
        "FIRST_ANSWER",
        {
            "mode": "research",
            "mode_reasoning": "Вторая самостоятельная задача",
            "goal": "SECOND_GOAL",
            "hypotheses": ["SECOND_HYPOTHESIS"],
            "research_plan": ["SECOND_STEP"],
            "recommended_corpus": "SPOKEN",
            "corpus_reasoning": "Второй корпус",
        },
        {
            "planned_actions": [
                {"action": "get_corpus_stats", "params": {"corpus": "SPOKEN"}}
            ],
            "reasoning": "Второй вызов",
        },
        {
            "new_facts": ["SECOND_FACT"],
            "is_goal_reached": True,
            "needs_replanning": False,
            "reasoning": "Вторая задача завершена",
        },
        "SECOND_ANSWER",
    )
    fake_nkrja.queue(
        "get_corpus_stats",
        {"marker": "FIRST_EVIDENCE"},
        {"marker": "SECOND_EVIDENCE"},
    )

    app = graph_module.workflow.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "regression-shared-thread"}}

    app.invoke(_research_input("Первый независимый вопрос"), config=config)
    second_result = app.invoke(
        _research_input("Второй независимый вопрос"),
        config=config,
    )

    assert second_result["goal"] == "SECOND_GOAL"
    assert second_result["hypotheses"] == ["SECOND_HYPOTHESIS"]
    assert second_result["facts"] == ["SECOND_FACT"]
    assert all(
        item.get("response", {}).get("marker") != "FIRST_EVIDENCE"
        for item in second_result["evidence"]
    )


def test_safeguard_warning_is_preserved_in_aggregator_prompt(
    fake_llm: Any,
) -> None:
    """A warning emitted beside a result must survive batch selection/formatting."""

    warning_text = (
        "PORTRAIT_MORPHEME недоступен для корпуса SPOKEN; повторять вызов нельзя"
    )
    warning = {
        "source": "System_Safeguard",
        "action": "get_word_portrait",
        "status": "warning",
        "message": warning_text,
    }
    result = {
        "source": "NKRJA",
        "action": "get_word_portrait",
        "params": {
            "lemma": "мать",
            "corpus": "SPOKEN",
            "resultType": ["PORTRAIT_CONCORDANCE"],
        },
        "response": {"marker": "supported-part"},
    }
    fake_llm.queue(
        {
            "new_facts": [],
            "is_goal_reached": True,
            "needs_replanning": False,
            "reasoning": "Доступная часть обработана",
        }
    )

    evidence_module.evidence_aggregator_node(
        _aggregator_state(
            evidence=[warning, result],
            last_evidence_batch=[warning, result],
            planned_actions=[
                {
                    "action": "get_word_portrait",
                    "params": result["params"],
                }
            ],
        ),
        config={},
    )

    messages = fake_llm.calls[0].input
    human_prompt = messages[-1].content
    assert warning_text in human_prompt


def test_equal_questions_use_distinct_explicit_run_ids(
    fake_llm: Any,
    monkeypatch: Any,
) -> None:
    """The question text is payload, not an execution identifier."""

    emitted_run_ids: list[str] = []

    def capture_trace(**event: Any) -> None:
        emitted_run_ids.append(event["run_id"])

    monkeypatch.setattr(planner_module, "emit_trace", capture_trace)
    fake_llm.queue(
        {"mode": "chat", "mode_reasoning": "Первый запуск"},
        {"mode": "chat", "mode_reasoning": "Второй запуск"},
    )

    planner_module.planner_node(
        {
            "research_question": "Одинаковый текст вопроса",
            "run_id": "run-001",
            "research_id": "research-001",
            "facts": [],
        }
    )
    planner_module.planner_node(
        {
            "research_question": "Одинаковый текст вопроса",
            "run_id": "run-002",
            "research_id": "research-002",
            "facts": [],
        }
    )

    assert emitted_run_ids == ["run-001", "run-002"]


def test_trace_uses_current_iteration_instead_of_default_zero(
    fake_llm: Any,
    monkeypatch: Any,
) -> None:
    """Every node trace must carry the branch's real iteration number."""

    emitted_iterations: list[int] = []

    def capture_trace(**event: Any) -> None:
        emitted_iterations.append(event.get("iteration", 0))

    monkeypatch.setattr(execution_module, "emit_trace", capture_trace)
    fake_llm.queue(
        {
            "planned_actions": [],
            "reasoning": "На четвёртой итерации новых действий нет",
        }
    )

    execution_module.execution_planner_node(
        {
            "research_question": "Проверить трассировку",
            "run_id": "run-iteration-check",
            "iteration_count": 3,
            "goal": "Проверить номер итерации",
            "hypotheses": [],
            "research_plan": [],
            "recommended_corpus": "MAIN",
            "corpus_reasoning": "Тест",
            "evidence": [],
            "facts": [],
        },
        config={},
    )

    assert emitted_iterations == [3]
