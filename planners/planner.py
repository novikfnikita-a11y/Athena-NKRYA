"""Typed asynchronous top-level research planning."""

import asyncio
import hashlib
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from LLM.client import get_llm
from LLM.prompts import (
    PLANNER_CLASSIFICATION_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
)
from LLM.structured import (
    ClassificationOutput,
    PlannerOutput,
    StructuredOutputError,
    ainvoke_structured,
)
from state.models import (
    ErrorKind,
    PlannerDecision,
    ResearchMode,
    ResearchSubGoal,
)
from state.schema import ResearchState
from tools.registry import CORPUS_TYPE_ENUM_DESCRIPTION, CORPUS_VALUES
from utils.trace import emit_trace


def _branch_id(research_id: str, index: int, corpus: str, goal: str) -> str:
    digest = hashlib.sha256(
        f"{research_id}\0{index}\0{corpus}\0{goal}".encode("utf-8")
    ).hexdigest()[:16]
    return f"branch-{index + 1}-{digest}"


def _scope(state: ResearchState) -> tuple[str, str]:
    research_id = str(state.get("research_id") or "research-unassigned")
    run_id = str(state.get("run_id") or "run-unassigned")
    return research_id, run_id


def _safe_json(value: Any) -> str:
    def encode(item: Any) -> Any:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        return str(item)

    return json.dumps(
        value,
        ensure_ascii=False,
        default=encode,
        sort_keys=True,
    )


def _trace_scope(state: ResearchState) -> dict[str, Any]:
    return {
        "thread_id": state.get("thread_id"),
        "turn_id": state.get("turn_id"),
        "research_id": state.get("research_id"),
        "branch_id": state.get("branch_id"),
        "batch_id": state.get("batch_id"),
        "iteration": state.get("iteration_count", 0),
    }


def _decision_from_output(
    output: PlannerOutput,
    state: ResearchState,
) -> PlannerDecision:
    research_id, run_id = _scope(state)
    max_branches = int(state.get("budgets", {}).get("max_branches", 4))
    if len(output.sub_goals) > max_branches:
        raise ValueError("planner output exceeds max_branches")
    sub_goals: list[ResearchSubGoal] = []
    for index, item in enumerate(output.sub_goals):
        if item.corpus not in CORPUS_VALUES:
            raise ValueError(f"unknown corpus in planner output: {item.corpus}")
        sub_goals.append(
            ResearchSubGoal(
                research_id=research_id,
                run_id=run_id,
                branch_id=_branch_id(research_id, index, item.corpus, item.goal),
                goal=item.goal,
                corpus=item.corpus,
                corpus_reasoning=item.corpus_reasoning,
                lemma=item.lemma,
                hypotheses=item.hypotheses,
                research_plan=item.research_plan,
            )
        )
    return PlannerDecision(
        research_id=research_id,
        run_id=run_id,
        mode=output.mode,
        reasoning=output.reasoning,
        goal=output.goal,
        sub_goals=tuple(sub_goals),
        error_kind=output.error_kind,
        error_message=output.error_message,
    )


def _decision_update(
    decision: PlannerDecision,
    state: ResearchState,
) -> dict[str, Any]:
    if decision.mode is ResearchMode.ERROR:
        return {
            "mode": "error",
            "planner_route": "error",
            "planner_reasoning": decision.reasoning,
            "research_status": "error",
            "termination_reason": "error",
            "error": {
                "kind": (decision.error_kind or ErrorKind.MODEL).value,
                "message": decision.error_message or "Не удалось сформировать план.",
                "node": "planner",
                "retryable": False,
            },
            "needs_replanning": False,
            "sub_goals": [],
        }
    if decision.mode is ResearchMode.CHAT:
        return {
            "mode": "chat",
            "planner_route": "chat",
            "planner_reasoning": decision.reasoning,
            "research_status": "running",
            "needs_replanning": False,
            "sub_goals": [],
        }

    first = decision.sub_goals[0]
    usage = dict(state.get("budget_usage", {}))
    usage["branches"] = len(decision.sub_goals)
    return {
        "mode": "research",
        "planner_route": "research",
        "planner_reasoning": decision.reasoning,
        "research_status": "running",
        "overall_goal": decision.goal or first.goal,
        "goal": first.goal,
        "sub_goals": [item.model_dump(mode="json") for item in decision.sub_goals],
        # Single-branch compatibility fields remain until Stage 4 dispatches
        # every sub-goal through Send.
        "branch_id": first.branch_id,
        "recommended_corpus": first.corpus,
        "corpus_reasoning": first.corpus_reasoning,
        "hypotheses": list(first.hypotheses),
        "research_plan": list(first.research_plan),
        "budget_usage": usage,
        "needs_replanning": False,
        "termination_reason": None,
        "error": None,
    }


async def planner_node_async(
    state: ResearchState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Classify a follow-up when needed, then produce a typed plan."""

    llm = get_llm()
    question = str(state.get("research_question") or "").strip()
    run_id = str(state.get("run_id") or "run-unassigned")
    trace_scope = _trace_scope(state)
    is_replan = bool(state.get("needs_replanning")) or state.get("mode") == "replan"

    try:
        context_id = state.get("selected_context_research_id")
        if context_id and not is_replan:
            snapshot = state.get("research_archive", {}).get(context_id, {})
            classification_context = (
                f"Новый вопрос: {question}\n"
                f"Прошлый вопрос: {snapshot.get('question', '')}\n"
                f"Прошлая цель: {snapshot.get('goal', '')}\n"
                f"Разрешённые прошлые факты: {_safe_json(state.get('context_facts', []))}"
            )
            classification = await ainvoke_structured(
                llm,
                [
                    SystemMessage(content=PLANNER_CLASSIFICATION_SYSTEM_PROMPT),
                    HumanMessage(content=classification_context),
                ],
                ClassificationOutput,
                node="planner_classification",
                config=config,
            )
            emit_trace(
                node="planner",
                event_type="planning",
                content={
                    "mode": classification.mode,
                    "reasoning_summary": classification.reasoning,
                },
                run_id=run_id,
                **trace_scope,
            )
            if classification.mode == "chat":
                decision = PlannerDecision(
                    research_id=_scope(state)[0],
                    run_id=_scope(state)[1],
                    mode=ResearchMode.CHAT,
                    reasoning=classification.reasoning,
                )
                return _decision_update(decision, state)

        system_prompt = PLANNER_SYSTEM_PROMPT.format(
            corpus_registry=CORPUS_TYPE_ENUM_DESCRIPTION
        )
        if is_replan:
            user_context = (
                f"Режим: replan\nВопрос: {question}\n"
                f"Текущая цель: {state.get('goal', '')}\n"
                f"Текущий план: {_safe_json(state.get('research_plan', []))}\n"
                f"Подтверждённые факты: {_safe_json(state.get('facts', []))}\n"
                f"Выполненные действия: {_safe_json(state.get('completed_actions', []))}\n"
                f"Непокрытая информация: {_safe_json(state.get('missing_information', []))}\n"
                f"Максимальное число корпусных подцелей: "
                f"{state.get('budgets', {}).get('max_branches', 4)}\n"
                "Пересобери методику только для текущего исследования. mode=replan."
            )
        else:
            user_context = (
                f"Новая самостоятельная задача: {question}\n"
                f"Максимальное число корпусных подцелей: "
                f"{state.get('budgets', {}).get('max_branches', 4)}\n"
                "Сформируй чистый план только из этого вопроса. Прошлые факты не доступны. "
                "Если нужен корпус, mode=research."
            )

        output = await ainvoke_structured(
            llm,
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_context),
            ],
            PlannerOutput,
            node="planner",
            config=config,
        )
        if is_replan and output.mode is not ResearchMode.REPLAN:
            raise ValueError("replan invocation must return mode=replan")
        if not is_replan and output.mode is ResearchMode.REPLAN:
            raise ValueError("a new task cannot return mode=replan")
        if not context_id and output.mode is ResearchMode.CHAT:
            raise ValueError("chat mode requires an explicitly selected context")
        decision = _decision_from_output(output, state)
        emit_trace(
            node="planner",
            event_type="decision",
            content={
                "mode": decision.mode.value,
                "reasoning_summary": decision.reasoning,
                "goal": decision.goal,
                "sub_goals": [
                    {"branch_id": item.branch_id, "corpus": item.corpus}
                    for item in decision.sub_goals
                ],
            },
            run_id=run_id,
            **trace_scope,
        )
        return _decision_update(decision, state)
    except StructuredOutputError as error:
        return {
            "mode": "error",
            "planner_route": "error",
            "planner_reasoning": error.result.message,
            "research_status": "error",
            "termination_reason": "error",
            "error": error.result.as_safe_error(),
            "needs_replanning": False,
            "sub_goals": [],
        }
    except Exception:
        return {
            "mode": "error",
            "planner_route": "error",
            "planner_reasoning": "План не прошёл доменную проверку.",
            "research_status": "error",
            "termination_reason": "error",
            "error": {
                "kind": "validation",
                "message": "План исследования не соответствует контракту.",
                "node": "planner",
                "retryable": False,
            },
            "needs_replanning": False,
            "sub_goals": [],
        }


def planner_node(
    state: ResearchState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Synchronous Stage-3 adapter; Stage 4 wires the async node directly."""

    return asyncio.run(planner_node_async(state, config))


__all__ = ["planner_node", "planner_node_async"]
