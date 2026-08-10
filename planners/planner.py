"""Planner node with an isolation boundary between classification and planning."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from LLM.client import get_llm
from LLM.prompts import PLANNER_SYSTEM_PROMPT
from state.schema import ResearchState
from tools.registry import CORPUS_TYPE_ENUM_DESCRIPTION
from utils.trace import emit_trace


def _decode_response(response: Any) -> dict[str, Any]:
    content = response.content.strip().replace("```json", "").replace("```", "")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("planner response must be a JSON object")
    return data


def planner_node(state: ResearchState) -> dict[str, Any]:
    print("\n--- УЗЕЛ: Linguistic Task Planner (Запуск ИИ) ---")

    run_id = state.get("run_id", "default-run")
    trace_scope = {
        "thread_id": state.get("thread_id"),
        "turn_id": state.get("turn_id"),
        "research_id": state.get("research_id"),
        "branch_id": state.get("branch_id"),
        "batch_id": state.get("batch_id"),
        "iteration": state.get("iteration_count", 0),
    }
    llm = get_llm()
    system_prompt = PLANNER_SYSTEM_PROMPT.format(
        corpus_registry=CORPUS_TYPE_ENUM_DESCRIPTION
    )
    question = state.get("research_question", "")
    feedback = state.get("aggregator_reasoning", "")

    try:
        classified_mode: str | None = None
        mode_reasoning = "Обоснование режима не предоставлено"

        # A previous research context is visible only to this classification
        # call.  If the question is new, the actual planning call below receives
        # no previous facts and cannot blend two independent investigations.
        context_id = state.get("selected_context_research_id")
        if not feedback and context_id:
            snapshot = state.get("research_archive", {}).get(context_id, {})
            context_facts = state.get("context_facts", [])
            classifier_content = (
                f"Новый вопрос пользователя: {question}\n\n"
                f"Предыдущий вопрос: {snapshot.get('question', 'не указан')}\n"
                f"Предыдущая цель: {snapshot.get('goal', 'не указана')}\n"
                f"Факты предыдущего исследования: {context_facts}\n\n"
                "Выполни только классификацию связи вопросов. Верни JSON с полями "
                'mode (строго "chat" или "research") и mode_reasoning. '
                '"chat" допустим только для уточнения, которое можно обработать по '
                "явно переданным фактам; самостоятельная задача — это research."
            )
            classification = _decode_response(
                llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=classifier_content),
                    ]
                )
            )
            classified_mode = str(classification.get("mode", "research"))
            if classified_mode not in {"chat", "research"}:
                raise ValueError("classifier mode must be chat or research")
            mode_reasoning = str(
                classification.get(
                    "mode_reasoning",
                    "Обоснование режима не предоставлено",
                )
            )
            emit_trace(
                node="planner",
                event_type="planning",
                content={"mode": classified_mode, "reasoning": mode_reasoning},
                run_id=run_id,
                **trace_scope,
            )
            if classified_mode == "chat":
                return {
                    "mode": "chat",
                    "planner_route": "chat",
                    "research_status": "running",
                    "needs_replanning": False,
                }

        if feedback:
            user_content = (
                f"Исследовательский вопрос пользователя: {question}\n\n"
                f"ОБРАТНАЯ СВЯЗЬ ТЕКУЩЕГО ИССЛЕДОВАНИЯ:\n{feedback}\n\n"
                "Пересобери цель, гипотезы, план и выбор корпуса только для текущего "
                'исследования. mode обязан быть "research".'
            )
        elif classified_mode == "research":
            user_content = (
                f"Новая самостоятельная исследовательская задача: {question}\n\n"
                "Сформируй цель, гипотезы, план и выбор корпуса исключительно из "
                'этого вопроса. Не используй сведения предыдущего исследования. mode="research".'
            )
        else:
            user_content = (
                f"Исследовательский вопрос пользователя: {question}\n\n"
                "Предыдущий исследовательский контекст отсутствует. Определи режим; "
                'если требуется работа с НКРЯ, верни mode="research" вместе с целью, '
                "гипотезами, планом и выбором корпуса."
            )

        data = _decode_response(
            llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_content),
                ]
            )
        )
        mode = "research" if classified_mode == "research" or feedback else str(
            data.get("mode", "research")
        )
        if mode not in {"chat", "research"}:
            raise ValueError("planner mode must be chat or research")
        mode_reasoning = str(data.get("mode_reasoning", mode_reasoning))
        print(f"[Planner] Определён режим: {mode} ({mode_reasoning})")

        if classified_mode is None:
            emit_trace(
                node="planner",
                event_type="planning",
                content={"mode": mode, "reasoning": mode_reasoning},
                run_id=run_id,
                **trace_scope,
            )
        if mode == "chat":
            return {
                "mode": "chat",
                "planner_route": "chat",
                "research_status": "running",
                "needs_replanning": False,
            }

        recommended_corpus = data.get("recommended_corpus", "MAIN")
        corpus_reasoning = data.get(
            "corpus_reasoning", "Обоснование не предоставлено"
        )
        emit_trace(
            node="planner",
            event_type="decision",
            content={
                "goal": data.get("goal"),
                "recommended_corpus": recommended_corpus,
                "corpus_reasoning": corpus_reasoning,
            },
            run_id=run_id,
            **trace_scope,
        )

        print(f"[Planner] Сформулирована цель: {data.get('goal')}")
        print(f"[Planner] Выдвинуто гипотез: {len(data.get('hypotheses', []))}")
        print(
            f"[Planner] Рекомендованный корпус: {recommended_corpus} "
            f"({corpus_reasoning})"
        )
        return {
            "mode": "research",
            "planner_route": "research",
            "research_status": "running",
            "goal": data.get("goal", "Цель не определена"),
            "hypotheses": data.get("hypotheses", []),
            "research_plan": data.get("research_plan", []),
            "recommended_corpus": recommended_corpus,
            "corpus_reasoning": corpus_reasoning,
            "needs_replanning": False,
        }
    except Exception as error:
        print(f"[Planner] Ошибка парсинга или вызова модели: {error}")
        return {
            "mode": "error",
            "planner_route": "error",
            "research_status": "error",
            "termination_reason": "error",
            "error": {
                "kind": "model",
                "message": "Не удалось сформировать план исследования.",
                "node": "planner",
                "retryable": True,
            },
            "needs_replanning": False,
        }


__all__ = ["planner_node"]
