"""Aggregate exactly one explicit evidence batch into research state."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from LLM.client import get_llm
from LLM.prompts import EVIDENCE_AGGREGATOR_SYSTEM_PROMPT
from state.schema import ResearchState
from tools.registry import get_registry_description


def _strict_boolean(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(f"{field_name} must be a boolean")


def _format_evidence(artifact: dict[str, Any], index: int) -> str:
    action = artifact.get("action") or artifact.get("tool", "unknown_action")
    params = artifact.get("params", {})
    status = artifact.get("status", "success")
    header = f"--- ДЕЙСТВИЕ {index}: {action} | ПАРАМЕТРЫ: {params} ---\n"
    if status in {"warning", "error", "unsupported"}:
        return header + f"СТАТУС: {str(status).upper()}\nСООБЩЕНИЕ: {artifact.get('message', 'не указано')}"
    payload = artifact.get("response", artifact.get("payload", {}))
    return header + json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def evidence_aggregator_node(
    state: ResearchState,
    config: RunnableConfig,
) -> dict[str, Any]:
    print("\n--- УЗЕЛ: EVIDENCE AGGREGATOR ---")

    current_batch = list(state.get("last_evidence_batch", []))
    current_iteration = state.get("iteration_count", 0)
    if not current_batch:
        print("[Aggregator] Текущий пакет пуст; старые evidence повторно не анализируются.")
        return {
            "is_goal_reached": False,
            "needs_replanning": False,
            "aggregator_status": "complete",
            "aggregator_reasoning": "Новых данных и действий для продолжения нет.",
            "termination_reason": "no_actions",
            "research_status": "partial",
            "last_evidence_batch": [],
            "planned_actions": [],
        }

    print(f"[Aggregator] Анализируем явный пакет из {len(current_batch)} артефактов.")
    blocks = [_format_evidence(dict(item), index) for index, item in enumerate(current_batch, 1)]
    budget = state.get("budgets", {}).get("max_context_chars", 300_000)
    selected_blocks: list[str] = []
    used = 0
    truncated = False
    for block in blocks:
        separator_size = 2 if selected_blocks else 0
        if used + separator_size + len(block) > budget:
            truncated = True
            break
        selected_blocks.append(block)
        used += separator_size + len(block)
    raw_data = "\n\n".join(selected_blocks)
    if truncated:
        raw_data += "\n\n[ПАКЕТ ЧАСТИЧНО УСЕЧЁН ПО ГРАНИЦЕ АРТЕФАКТОВ]"

    context_message = (
        f"Вопрос пользователя: {state.get('research_question', '')}\n"
        f"Текущая цель исследования: {state.get('goal', '')}\n"
        f"Проверяемые гипотезы: {state.get('hypotheses', [])}\n\n"
        f"Новый пакет данных и системных ограничений:\n{raw_data}"
    )
    messages = [
        SystemMessage(
            content=EVIDENCE_AGGREGATOR_SYSTEM_PROMPT.format(
                registry_description=get_registry_description()
            )
        ),
        HumanMessage(content=context_message),
    ]

    try:
        response = get_llm().invoke(messages, config=config)
        content = response.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(content)
        new_deductions = data.get("new_facts", [])
        if not isinstance(new_deductions, list):
            raise ValueError("new_facts must be a list")
        if not all(isinstance(item, str) and item.strip() for item in new_deductions):
            raise ValueError("new_facts must contain non-empty text interpretations")
        is_goal_reached = _strict_boolean(
            data.get("is_goal_reached", False), "is_goal_reached"
        )
        needs_replanning = _strict_boolean(
            data.get("needs_replanning", False), "needs_replanning"
        )
        reasoning = str(data.get("reasoning", "Нет обоснования")).strip() or "Нет обоснования"
        missing = data.get("missing_information", [])
        if not isinstance(missing, list):
            missing = []

        if is_goal_reached:
            aggregator_status = "complete"
            termination_reason = "goal_reached"
            research_status = "complete"
        elif needs_replanning:
            aggregator_status = "replan"
            termination_reason = None
            research_status = "running"
        else:
            aggregator_status = "continue"
            termination_reason = None
            research_status = "running"

        print(f"[Aggregator] Извлечено новых интерпретаций: {len(new_deductions)}")
        print(f"[Aggregator] Цель достигнута: {is_goal_reached} ({reasoning})")
        return {
            # LLM output is interpretation, never an observable Fact. Numeric
            # facts enter state exclusively through research.fact_extractor.
            "deductions": new_deductions,
            "is_goal_reached": is_goal_reached,
            "needs_replanning": needs_replanning,
            "aggregator_status": aggregator_status,
            "aggregator_reasoning": reasoning,
            "missing_information": missing,
            "iteration_count": current_iteration + 1,
            "termination_reason": termination_reason,
            "research_status": research_status,
            "last_evidence_batch": [],
            "planned_actions": [],
        }
    except Exception as error:
        print(f"[Aggregator] Контролируемая ошибка анализа: {error}")
        return {
            "is_goal_reached": False,
            "needs_replanning": False,
            "aggregator_status": "error",
            "aggregator_reasoning": "Не удалось проверить новый пакет evidence.",
            "iteration_count": current_iteration + 1,
            "termination_reason": "error",
            "research_status": "error",
            "error": {
                "kind": "validation",
                "message": "Некорректный ответ агрегатора evidence.",
                "node": "evidence_aggregator",
                "retryable": True,
            },
            "last_evidence_batch": [],
            "planned_actions": [],
        }


__all__ = ["evidence_aggregator_node"]
