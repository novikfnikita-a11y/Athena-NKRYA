# state/schema.py

import operator
from typing import TypedDict, Annotated, Any


class ResearchState(TypedDict):
    # исходная постановка задачи
    research_question: str
    goal: str
    final_response: str

    # план оркестрации
    research_plan: list[str]  # Общий пошаговый план от Планнера
    current_step_index: int  # На каком шаге плана мы сейчас находимся

    # план исследования
    hypotheses: Annotated[list[str], operator.add]
    open_questions: Annotated[list[str], operator.add]

    # данные исследования
    evidence: Annotated[list[dict], operator.add]
    facts: Annotated[list[str], operator.add]
    deductions: Annotated[list[str], operator.add]

    # служебные поля агрегатора
    aggregator_reasoning: str
    missing_information: Annotated[list[str], operator.add]

    # история уже  выполненных действий
    completed_actions: Annotated[list[dict], operator.add]

    # управление графом
    confidence: float
    is_goal_reached: bool

    #
    # очередь задач. полностью перезаписывается на каждой итерации (без operator.add)
    planned_actions: list[dict[str, Any]]

    # LEGACY от версии без батча на оркестрации
    next_action: str
    action_params: dict