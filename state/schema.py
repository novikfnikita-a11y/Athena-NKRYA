# state/schema.py

import operator
from typing import TypedDict, Annotated, Any, Literal


class ResearchState(TypedDict):
    # исходная постановка задачи
    research_question: str
    goal: str
    final_response: str
    iteration_count: int

    # НОВОЕ: режим работы графа, определяется planner_node на каждом входе в узел.
    # "research" - нужен полный цикл сбора данных через API НКРЯ
    # "chat"     - уточняющий вопрос по уже собранным фактам, новые API-вызовы не нужны
    mode: Literal["research", "chat"]

    # план оркестрации
    research_plan: list[str]  # Общий пошаговый план от Планнера
    current_step_index: int  # На каком шаге плана мы сейчас находимся

    # ВАРИАНТ Б: решение о корпусе принимается ОДИН РАЗ на этапе planner_node,
    # а не заново на каждой итерации execution_planner. Поля перезаписываются
    # целиком (без operator.add), т.к. это единое актуальное решение, а не история.
    recommended_corpus: str      # одно из значений CorpusTypeEnum, выбранное планировщиком
    corpus_reasoning: str        # обоснование выбора корпуса (для прозрачности методологии)

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

    needs_replanning: bool # True - возврат к Planner, False - возврат к Execution
    pagination_context: dict  # Хранит состояние пагинаци
    # управление графом
    confidence: float
    is_goal_reached: bool

    # очередь задач. полностью перезаписывается на каждой итерации (без operator.add)
    planned_actions: list[dict[str, Any]]

    # LEGACY от версии без батча на оркестрации
    next_action: str
    action_params: dict