import os
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from planners.planner import planner_node
from planners.execution import execution_planner_node
from tools.orchestrator import api_orchestrator_node
from research.evidence import evidence_aggregator_node
from research.assistant import assistant_node
from state.schema import ResearchState

def mode_router(state: ResearchState) -> Literal["execution_planner", "assistant"]:
    if state.get("mode") == "chat":
        print("\n[Router] mode=chat: пропускаем сбор данных, сразу к финальному ответу.")
        return "assistant"
    print("\n[Router] mode=research: переходим к execution_planner.")
    return "execution_planner"

def evidence_router(state: ResearchState) -> Literal["assistant", "planner", "execution_planner"]:
    if state.get("is_goal_reached") or state.get("next_action") == "finish":
        print("\n[Router] Завершаем исследование. Переход к генерации финального ответа.")
        return "assistant"

    max_iterations = 4
    current_iterations = state.get("iteration_count", 0)

    if current_iterations >= max_iterations:
        print(f"\n[Router] ПРЕДОХРАНИТЕЛЬ: Лимит итераций ({max_iterations}) исчерпан!")
        print("[Router] Принудительно завершаем исследование и передаем то, что успели собрать.")
        return "assistant"

    if state.get("needs_replanning"):
        print(f"\n[Router] Аналитик требует ПОЛНОГО переосмысления методологии (шаг {current_iterations + 1}/{max_iterations}). Возврат к planner.")
        return "planner"

    print(f"\n[Router] Цель не достигнута (шаг {current_iterations + 1}/{max_iterations}). Донабор инструментов через execution_planner.")
    return "execution_planner"

workflow = StateGraph(ResearchState)

workflow.add_node("planner", planner_node)
workflow.add_node("execution_planner", execution_planner_node)
workflow.add_node("api_orchestrator", api_orchestrator_node)
workflow.add_node("evidence_aggregator", evidence_aggregator_node)
workflow.add_node("assistant", assistant_node)

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    mode_router,
    {
        "execution_planner": "execution_planner",
        "assistant": "assistant"
    }
)

workflow.add_edge("execution_planner", "api_orchestrator")
workflow.add_edge("api_orchestrator", "evidence_aggregator")

workflow.add_conditional_edges(
    "evidence_aggregator",
    evidence_router,
    {
        "assistant": "assistant",
        "planner": "planner",
        "execution_planner": "execution_planner"
    }
)

workflow.add_edge("assistant", END)

# Условная компиляция графа:
# Переменная будет равна "1" ТОЛЬКО если мы запускаем локальный main.py
if os.getenv("USE_LOCAL_CHECKPOINTER") == "1":
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    print(" [Гgраф] Скомпилирован С ЛОКАЛЬНОЙ ПАМЯТЬЮ MemorySaver")
else:
    app = workflow.compile()
    print(" [Граф] Скомпилирован БЕЗ встроенной памяти для langgraph studio)")