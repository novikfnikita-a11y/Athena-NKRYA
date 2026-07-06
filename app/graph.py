import pprint
from typing import Literal, Any
from langgraph.graph import StateGraph, END
from planners.planner import planner_node
from planners.execution import execution_planner_node
from tools.orchestrator import api_orchestrator_node
from research.evidence import evidence_aggregator_node
from research.assistant import assistant_node
from state.schema import ResearchState



def router(state: ResearchState) -> Literal["assistant", "planner"]:
    if state.get("is_goal_reached") or state.get("next_action") == "finish":
        print("\n[Router] Завершаем исследование. Переход к генерации финального ответа.")
        return "assistant"

    # BREAKER: Защита от бесконечного цикла
    # ВРЕМЕННО
        # ВРЕМЕННО
            #ВРЕМЕННО
                    # ХАРД КОД
    max_iterations = 4
    current_iterations = len(state.get("evidence", []))

    if current_iterations >= max_iterations:
        print(f"\n[Router] ПРЕДОХРАНИТЕЛЬ: Лимит итераций ({max_iterations}) исчерпан!")
        print("[Router] Принудительно завершаем исследование и передаем то, что успели собрать.")
        return "assistant"

    print(
        f"\n[Router] Цель не достигнута (итерация {current_iterations + 1}/{max_iterations}). Возврат к планировщику.")
    return "planner"



workflow = StateGraph(ResearchState)

workflow.add_node("planner", planner_node)
workflow.add_node("execution_planner", execution_planner_node)
workflow.add_node("api_orchestrator", api_orchestrator_node)
workflow.add_node("evidence_aggregator", evidence_aggregator_node)
workflow.add_node("assistant", assistant_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "execution_planner")
workflow.add_edge("execution_planner", "api_orchestrator")
workflow.add_edge("api_orchestrator", "evidence_aggregator")

workflow.add_conditional_edges(
    "evidence_aggregator",
    router,
    {
        "assistant": "assistant",
        "planner": "planner"
    }
)
workflow.add_edge("assistant", END)

app = workflow.compile()


if __name__ == "__main__":


    """ визуализация системы в  пнг 
    try:
        with open("graph_visualization.png", "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print("схема графа успешно сохранена в файл: graph_visualization.png\n")
    except Exception as e:
        print(f"ошибка  {e}\n")"""



# хард код базового состояния - временное решение . в будущем реализую краткосрочную память


    initial_state: dict[str, Any] = {
        "goal": "",
        "research_question": "Как часто в текстах встречается слово 'крипта' и в каких предложениях его используют? Найди пару примеров",
        "hypotheses": [],
        "open_questions": [],
        "evidence": [],
        "facts": [],
        "deductions": [],
        "confidence": 0.0,
        "is_goal_reached": False,
        "next_action": "",
        "action_params": {}
    }


    final_output = app.invoke(initial_state)

    print("\n ТЕСТ ЗАВЕРШЕН УСПЕШНО ")
    print("Финальное состояние памяти исследования (ResearchState):")

    # вывод в консоль
    pprint.pprint(final_output)