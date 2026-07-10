import pprint
from typing import Literal, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from planners.planner import planner_node
from planners.execution import execution_planner_node
from tools.orchestrator import api_orchestrator_node
from research.evidence import evidence_aggregator_node
from research.assistant import assistant_node
from state.schema import ResearchState


def mode_router(state: ResearchState) -> Literal["execution_planner", "assistant"]:
    """
    Развилка сразу после planner_node - соответствует разветвлению
    mode=research / mode=chat на схеме.
    """
    if state.get("mode") == "chat":
        print("\n[Router] mode=chat: пропускаем сбор данных, сразу к финальному ответу.")
        return "assistant"

    print("\n[Router] mode=research: переходим к execution_planner.")
    return "execution_planner"


def evidence_router(state: ResearchState) -> Literal["assistant", "planner", "execution_planner"]:
    """
    Развилка после evidence_aggregator_node. Три исхода, соответствующие схеме:
      - "assistant"         -> цель достигнута или исчерпан лимит итераций
      - "planner"           -> needs_replanning=True, редкая красная петля (методология неверна)
      - "execution_planner" -> онужно просто вызвать ещё инструменты
    """
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

# planner -> {execution_planner, assistant}, в зависимости от mode
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

# evidence_aggregator -> {assistant, planner, execution_planner}
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

# НОВОЕ: чекпоинтер даёт графу память в рамках thread_id между вызовами app.invoke().
# Без него planner_node на втором вопросе не увидит facts/goal предыдущего цикла,
# и mode="chat" в принципе не мог бы сработать - проверять было бы нечего.
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


if __name__ == "__main__":

    '''
    try:
        with open("graph_visualization.png", "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print("схема графа успешно сохранена в файл: graph_visualization.png\n")
    except Exception as e:
        print(f"ошибка  {e}\n") '''

    # НОВОЕ: интерактивный цикл вместо одноразового invoke.
    # thread_id объединяет все вопросы сессии в одну историю state -
    # именно за счёт этого planner_node на втором и последующих вопросах
    # видит накопленные facts/goal и может выбрать mode="chat".
    config = {"configurable": {"thread_id": "cli-session-1"}}

    print("Исследовательский агент НКРЯ. Введите вопрос (или 'exit' для выхода).\n")

    while True:
        user_question = input("Ваш запрос в Национальный Корпус Русского Языка: ").strip()
        if user_question.lower() in ("exit", "quit", "выход"):
            print("Сессия завершена.")
            break
        if not user_question:
            continue


        invoke_input: dict[str, Any] = {
            "research_question": user_question,
            "iteration_count": 0,
            "is_goal_reached": False,
            "needs_replanning": False,
            "planned_actions": [],
            "next_action": "",
            "action_params": {}
        }

        final_output = app.invoke(invoke_input, config=config)

        print("\n-ОТВЕТ АГЕНТА ")
        print(final_output.get("final_response", "Ответ не сформирован"))
        print("--------------------\n")