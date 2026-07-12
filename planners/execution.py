import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from LLM.client import get_llm
from LLM.prompts import EXECUTION_PLANNER_SYSTEM_PROMPT
from state.schema import ResearchState
from tools.registry import get_registry_description
from utils.trace import emit_trace

def execution_planner_node(state: ResearchState, config : RunnableConfig):
    print("\n--- УЗЕЛ: Execution Planner (Выбор инструмента) ---")
    run_id = state.get("research_question", "default_run")
    llm = get_llm()

    system_prompt_ready = EXECUTION_PLANNER_SYSTEM_PROMPT.format(
        registry_description=get_registry_description()
    )

    # 1. Извлекаем историю уже запущенных инструментов для защиты от зацикливания
    past_actions = [ev.get("action") for ev in state.get("evidence", []) if "action" in ev]
    past_actions_str = ", ".join(past_actions) if past_actions else "Действий еще не было"

    # 2. АНАЛИЗ ОШИБОК: Собираем все предупреждения и отказы бэкенда из прошлых шагов
    safeguard_warnings = []
    for ev in state.get("evidence", []):
        if ev.get("status") in ["error", "warning"]:
            source_label = ev.get("source", "System")
            safeguard_warnings.append(
                f"- [{source_label}] Инструмент '{ev.get('action')}' не сработал. "
                f"Статус: {ev.get('status').upper()}. Сообщение: {ev.get('message')}"
            )

    safeguard_warnings_str = (
        "\n".join(safeguard_warnings)
        if safeguard_warnings
        else "Ошибок выполнения или блокировок со стороны защитного слоя пока не зафиксировано."
    )

    # 3. Извлекаем уже накопленные факты
    current_facts = (
        "\n".join([f"- {fact}" for fact in state.get("facts", [])])
        if state.get("facts")
        else "Факты еще не собраны"
    )

    # 4. Извлекаем текущий план исследования
    research_plan = state.get("research_plan", [])
    research_plan_str = (
        "\n".join([f"{i + 1}. {step}" for i, step in enumerate(research_plan)])
        if research_plan
        else "План не задан (простой запрос)"
    )

    # 5. Формируем контекст для модели (теперь с блоком критических ошибок)
    context_message = (
        f"ОБЩАЯ ЦЕЛЬ (goal): {state.get('goal', 'Не задана')}\n"
        f"ГИПОТЕЗЫ (hypotheses): {state.get('hypotheses', [])}\n"
        f"ПЛАН ИССЛЕДОВАНИЯ (research_plan):\n{research_plan_str}\n\n"
        f"РЕКОМЕНДОВАННЫЙ КОРПУС (recommended_corpus): {state.get('recommended_corpus', 'MAIN')}\n"
        f"ОБОСНОВАНИЕ ВЫБОРА КОРПУСА: {state.get('corpus_reasoning', 'Не указано')}\n\n"
        f"УЖЕ ВЫПОЛНЕННЫЕ ДЕЙСТВИЯ (ИНСТРУМЕНТЫ): {past_actions_str}\n"
        f"УЖЕ УСТАНОВЛЕННЫЕ ФАКТЫ:\n{current_facts}\n\n"
        f"КРИТИЧЕСКИ ВАЖНО! ОШИБКИ И ОГРАНИЧЕНИЯ БЭКЕНДА ИЗ ПРЕДЫДУЩИХ ШАГОВ:\n{safeguard_warnings_str}\n"
        f"ИНСТРУКЦИЯ: Если инструмент или конкретная комбинация параметров (например, корпус + тип данных) "
        f"вернули ошибку или предупреждение защитного слоя, вы ОБЯЗАНЫ скорректировать вызов! "
        f"Запрещено повторно отправлять те же самые ошибочные параметры в planned_actions."
    )

    messages = [
        SystemMessage(content=system_prompt_ready),
        HumanMessage(content=context_message)
    ]

    try:

        response = llm.invoke(messages, config=config)
        clean_content = response.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean_content)

        planned_actions = data.get("planned_actions", [])
        reasoning = data.get("reasoning", "Обоснование не предоставлено")

        # --- ИНТЕГРАЦИЯ ТРЕЙСИНГА ---
        emit_trace(
            node="execution_planner",
            event_type="thought",
            content=reasoning,
            run_id=run_id
        )

        if planned_actions:
            emit_trace(
                node="execution_planner",
                event_type="action",
                content=planned_actions,
                run_id=run_id
            )

        print(f"[ExecPlanner] Запланировано действий: {len(planned_actions)}")
        for i, action in enumerate(planned_actions):
            print(f"  - {i + 1}. {action.get('action')} | {action.get('params')}")
        print(f"[ExecPlanner] Обоснование: {reasoning}")

        return {
            "planned_actions": planned_actions,
            "next_action": "",
            "action_params": {}
        }

    except Exception as e:
        print(f"[ExecPlanner] Ошибка парсинга или выбора инструмента: {e}")
        return {
            "planned_actions": [],
            "next_action": "",
            "action_params": {}
        }