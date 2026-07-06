# planners/planner.py
import json
from langchain_core.messages import SystemMessage, HumanMessage
from LLM.client import get_llm
from LLM.prompts import PLANNER_SYSTEM_PROMPT
from state.schema import ResearchState


def planner_node(state: ResearchState):
    print("\n--- УЗЕЛ: Linguistic Task Planner (Запуск DeepSeek) ---")

    llm = get_llm()

    # извлекаем обратную связь от агрегатора из предыдущих итераций
    # !!!!!помни про необходимость реализации human in the loop
    feedback = state.get("aggregator_reasoning", "")

    # формируем текст для модели если это первый шаг, отправляем просто вопрос
    # если это повторный цикл добавляем  требование скорректировать курс
    if feedback:
        user_content = (
            f"Исследовательский вопрос пользователя: {state['research_question']}\n\n"
            f"ОБРАТНАЯ СВЯЗЬ ОТ АНАЛИТИКА (ПРЕДЫДУЩИЙ ШАГ):\n{feedback}\n\n"
            f"ВАЖНО: Предыдущие действия не позволили полностью закрыть цель. "
            f"Скорректируй или расширь цель исследования и гипотезы так, чтобы они учитывали "
            f"недостающие филологические данные, указанные аналитиком выше."
        )
    else:
        user_content = f"Исследовательский вопрос пользователя: {state['research_question']}"

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_content)
    ]

    try:
        response = llm.invoke(messages)
        clean_content = response.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean_content)

        print(f"[Planner] Сформулирована цель: {data.get('goal')}")
        print(f"[Planner] Выдвинуто гипотез: {len(data.get('hypotheses', []))}")

        return {
            "goal": data.get("goal", "Цель не определена"),
            "hypotheses": data.get("hypotheses", [])
        }

    except Exception as e:
        print(f"[Planner] Ошибка парсинга или вызова модели: {e}")
        return {
            "goal": "Ошибка планирования",
            "hypotheses": ["Не удалось сформировать гипотезы"]
        }