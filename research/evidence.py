import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from LLM.client import get_llm
from LLM.prompts import EVIDENCE_AGGREGATOR_SYSTEM_PROMPT
from state.schema import ResearchState


def evidence_aggregator_node(state: ResearchState):
    print("\nУЗЕЛ: evidence aggregator")

    if not state.get("evidence"):
        print("[Aggregator] Нет новых сырых данных для анализа.")
        return {
            "is_goal_reached": False,
            "aggregator_reasoning": "сырые данные не поступили."
        }

    latest_evidence = state["evidence"][-1]
    llm = get_llm()

    # чанкирование
    raw_data_str = json.dumps(latest_evidence, ensure_ascii=False)
    if len(raw_data_str) > 3000:
        raw_data_str = raw_data_str[:3000] + "[ДАННЫЕ ОБРЕЗАНЫ]"

    context_message = (
        f"Вопрос: {state['research_question']}\n"
        f"Цель: {state['goal']}\n"
        f"Гипотезы: {state['hypotheses']}\n\n"
        f"Новые сырые данные от API:\n{raw_data_str}"
    )

    messages = [
        SystemMessage(content=EVIDENCE_AGGREGATOR_SYSTEM_PROMPT),
        HumanMessage(content=context_message)
    ]

    response = None

    try:
        response = llm.invoke(messages)
        content = response.content.strip()

        json_match = re.search(r'\{.*}', content, re.DOTALL)
        if json_match:
            clean_content = json_match.group(0)
        else:
            clean_content = content

        data = json.loads(clean_content)

        new_facts = data.get("new_facts", [])
        is_goal_reached = data.get("is_goal_reached", False)
        reasoning = data.get("reasoning", "Нет обоснования")

        print(f"[Aggregator] Извлечено новых фактов: {len(new_facts)}")
        print(f"[Aggregator] Цель достигнута: {is_goal_reached} ({reasoning})")

        return {
            "facts": new_facts,
            "is_goal_reached": is_goal_reached,
            "aggregator_reasoning": reasoning
        }

    except json.JSONDecodeError as e:
        print(f"[Aggregator] ОШИБКА ПАРСИНГА JSON от модели: {e}")
        if response:
            print(f"[Aggregator] сырой ответ модели был:\n{response.content}")
        return {
            "facts": ["Не удалось извлечь факты из-за ошибки генерации JSON."],
            "is_goal_reached": True,
            "aggregator_reasoning": "критическая ошибка парсинга ответа от LLM."
        }
    except Exception as e:
        print(f"[Aggregator] непредвиденная ошибка: {e}")
        return {
            "facts": [f" ошибка: {str(e)}"],
            "is_goal_reached": True,
            "aggregator_reasoning": f" сбой: {str(e)}"
        }