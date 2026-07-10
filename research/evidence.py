import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from LLM.client import get_llm
from LLM.prompts import EVIDENCE_AGGREGATOR_SYSTEM_PROMPT
from state.schema import ResearchState
from tools.registry import get_registry_description


def evidence_aggregator_node(state: ResearchState):
    print("\n--- УЗЕЛ: EVIDENCE AGGREGATOR ---")

    if not state.get("evidence"):
        print("[Aggregator] Нет новых сырых данных для анализа.")
        return {
            "is_goal_reached": False,
            "needs_replanning": False,
            "aggregator_reasoning": "Сырые данные от API не поступили.",
            "planned_actions": []  # Гарантированная очистка на случай сбоя
        }

    # ФИКС ОШИБКИ 1: Забираем размер батча из planned_actions.
    # Очередь еще НЕ очищена в оркестраторе, поэтому мы точно знаем, сколько инструментов выполнилось.
    planned_actions = state.get("planned_actions", [])
    batch_size = len(planned_actions)

    if batch_size == 0:
        print("[Aggregator] Внимание: Очередь planned_actions пуста. Фоллбек на последний артефакт.")
        current_batch_evidence = [state["evidence"][-1]]
    else:
        current_batch_evidence = state["evidence"][-batch_size:]

    print(f"[Aggregator] Анализируем пакет из {len(current_batch_evidence)} ответов API.")

    # Красиво структурируем ответы по разным ключам
    formatted_blocks = []
    for idx, ev in enumerate(current_batch_evidence):
        action = ev.get("action", "unknown_action")
        params = ev.get("params", {})
        response = ev.get("response", {})

        block = f"--- ДЕЙСТВИЕ {idx + 1}: {action} | ПАРАМЕТРЫ: {params} ---\n"

        if ev.get("status") == "error":
            block += f"ОШИБКА ВЫПОЛНЕНИЯ: {ev.get('message')}\n"
        else:
            block += json.dumps(response, ensure_ascii=False, indent=2)

        formatted_blocks.append(block)

    # ФИКС ОШИБКИ 2: Безопасное итеративное усечение строго ПО ГРАНИЦАМ БЛОКОВ
    raw_data_str = ""
    is_truncated = False

    for block in formatted_blocks:
        if len(raw_data_str) + len(block) + 2 > 90000:
            is_truncated = True
            break
        if raw_data_str:
            raw_data_str += "\n\n" + block
        else:
            raw_data_str = block

    if is_truncated:
        print("[Aggregator] ВНИМАНИЕ: Данные превысили безопасный лимит и были усечены по границе блоков.")
        raw_data_str += "\n\n[ДАННЫЕ ПАКЕТА БЫЛИ ИЗБЫТОЧНЫ И ЧАСТИЧНО ОБРЕЗАНЫ ПО ГРАНИЦЕ БЛОКОВ ДЛЯ ОПТИМИЗАЦИИ ТОКЕНОВ]"

    # Контекст для аналитика
    context_message = (
        f"Вопрос пользователя: {state['research_question']}\n"
        f"Текущая цель исследования: {state['goal']}\n"
        f"Проверяемые гипотезы: {state['hypotheses']}\n\n"
        f"Новые сырые данные от API (Логи работы инструментов):\n{raw_data_str}"
    )

    system_prompt_ready = EVIDENCE_AGGREGATOR_SYSTEM_PROMPT.format(
        registry_description=get_registry_description()
    )

    messages = [
        SystemMessage(content=system_prompt_ready),
        HumanMessage(content=context_message)
    ]

    llm = get_llm()

    # Инициализируем дефолтные значения переменных для единого return
    new_facts = []
    is_goal_reached = False
    needs_replanning = False
    reasoning = "Обоснование не сформировано"

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
        needs_replanning = data.get("needs_replanning", False)
        reasoning = data.get("reasoning", "Нет обоснования")

        print(f"[Aggregator] Извлечено новых фактов: {len(new_facts)}")
        print(f"[Aggregator] Цель достигнута: {is_goal_reached} ({reasoning})")
        if needs_replanning:
            print("[Aggregator] Запрошено ПОЛНОЕ переосмысление методологии (needs_replanning=True)")

    except json.JSONDecodeError as e:
        print(f"[Aggregator] ОШИБКА ПАРСИНГА JSON от модели: {e}")
        new_facts = []
        is_goal_reached = False
        needs_replanning = False
        reasoning = f"Ошибка разбора JSON-ответа от аналитика: {str(e)}"

    except Exception as e:
        print(f"[Aggregator] Непредвиденная ошибка узла: {e}")
        new_facts = []
        is_goal_reached = False
        needs_replanning = False
        reasoning = f"Сбой внутренней логики агрегатора: {str(e)}"

    current_iter = state.get("iteration_count", 0)

    # Единая точка возврата. Всегда выполняется корректно.
    return {
        "facts": new_facts,
        "is_goal_reached": is_goal_reached,
        "needs_replanning": needs_replanning,
        "aggregator_reasoning": reasoning,
        "iteration_count": current_iter + 1,
        "planned_actions": [],  # ОЧИСТКА ОЧЕРЕДИ
        "next_action": "",
        "action_params": {}
    }