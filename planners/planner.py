import json
from langchain_core.messages import SystemMessage, HumanMessage
from LLM.client import get_llm
from LLM.prompts import PLANNER_SYSTEM_PROMPT
from state.schema import ResearchState
from tools.registry import CORPUS_TYPE_ENUM_DESCRIPTION
from utils.trace import emit_trace

def planner_node(state: ResearchState):
    print("\n--- УЗЕЛ: Linguistic Task Planner (Запуск ИИ) ---")

    run_id = state.get("research_question", "default_run")
    llm = get_llm()

    system_prompt_ready = PLANNER_SYSTEM_PROMPT.format(
        corpus_registry=CORPUS_TYPE_ENUM_DESCRIPTION
    )

    feedback = state.get("aggregator_reasoning", "")

    existing_facts = state.get("facts", [])
    existing_facts_str = (
        "\n".join([f"- {fact}" for fact in existing_facts])
        if existing_facts
        else "Фактов пока нет (это первый запрос в сессии)"
    )
    existing_goal = state.get("goal", "Цель ещё не формулировалась")

    if feedback:
        user_content = (
            f"Исследовательский вопрос пользователя: {state['research_question']}\n\n"
            f"ОБРАТНАЯ СВЯЗЬ ОТ АНАЛИТИКА (ПРЕДЫДУЩИЙ ШАГ):\n{feedback}\n\n"
            f"ВАЖНО: Предыдущие действия не позволили полностью закрыть цель, и аналитик прямо запросил "
            f"пересмотр методологии. Скорректируй или расширь цель исследования и гипотезы так, чтобы они "
            f"учитывали недостающие филологические данные, указанные аналитиком выше.\n\n"
            f"РАНЕЕ ВЫБРАННЫЙ КОРПУС: {state.get('recommended_corpus', 'MAIN')} "
            f"(обоснование: {state.get('corpus_reasoning', 'не указано')}). "
            f"Пересмотри выбор корпуса, если обратная связь аналитика указывает, "
            f"что нужных данных нет именно из-за неверно выбранного корпуса.\n\n"
            f"Это шаг ПЕРЕПЛАНИРОВАНИЯ внутри активного исследования - mode ОБЯЗАН быть \"research\"."
        )
    else:
        user_content = (
            f"Исследовательский вопрос пользователя: {state['research_question']}\n\n"
            f"РАНЕЕ СФОРМУЛИРОВАННАЯ ЦЕЛЬ ИССЛЕДОВАНИЯ (если это не первый вопрос в сессии): {existing_goal}\n"
            f"УЖЕ НАКОПЛЕННЫЕ ФАКТЫ ИЗ ПРЕДЫДУЩИХ ШАГОВ:\n{existing_facts_str}\n\n"
            f"Реши, является ли этот вопрос НОВОЙ исследовательской задачей, требующей обращения к API НКРЯ "
            f"(mode=\"research\"), или это уточняющий вопрос по УЖЕ собранным фактам выше, на который можно "
            f"ответить без новых обращений к API (mode=\"chat\")."
        )

    messages = [
        SystemMessage(content=system_prompt_ready),
        HumanMessage(content=user_content)
    ]

    try:
        response = llm.invoke(messages)
        clean_content = response.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(clean_content)

        mode = data.get("mode", "research")
        mode_reasoning = data.get("mode_reasoning", "Обоснование режима не предоставлено")
        print(f"[Planner] Определён режим: {mode} ({mode_reasoning})")

        emit_trace(
            node="planner",
            event_type="thought",
            content={"mode": mode, "reasoning": mode_reasoning},
            run_id=run_id
        )

        if mode == "chat":
            return {"mode": "chat", "needs_replanning": False}

        recommended_corpus = data.get("recommended_corpus", "MAIN")
        corpus_reasoning = data.get("corpus_reasoning", "Обоснование не предоставлено")

        emit_trace(
            node="planner",
            event_type="decision",
            content={
                "goal": data.get("goal"),
                "recommended_corpus": recommended_corpus,
                "corpus_reasoning": corpus_reasoning
            },
            run_id=run_id
        )

        print(f"[Planner] Сформулирована цель: {data.get('goal')}")
        print(f"[Planner] Выдвинуто гипотез: {len(data.get('hypotheses', []))}")
        print(f"[Planner] Рекомендованный корпус: {recommended_corpus} ({corpus_reasoning})")

        return {
            "mode": "research",
            "goal": data.get("goal", "Цель не определена"),
            "hypotheses": data.get("hypotheses", []),
            "research_plan": data.get("research_plan", []),
            "recommended_corpus": recommended_corpus,
            "corpus_reasoning": corpus_reasoning,
            "needs_replanning": False
        }

    except Exception as e:
        print(f"[Planner] Ошибка парсинга или вызова модели: {e}")
        return {
            "mode": "research",
            "goal": "Ошибка планирования",
            "hypotheses": ["Не удалось сформировать гипотезы"],
            "research_plan": [],
            "recommended_corpus": "MAIN",
            "corpus_reasoning": "Ошибка планирования - используется корпус по умолчанию",
            "needs_replanning": False
        }