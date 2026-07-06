from langchain_core.messages import SystemMessage, HumanMessage
from utils.logger import logger
from state.schema import ResearchState

from LLM.prompts import ASSISTANT_SYSTEM_PROMPT, ASSISTANT_USER_PROMPT_TEMPLATE
from LLM.client import get_llm
def assistant_node(state: ResearchState) -> dict:
    """
            Узел ассистента. Анализирует собранные факты и пишет финальный ответ.
            Использует клиент LangChain для корректной маршрутизации ролей.
    """
    logger.info(" ASSISTANT NODE ФИНАЛЬНАЯ ГЕНЕРАЦИЯ ОТВЕТА")
    question = state.get("research_question", "неизвестный вопрос")
    facts = state.get("facts", [])

    # 1. подготовка контекста из фактов
    if not facts:
        logger.warning("факты для генерации ответа отсутствуют.")
        facts_text = "исследовательская система не смогла извлечь релевантные факты по этому запросу."
    else:
        # массив фактов список
        facts_text = "\n".join([f"- {fact}" for fact in facts])
        logger.info(f"В LLM отправляется {len(facts)} фактов для формирования ответа.")

    # 2. пользовательский запрос . формирование через шаблон
    user_prompt_content = ASSISTANT_USER_PROMPT_TEMPLATE.format(
        question=question,
        facts=facts_text
    )

    # 3. упаковка промптов в структуру langchain сообщений
    messages = [
        SystemMessage(content=ASSISTANT_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt_content)
    ]

    # 4. вызов модели и генерация
    try:
        llm = get_llm()
        logger.debug("Отправка сформированных сообщений в DeepSeek (через VseGPT)...")

        # invoke в LС принимает список сообщений и возвращает AIMessage
        response = llm.invoke(messages)

        # извлекаем чистое текстовое содержимое ответа
        final_answer = response.content
        logger.info("итоговый лингвистический ответ успешно сгенерирован")

    except Exception as e:
        # При критическом сбое логируем полный traceback для отладки
        logger.error(f"ошибка при обращении к LLM в узле Assistant: {e}", exc_info=True)
        final_answer = "ошибка при попытке генерации финального ответа"


    return {
        "final_response": final_answer
    }