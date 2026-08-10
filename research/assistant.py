from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from utils.logger import bind_context
from utils.trace import emit_trace
from state.schema import ResearchState

from LLM.prompts import ASSISTANT_SYSTEM_PROMPT, ASSISTANT_USER_PROMPT_TEMPLATE
from LLM.client import get_llm

def assistant_node(state: ResearchState, config : RunnableConfig) -> dict:
    """
            Узел ассистента. Анализирует собранные факты и пишет финальный ответ.
            Использует клиент LangChain для корректной маршрутизации ролей.
    """
    log = bind_context(
        thread_id=state.get("thread_id"),
        turn_id=state.get("turn_id"),
        research_id=state.get("research_id"),
        run_id=state.get("run_id"),
        branch_id=state.get("branch_id"),
        iteration=state.get("iteration_count", 0),
    )

    def trace_completion(status: str, termination_reason: str) -> None:
        run_id = state.get("run_id")
        if not run_id:
            return
        emit_trace(
            node="assistant",
            event_type="completion",
            content={"status": status, "termination_reason": termination_reason},
            run_id=run_id,
            thread_id=state.get("thread_id"),
            turn_id=state.get("turn_id"),
            research_id=state.get("research_id"),
            branch_id=state.get("branch_id"),
            batch_id=state.get("batch_id"),
            iteration=state.get("iteration_count", 0),
            public=True,
        )

    log.info("ASSISTANT NODE: финальная генерация ответа")
    question = state.get("research_question", "неизвестный вопрос")
    if state.get("research_status") == "error" and state.get("error"):
        safe_message = state["error"].get("message", "Исследование завершилось с ошибкой.")
        log.warning("Исследование завершено контролируемой ошибкой: {}", safe_message)
        trace_completion("error", str(state.get("termination_reason") or "error"))
        return {
            "final_response": safe_message,
            "research_status": "error",
            "termination_reason": state.get("termination_reason") or "error",
        }

    facts = state.get("facts", [])
    if state.get("mode") == "chat":
        facts = state.get("context_facts", [])

    # 1. подготовка контекста из фактов
    if not facts:
        log.warning("Факты для генерации ответа отсутствуют.")
        facts_text = "исследовательская система не смогла извлечь релевантные факты по этому запросу."
    else:
        # массив фактов список
        facts_text = "\n".join([f"- {fact}" for fact in facts])
        log.info("В LLM отправляется {} фактов для формирования ответа.", len(facts))

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
        log.debug("Отправка сформированных сообщений в настроенную LLM.")

        # invoke в LС принимает список сообщений и возвращает AIMessage
        response = llm.invoke(messages,config=config)

        # извлекаем чистое текстовое содержимое ответа
        final_answer = response.content
        log.info("Итоговый лингвистический ответ успешно сгенерирован.")

    except Exception as e:
        log.error("Ошибка при обращении к LLM в узле Assistant: {}", e)
        trace_completion("error", "error")
        return {
            "final_response": "Не удалось сформировать итоговый ответ.",
            "research_status": "error",
            "termination_reason": "error",
            "error": {
                "kind": "model",
                "message": "Не удалось сформировать итоговый ответ.",
                "node": "assistant",
                "retryable": True,
            },
        }

    termination_reason = str(state.get("termination_reason") or "plan_complete")
    trace_completion("complete", termination_reason)
    return {
        "final_response": final_answer,
        "research_status": "complete",
        "termination_reason": termination_reason,
    }
