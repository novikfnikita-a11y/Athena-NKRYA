"""Asynchronous final answer generation from explicitly permitted facts."""

import asyncio
import json
from typing import Any, Mapping

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from LLM.client import get_llm
from LLM.prompts import ASSISTANT_SYSTEM_PROMPT, ASSISTANT_USER_PROMPT_TEMPLATE
from state.schema import ResearchState
from utils.logger import bind_context
from utils.trace import emit_trace


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value if isinstance(value, Mapping) else None


def format_facts(records: list[Any]) -> str:
    """Render facts without losing their evidence identifiers."""

    lines: list[str] = []
    for record in records:
        item = _as_mapping(record)
        if item is None:
            # Transitional archived facts may predate the typed model.  They are
            # marked as such instead of receiving invented provenance.
            lines.append(
                f"- {record} [evidence_id: отсутствует в устаревшем архиве]"
            )
            continue
        evidence_id = item.get("evidence_id")
        fact_id = item.get("fact_id")
        corpus = item.get("corpus")
        metric = item.get("metric")
        value = item.get("value")
        unit = item.get("unit")
        lemma = item.get("lemma")
        lines.append(
            "- "
            + json.dumps(
                {
                    "fact_id": fact_id,
                    "evidence_id": evidence_id,
                    "corpus": corpus,
                    "lemma": lemma,
                    "metric": metric,
                    "value": value,
                    "unit": unit,
                },
                ensure_ascii=False,
                default=str,
            )
        )
    return "\n".join(lines)


def _trace_completion(
    state: ResearchState,
    status: str,
    termination_reason: str,
) -> None:
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


async def assistant_node_async(
    state: ResearchState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    log = bind_context(
        thread_id=state.get("thread_id"),
        turn_id=state.get("turn_id"),
        research_id=state.get("research_id"),
        run_id=state.get("run_id"),
        branch_id=state.get("branch_id"),
        iteration=state.get("iteration_count", 0),
    )
    if state.get("research_status") == "error" and state.get("error"):
        safe_message = state["error"].get(
            "message", "Исследование завершилось с ошибкой."
        )
        _trace_completion(state, "error", "error")
        return {
            "final_response": safe_message,
            "research_status": "error",
            "termination_reason": "error",
        }

    mode = str(state.get("mode") or "research")
    facts = (
        list(state.get("context_facts", []))
        if mode == "chat"
        else list(state.get("facts", []))
    )
    context_snapshot: Mapping[str, Any] = {}
    if mode == "chat" and state.get("selected_context_research_id"):
        context_snapshot = state.get("research_archive", {}).get(
            state["selected_context_research_id"], {}
        )
    termination_reason = str(
        state.get("termination_reason")
        or context_snapshot.get("termination_reason")
        or "plan_complete"
    )
    is_partial = (
        str(state.get("research_status")) == "partial"
        or str(context_snapshot.get("status")) == "partial"
        or termination_reason
        in {"budget_exhausted", "no_progress", "no_actions"}
    )
    if not facts:
        answer = (
            "Недостаточно проверяемых фактов для ответа. "
            "Исследование завершено без доказательств, на которые можно безопасно сослаться."
        )
        _trace_completion(state, "partial", termination_reason)
        return {
            "final_response": answer,
            "research_status": "partial",
            "termination_reason": termination_reason,
        }

    prompt = ASSISTANT_USER_PROMPT_TEMPLATE.format(
        question=state.get("research_question", ""),
        mode=mode,
        completion_status="частичный" if is_partial else "полный",
        termination_reason=termination_reason,
        facts=format_facts(facts),
    )
    try:
        response = await get_llm().ainvoke(
            [
                SystemMessage(content=ASSISTANT_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ],
            config=config,
        )
        final_answer = str(getattr(response, "content", response)).strip()
        if not final_answer:
            raise ValueError("assistant returned empty content")
    except Exception:
        log.error("Модель не сформировала итоговый ответ.")
        _trace_completion(state, "error", "error")
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

    final_status = "partial" if is_partial else "complete"
    _trace_completion(state, final_status, termination_reason)
    return {
        "final_response": final_answer,
        "research_status": final_status,
        "termination_reason": termination_reason,
    }


def assistant_node(
    state: ResearchState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Synchronous Stage-3 adapter; Stage 4 wires the async node directly."""

    return asyncio.run(assistant_node_async(state, config))


__all__ = ["assistant_node", "assistant_node_async", "format_facts"]
