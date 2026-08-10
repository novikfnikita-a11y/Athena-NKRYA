"""Interactive CLI entry point with explicit resource lifecycle."""

from __future__ import annotations

import os
from typing import Any


# The graph must know whether the CLI needs an in-process conversation
# checkpointer before app.graph is imported.
os.environ["USE_LOCAL_CHECKPOINTER"] = "1"

from app.graph import app
from utils.logger import init_logger
from utils.trace import close_trace, configure_trace, flush_traces


def main() -> None:
    init_logger()
    configure_trace(os.environ.get("ATHENA_TRACE_DB", "research_traces.db"))
    config = {
        "configurable": {"thread_id": "cli-session-1"},
        "metadata": {
            "environment": "development",
            "project": "nkrja-multiagent-agent",
            "interface": "cli",
        },
        "tags": ["Interactive_Research_Session"],
    }

    print("Исследовательский агент НКРЯ. Введите вопрос (или 'exit' для выхода).\n")
    try:
        while True:
            user_question = input(
                "Ваш запрос в Национальный Корпус Русского Языка: "
            ).strip()
            if user_question.lower() in ("exit", "quit", "выход"):
                print("Сессия завершена.")
                break
            if not user_question:
                continue

            invoke_input: dict[str, Any] = {"research_question": user_question}
            final_output = app.invoke(invoke_input, config=config)

            print("\n-ОТВЕТ АГЕНТА ")
            print(final_output.get("final_response", "Ответ не сформирован"))
            print("--------------------\n")
    finally:
        flush_traces(timeout=5.0)
        close_trace(timeout=5.0)


if __name__ == "__main__":
    main()
