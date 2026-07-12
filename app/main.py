import os
from typing import Any

os.environ["USE_LOCAL_CHECKPOINTER"] = "1"
os.environ["LANGCHAIN_PROJECT"] = "nkrja-multiagent-agent"

from app.graph import app

if __name__ == "__main__":
    #Настройки сессии для сохранения контекста вопросов в рамках одного thread_id
    config = {
        "configurable": {"thread_id": "cli-session-1"},
        "metadata": {
            "environment": "development",
            "project": "nkrja-multiagent-agent",
            "interface": "cli"
        },
        "tags": ["Interactive_Research_Session"]
    }

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