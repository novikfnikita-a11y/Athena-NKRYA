from state.schema import ResearchState
from tools.nkrja_client import NKRJAClient
from tools.registry import CAPABILITY_REGISTRY

client = NKRJAClient()





def api_orchestrator_node(state: ResearchState):
    """
            пакетный API Orchestrator.

        Принимает список запланированных действий (planned_actions),
    проходит по каждому из них, вызывает методы от nkrjaclient
            !результаты единым массивом evidence.
    """

    print("\n--- API ORCHESTRATOR (BATCH MODE) ---")

    # Теперь мы ожидаем массив действий.
    # Если его нет, фоллбек на старые поля для обратной совместимости.
    planned_actions = state.get("planned_actions", [])

    legacy_action = state.get("next_action")
    if not planned_actions and legacy_action and legacy_action != "finish":
        planned_actions = [{"action": legacy_action, "params": state.get("action_params", {})}]

    if not planned_actions:
        print("[Orchestrator] Запланированных действий нет.")
        return {
            "planned_actions": [],
            "next_action": "",
            "action_params": {}
        }

    new_evidence = []

    for item in planned_actions:
        action = item.get("action")
        params = item.get("params", {})

        if not action or action == "finish":
            continue

        # проверяем  существует ли capability
        if action not in CAPABILITY_REGISTRY:
            new_evidence.append({
                "source": "System",
                "action": action,
                "status": "error",
                "message": f"Инструмент '{action}' отсутствует в Capability Registry."
            })
            continue

        # пытаемся найти одноимённый метод в клиенте
        handler = getattr(client, action, None)

        if handler is None:
            new_evidence.append({
                "source": "System",
                "action": action,
                "status": "error",
                "message": f"Метод '{action}' ещё не реализован в NKRJAClient"
            })
            continue

        try:
            print(f"[Orchestrator] выполняется: {action} | параметры: {params}")

            if params:
                result = handler(**params)
            else:
                result = handler()

            new_evidence.append({
                "source": "NKRJA",
                "action": action,
                "params": params,
                "response": result
            })
            print(f"[Orchestrator] Успешно: {action}")

        except Exception as e:
            print(f"[Orchestrator] Ошибка в {action}: {e}")
            new_evidence.append({
                "source": "System",
                "action": action,
                "status": "error",
                "message": str(e)
            })

    print(f"[Orchestrator] пакетная обработка завершена. Собрано {len(new_evidence)} артефактов.")

    return {
        "evidence": new_evidence,
        "planned_actions": [],  #  очистка очереди
        "next_action": "",
        "action_params": {}
    }