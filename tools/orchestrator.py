from state.schema import ResearchState
from tools.nkrja_client import NKRJAClient
from tools.registry import CAPABILITY_REGISTRY
from whitelist_generated import RESULTTYPE_CORPUS_WHITELIST
from utils.trace import emit_trace

client = NKRJAClient()


def api_orchestrator_node(state: ResearchState):
    """
    Пакетный API Orchestrator с защитным слоем фильтрации несовместимых типов.
    Принимает список запланированных действий (planned_actions),
    проходит по каждому из них, вызывает методы от NKRJAClient.
    Результаты аккумулируются в массив evidence.
    """
    print("\n--- API ORCHESTRATOR (BATCH MODE) ---")
    run_id = state.get("research_question", "default_run")

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

        if action == "get_word_portrait" and "resultType" in params:
            corpus = params.get("corpus", "MAIN")
            original_types = params.get("resultType", [])

            #оставляем только те типы, для которых текущий корпус находится в белом списке
            valid_types = [rt for rt in original_types if corpus in RESULTTYPE_CORPUS_WHITELIST.get(rt, [])]
            removed_types = set(original_types) - set(valid_types)

            if removed_types:
                msg = f"Из запроса к корпусу '{corpus}' автоматически удалены неподдерживаемые бэкендом типы: {list(removed_types)}. Измените планирование вызова."
                emit_trace(
                    node="api_orchestrator",
                    event_type="observation",
                    content={"status": "warning", "message": msg},
                    run_id=run_id
                )
                print(f"[Orchestrator] Отфильтрованы неподдерживаемые типы для {corpus}: {removed_types}")

                new_evidence.append({
                    "source": "System_Safeguard",
                    "action": action,
                    "status": "warning",
                    "message": msg
                })

            params["resultType"] = valid_types

            if not valid_types:
                msg_err = f"Ошибка вызова: ни один из запрошенных типов {original_types} не применим к корпусу '{corpus}'."
                print(f"[Orchestrator] Вызов отменен: {msg_err}")

                emit_trace(
                    node="api_orchestrator",
                    event_type="observation",
                    content={"status": "error", "message": msg_err},
                    run_id=run_id
                )
                new_evidence.append({
                    "source": "System_Safeguard",
                    "action": action,
                    "status": "error",
                    "message": msg_err
                })
                continue

        if not action or action == "finish":
            continue

        if action not in CAPABILITY_REGISTRY:
            msg_missing = f"Инструмент '{action}' отсутствует в Capability Registry."
            print(f"[Orchestrator] Ошибка: {msg_missing}")

            emit_trace(
                node="api_orchestrator",
                event_type="observation",
                content={"action": action, "status": "error", "message": msg_missing},
                run_id=run_id
            )
            new_evidence.append({
                "source": "System",
                "action": action,
                "status": "error",
                "message": msg_missing
            })
            continue

        handler = getattr(client, action, None)
        if handler is None:
            msg_unimplemented = f"Метод '{action}' ещё не реализован в NKRJAClient"
            print(f"[Orchestrator] Ошибка: {msg_unimplemented}")

            emit_trace(
                node="api_orchestrator",
                event_type="observation",
                content={"action": action, "status": "error", "message": msg_unimplemented},
                run_id=run_id
            )
            new_evidence.append({
                "source": "System",
                "action": action,
                "status": "error",
                "message": msg_unimplemented
            })
            continue

        try:
            print(f"[Orchestrator] выполняется: {action} | параметры: {params}")

            if params:
                result = handler(**params)
            else:
                result = handler()

            emit_trace(
                node="api_orchestrator",
                event_type="observation",
                content={"action": action, "status": "success", "response_preview": str(result)[:500] + "..."},
                run_id=run_id
            )

            new_evidence.append({
                "source": "NKRJA",
                "action": action,
                "params": params,
                "response": result
            })
            print(f"[Orchestrator] Успешно: {action}")

        except Exception as e:
            print(f"[Orchestrator] Ошибка в {action}: {e}")

            emit_trace(
                node="api_orchestrator",
                event_type="observation",
                content={"action": action, "status": "error", "message": str(e)},
                run_id=run_id
            )

            new_evidence.append({
                "source": "System",
                "action": action,
                "status": "error",
                "message": str(e)
            })

    print(f"[Orchestrator] пакетная обработка завершена. Собрано {len(new_evidence)} артефактов.")

    return {
        "evidence": state.get("evidence", []) + new_evidence,
        "next_action": "",
        "action_params": {}
    }