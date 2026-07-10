from state.schema import ResearchState
from tools.nkrja_client import NKRJAClient
from tools.registry import CAPABILITY_REGISTRY
from whitelist_generated import RESULTTYPE_CORPUS_WHITELIST

client = NKRJAClient()

def api_orchestrator_node(state: ResearchState):
    """
    Пакетный API Orchestrator с защитным слоем фильтрации несовместимых типов.
    Принимает список запланированных действий (planned_actions),
    проходит по каждому из них, вызывает методы от NKRJAClient.
    Результаты аккумулируются в массив evidence.
    """
    print("\n--- API ORCHESTRATOR (BATCH MODE) ---")

    planned_actions = state.get("planned_actions", [])

    # Фоллбек на старые поля для обратной совместимости
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

        # --- ЗАЩИТНЫЙ СЛОЙ ДЛЯ СНЯТИЯ ГАЛЛЮЦИНАЦИЙ РЕЗУЛЬТАТОВ ---
        if action == "get_word_portrait" and "resultType" in params:
            corpus = params.get("corpus", "MAIN")
            original_types = params.get("resultType", [])

            # Оставляем только те типы, для которых текущий корпус находится в белом списке
            valid_types = [rt for rt in original_types if corpus in RESULTTYPE_CORPUS_WHITELIST.get(rt, [])]
            removed_types = set(original_types) - set(valid_types)

            if removed_types:
                print(f"[Orchestrator] Отфильтрованы неподдерживаемые типы для {corpus}: {removed_types}")
                new_evidence.append({
                    "source": "System_Safeguard",
                    "action": action,
                    "status": "warning",
                    "message": f"Из запроса к корпусу '{corpus}' автоматически удалены неподдерживаемые бэкендом типы: {list(removed_types)}. Измените планирование вызова."
                })

            params["resultType"] = valid_types

            # Если после фильтрации вообще ничего не осталось (модель полностью ошиблась с набором)
            if not valid_types:
                print(f"[Orchestrator] Вызов отменен: ни один из типов не валиден для корпуса {corpus}")
                new_evidence.append({
                    "source": "System_Safeguard",
                    "action": action,
                    "status": "error",
                    "message": f"Ошибка вызова: ни один из запрошенных типов {original_types} не применим к корпусу '{corpus}'."
                })
                continue  # Переходим к следующему инструменту в батче, не дергая API вхолостую

        if not action or action == "finish":
            continue

        # Проверяем, существует ли capability в реестре
        if action not in CAPABILITY_REGISTRY:
            new_evidence.append({
                "source": "System",
                "action": action,
                "status": "error",
                "message": f"Инструмент '{action}' отсутствует в Capability Registry."
            })
            continue

        # Пытаемся найти одноимённый метод в клиенте
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

    # Возвращаем НАКОПЛЕННЫЙ массив (копия старого + новые артефакты)
    return {
        "evidence": state.get("evidence", []) + new_evidence,
        "next_action": "",
        "action_params": {}
    }