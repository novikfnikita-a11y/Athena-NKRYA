import copy
import uuid

from state.schema import ResearchState
from tools.nkrja_client import NKRJAClient
from tools.registry import CAPABILITY_REGISTRY
from tools.evidence_compressor import compress_word_portrait_response
from whitelist_generated import RESULTTYPE_CORPUS_WHITELIST
from utils.trace import emit_trace

RESPONSE_COMPRESSORS = {
    "get_word_portrait": compress_word_portrait_response,
}

# НОВОЕ: Оборачиваем выполнение узла. Локальный emit_trace и облачный LangSmith теперь пишут параллельно.
def api_orchestrator_node(state: ResearchState):
    """
    Пакетный API Orchestrator с защитным слоем фильтрации несовместимых типов.
    Принимает список запланированных действий (planned_actions),
    проходит по каждому из них, вызывает методы от NKRJAClient.
    Результаты аккумулируются в массив evidence.
    """
    print("\n--- API ORCHESTRATOR (BATCH MODE) ---")
    run_id = state.get("run_id", "default-run")
    research_id = state.get("research_id", "legacy-research")
    branch_id = state.get("branch_id", "legacy-branch")
    batch_id = state.get("batch_id") or f"batch-{uuid.uuid4()}"
    iteration = state.get("iteration_count", 0)

    def trace(event_type, content, *, action_id=None):
        emit_trace(
            node="api_orchestrator",
            event_type=event_type,
            content=content,
            run_id=run_id,
            thread_id=state.get("thread_id"),
            turn_id=state.get("turn_id"),
            research_id=research_id,
            branch_id=branch_id,
            batch_id=batch_id,
            action_id=action_id,
            iteration=iteration,
        )

    planned_actions = state.get("planned_actions", [])

    if not planned_actions:
        print("[Orchestrator] Запланированных действий нет.")
        return {
            "planned_actions": [],
            "last_evidence_batch": [],
            "execution_status": "complete",
        }

    client = NKRJAClient()
    new_evidence = []
    completed_actions = []

    for item in planned_actions:
        action = item.get("action") or item.get("tool")
        action_id = item.get("action_id") or f"action-{uuid.uuid4()}"
        params = copy.deepcopy(item.get("params", {}))

        if action == "get_word_portrait" and "resultType" in params:
            corpus = params.get("corpus", "MAIN")
            original_types = params.get("resultType", [])

            valid_types = [rt for rt in original_types if corpus in RESULTTYPE_CORPUS_WHITELIST.get(rt, [])]
            removed_types = set(original_types) - set(valid_types)

            if removed_types:
                msg = f"Из запроса к корпусу '{corpus}' автоматически удалены неподдерживаемые бэкендом типы: {list(removed_types)}. Измените планирование вызова."
                # Ваша локальная система SQLite
                trace("observation", {"status": "warning", "message": msg}, action_id=action_id)
                print(f"[Orchestrator] Отфильтрованы неподдерживаемые типы для {corpus}: {removed_types}")

                new_evidence.append({
                    "research_id": research_id,
                    "run_id": run_id,
                    "branch_id": branch_id,
                    "batch_id": batch_id,
                    "action_id": action_id,
                    "evidence_id": f"evidence-{uuid.uuid4()}",
                    "source": "System_Safeguard",
                    "action": action,
                    "tool": action,
                    "params": copy.deepcopy(params),
                    "status": "warning",
                    "message": msg
                })

            params["resultType"] = valid_types

            if not valid_types:
                msg_err = f"Ошибка вызова: ни один из запрошенных типов {original_types} не применим к корпусу '{corpus}'."
                print(f"[Orchestrator] Вызов отменен: {msg_err}")

                # Ваша локальная система SQLite
                trace("observation", {"status": "error", "message": msg_err}, action_id=action_id)
                new_evidence.append({
                    "research_id": research_id,
                    "run_id": run_id,
                    "branch_id": branch_id,
                    "batch_id": batch_id,
                    "action_id": action_id,
                    "evidence_id": f"evidence-{uuid.uuid4()}",
                    "source": "System_Safeguard",
                    "action": action,
                    "tool": action,
                    "params": copy.deepcopy(params),
                    "status": "error",
                    "message": msg_err
                })
                completed_actions.append({
                    "research_id": research_id,
                    "run_id": run_id,
                    "branch_id": branch_id,
                    "batch_id": batch_id,
                    "action_id": action_id,
                    "tool": action,
                    "params": copy.deepcopy(params),
                    "status": "failed",
                    "iteration": iteration,
                    "message": msg_err,
                })
                continue

        if not action or action == "finish":
            continue

        if action not in CAPABILITY_REGISTRY:
            msg_missing = f"Инструмент '{action}' отсутствует в Capability Registry."
            print(f"[Orchestrator] Ошибка: {msg_missing}")

            # Ваша локальная система SQLite
            trace("observation", {"action": action, "status": "error", "message": msg_missing}, action_id=action_id)
            new_evidence.append({
                "research_id": research_id,
                "run_id": run_id,
                "branch_id": branch_id,
                "batch_id": batch_id,
                "action_id": action_id,
                "evidence_id": f"evidence-{uuid.uuid4()}",
                "source": "System",
                "action": action,
                "tool": action or "unknown_action",
                "params": copy.deepcopy(params),
                "status": "error",
                "message": msg_missing
            })
            completed_actions.append({
                "research_id": research_id, "run_id": run_id,
                "branch_id": branch_id, "batch_id": batch_id,
                "action_id": action_id, "tool": action or "unknown_action",
                "params": copy.deepcopy(params), "status": "failed",
                "iteration": iteration, "message": msg_missing,
            })
            continue

        handler = getattr(client, action, None)
        if handler is None:
            msg_unimplemented = f"Метод '{action}' ещё не реализован в NKRJAClient"
            print(f"[Orchestrator] Ошибка: {msg_unimplemented}")

            # Ваша локальная система SQLite
            trace("observation", {"action": action, "status": "error", "message": msg_unimplemented}, action_id=action_id)
            new_evidence.append({
                "research_id": research_id,
                "run_id": run_id,
                "branch_id": branch_id,
                "batch_id": batch_id,
                "action_id": action_id,
                "evidence_id": f"evidence-{uuid.uuid4()}",
                "source": "System",
                "action": action,
                "tool": action,
                "params": copy.deepcopy(params),
                "status": "error",
                "message": msg_unimplemented
            })
            completed_actions.append({
                "research_id": research_id, "run_id": run_id,
                "branch_id": branch_id, "batch_id": batch_id,
                "action_id": action_id, "tool": action,
                "params": copy.deepcopy(params), "status": "failed",
                "iteration": iteration, "message": msg_unimplemented,
            })
            continue

        try:
            print(f"[Orchestrator] выполняется: {action} | параметры: {params}")

            if params:
                result = handler(**params)
            else:
                result = handler()

            raw_size = len(str(result))
            compressor = RESPONSE_COMPRESSORS.get(action)
            if compressor is not None:
                try:
                    result = compressor(result)
                except Exception as compress_err:
                    print(f"[Orchestrator] ВНИМАНИЕ: компрессор для '{action}' упал ({compress_err}), кладём сырой ответ как есть.")
                    # Ваша локальная система SQLite
                    trace("observation", {"action": action, "status": "warning", "message": f"Компрессор упал: {compress_err}"}, action_id=action_id)

            compressed_size = len(str(result))
            print(f"[Orchestrator] Успешно: {action} | размер ответа: {raw_size} -> {compressed_size} байт")

            # Ваша локальная система SQLite
            trace("observation", {"action": action, "status": "success", "response_preview": str(result)[:500] + "..."}, action_id=action_id)

            new_evidence.append({
                "research_id": research_id,
                "run_id": run_id,
                "branch_id": branch_id,
                "batch_id": batch_id,
                "action_id": action_id,
                "evidence_id": f"evidence-{uuid.uuid4()}",
                "source": "NKRJA",
                "action": action,
                "tool": action,
                "params": copy.deepcopy(params),
                "response": result,
                "payload": result,
                "status": "success",
            })
            completed_actions.append({
                "research_id": research_id, "run_id": run_id,
                "branch_id": branch_id, "batch_id": batch_id,
                "action_id": action_id, "tool": action,
                "params": copy.deepcopy(params), "status": "succeeded",
                "iteration": iteration,
            })

        except Exception as e:
            print(f"[Orchestrator] Ошибка в {action}: {e}")

            # Ваша локальная система SQLite
            trace("observation", {"action": action, "status": "error", "message": str(e)}, action_id=action_id)

            new_evidence.append({
                "research_id": research_id,
                "run_id": run_id,
                "branch_id": branch_id,
                "batch_id": batch_id,
                "action_id": action_id,
                "evidence_id": f"evidence-{uuid.uuid4()}",
                "source": "System",
                "action": action,
                "tool": action or "unknown_action",
                "params": copy.deepcopy(params),
                "status": "error",
                "message": str(e)
            })
            completed_actions.append({
                "research_id": research_id, "run_id": run_id,
                "branch_id": branch_id, "batch_id": batch_id,
                "action_id": action_id, "tool": action or "unknown_action",
                "params": copy.deepcopy(params), "status": "failed",
                "iteration": iteration, "message": str(e),
            })

    print(f"[Orchestrator] пакетная обработка завершена. Собрано {len(new_evidence)} артефактов.")

    return {
        "evidence": new_evidence,
        "last_evidence_batch": new_evidence,
        "completed_actions": completed_actions,
        "planned_actions": [],
        "batch_id": batch_id,
        "execution_status": "complete",
    }
