import json
import sqlite3
import time
import threading
import queue
from typing import Literal, Any

DB_PATH = "research_traces.db"

_trace_queue = queue.Queue(maxsize=1000)


def _db_worker():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                branch_id TEXT,
                iteration INTEGER,
                node TEXT,
                type TEXT,
                content TEXT,
                ts REAL
            )
        """)
        conn.commit()

        while True:
            try:

                event = _trace_queue.get()
                if event is None:
                    break

                conn.execute(
                    "INSERT INTO traces (run_id, branch_id, iteration, node, type, content, ts) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (event['run_id'], event.get('branch_id'), event.get('iteration', 0), event['node'], event['type'],
                     event['content'], event['ts'])
                )
                conn.commit()
                _trace_queue.task_done()
            except Exception as e:
                print(f"[Trace DB Worker Error] {e}")


threading.Thread(target=_db_worker, daemon=True).start()


def emit_trace(
        node: str,
        event_type: Literal["thought", "action", "observation", "decision", "final_answer"],
        content: Any,
        *,
        run_id: str,
        iteration: int = 0,
        branch_id: str = None
):

    content_str = json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)

    event = {
        "run_id": run_id,
        "branch_id": branch_id,
        "iteration": iteration,
        "node": node,
        "type": event_type,
        "content": content_str,
        "ts": time.time()
    }

    try:
        _trace_queue.put_nowait(event)
    except queue.Full:
        print("[Trace Warning] Очередь логов переполнена, событие пропущено.")