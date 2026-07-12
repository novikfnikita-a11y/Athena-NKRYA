# check_logs.py
import sqlite3
import json

conn = sqlite3.connect("research_traces.db")
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

try:
    cursor.execute("SELECT * FROM traces ORDER BY ts ASC")
    rows = cursor.fetchall()
    print(f" Всего найдено записей в БД: {len(rows)}\n" + "=" * 50)

    for row in rows:
        print(f"[{row['node'].upper()}] Тип: {row['type']} | Время: {row['ts']}")
        try:
            content_json = json.loads(row['content'])
            print(json.dumps(content_json, ensure_ascii=False, indent=2))
        except:
            print(f"Content: {row['content']}")
        print("-" * 50)
except Exception as e:
    print(f"Ошибка при чтении логов: {e}")
finally:
    conn.close()