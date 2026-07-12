import os
from dotenv import load_dotenv

load_dotenv()

NKRJA_API_KEY = os.getenv("NKRJA_API_KEY")
VSEGPT_API_KEY = os.getenv("VSEGPT_API_KEY")


if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    if not os.getenv("LANGCHAIN_API_KEY"):
        raise ValueError("Включен LANGCHAIN_TRACING_V2, но LANGCHAIN_API_KEY не найден в .env")


if not NKRJA_API_KEY or not VSEGPT_API_KEY:
    raise ValueError("API ключи не найдены проверь файл .env")