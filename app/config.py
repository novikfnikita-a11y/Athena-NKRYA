import os
from dotenv import load_dotenv

load_dotenv()

NKRJA_API_KEY = os.getenv("NKRJA_API_KEY")
VSEGPT_API_KEY = os.getenv("VSEGPT_API_KEY")

if not NKRJA_API_KEY or not VSEGPT_API_KEY:
    raise ValueError("API ключи не найдены проверь файл .env")