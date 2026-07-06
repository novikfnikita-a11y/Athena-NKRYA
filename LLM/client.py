# LLM/client.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()
VSEGPT_API_KEY = os.getenv("VSEGPT_API_KEY")


def get_llm():

    if not VSEGPT_API_KEY:
        raise ValueError("API ключ VSEGPT_API_KEY не найден в .env!")

    base_model = "deepseek/deepseek-chat"


        #  :nojsonencode отключает принудительное экранирование кириллицы.
            #     в логах и ответах будет чистый читаемый русский текст,
    #    что защитит регулярные выражения в узле evidence aggregator от сбоев парсинга.
    #  :x-title --передает имя нашего приложения для красивого дашборда статистики
    model_with_modifiers = f"{base_model}:nojsonencode:x-title=NKRJA_Research_Agent"

    return ChatOpenAI(
        model=model_with_modifiers,
        api_key=SecretStr(VSEGPT_API_KEY),
        base_url="https://api.vsegpt.ru/v1",
        max_tokens=2048,
        temperature=0.2
    )