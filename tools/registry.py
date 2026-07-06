# tools/registry.py это жесткий справочник доступных инструментов нашего MVP
CAPABILITY_REGISTRY = {
    "get_word_portrait": {
        "description": "Возвращает портрет слова из Основного корпуса НКРЯ (возможные части речи, семантические характеристики).",
        "requires_params": ["lemma"]
    },
    "get_corpus_stats": {
        "description": "Возвращает статистику по выбранному корпусу (Можно получить число словоупотреблений, предложений и текстов).",
        "requires_params": ["corpus"]
    },
    "get_sketch_difference": {
        "description": "Возвращает результат сравнения скетчей двух слов из портрета.",
        "requires_params": ["lemma_1", "lemma_2", "corpus", "pos"]

    },
    "get_simple_concordance": {
        "description": "Безопасная обертка для LLM. Выполняет простой лексико-грамматический поиск конкорданса по одной лемме.",
        "requires_params": ["lemma", "corpus"]
    },
    "get_corpus_config": {
        "description": "Возвращает конфигурацию выбранного корпуса.",
        "requires_params": ["corpus"]
    },
    "get_corpus_attributes": {
        "description": "Возвращает список атрибутов указанного корпуса.",
        "requires_params": ["corpus"]
    },
    "get_attribute_values": {
        "description": "Возвращает значения конкретного атрибута в корпусе.",
        "requires_params": ["attr_name", "corpus"]
    },
    "check_auth": {
        "description": "Проверяет, авторизован ли пользователь в системе.",
        "requires_params": []
    }
}

def get_registry_description() -> str:
    """Превращает словарь выше в понятный для модели текст"""
    desc = "Доступные инструменты API НКРЯ:\n"
    for tool_name, details in CAPABILITY_REGISTRY.items():
        desc += f"- {tool_name}: {details['description']} Параметры: {details['requires_params']}\n"
    return desc