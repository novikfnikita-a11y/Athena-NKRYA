import requests

API_KEY = "6oq97zdSLzY:8adc6fca9cfd8b2a262e3f07d87143e5810e321d"

BASE_URL = "https://ruscorpora.ru"

# спецификация требует тип авторизации Bearer
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def check_authorization():
    print("Проверка авторизации...")
    url = f"{BASE_URL}/api/v1/auth/check-authenticated/"

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("Успех! Статус авторизации:", response.json())
    else:
        print(f"Ошибка {response.status_code}: Не удалось авторизоваться.")
        print(response.text)


def test_word_portrait(target_lemma):
    print(f"\nПолучаем портрет для леммы: '{target_lemma}'...")
    url = f"{BASE_URL}/api/v1/word-portrait/"

    params = {
        "query": f'{{"lemma":"{target_lemma}","corpus":{{"type":"MAIN"}},"resultType":["PORTRAIT_WORD_INFO"]}}'
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        print("Данные успешно получены!")
        data = response.json()
        possible_pos = data.get("possiblePos", [])
        print(f"Доступные части речи для слова '{target_lemma}': {possible_pos}")
    else:
        print(f"Ошибка {response.status_code} при запросе портрета слова.")
        print(response.text)


if __name__ == "__main__":
    check_authorization()

    test_word_portrait("печь")