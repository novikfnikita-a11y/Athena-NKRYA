import requests
import json
from app.config import NKRJA_API_KEY

class NKRJAClient:
    def __init__(self):
        self.base_url = "https://ruscorpora.ru"
        self.headers = {
            "Authorization": f"Bearer {NKRJA_API_KEY}",
            "Content-Type": "application/json"
        }

    def _make_get_request(self, endpoint: str, param_name: str, payload: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        params = {param_name: json.dumps(payload)} if payload else {}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json() if response.text else {"status": "ok"}

    def _make_post_request(self, endpoint: str, payload: dict) -> dict:
        """вспомогательный метод для выполнения POST-запросов (требуется для конкорданса)"""
        url = f"{self.base_url}{endpoint}"
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json() if response.text else {"status": "ok"}

    def _safe_corpus(self, corpus: str) -> str:
        if not corpus:
            return "MAIN"

        c = str(corpus).strip().upper()

        # легаси , тк в новом провайдере моделей уже есть метод подобный.
        mapping = {
            "ОСНОВНОЙ": "MAIN",
            "УСТНЫЙ": "SPOKEN",
            "ПОЭТИЧЕСКИЙ": "POETIC",
            "MAIN_CORPUS": "MAIN",
            "ГАЗЕТНЫЙ": "NEWSPAPER",
            "ОБУЧАЮЩИЙ": "EDUCATIONAL",
            "МУЛЬТИМЕДИЙНЫЙ": "MULTIMEDIA"
        }


        return mapping.get(c, c)

    def _safe_string(self, text: str) -> str:
        # очистка лемм и параметров от случайных пробелов
        return str(text).strip() if text else ""

    def get_word_portrait(self, lemma: str) -> dict:
        query_data = {
            "lemma": self._safe_string(lemma),
            "corpus": {"type": "MAIN"},
            "resultType": ["PORTRAIT_WORD_INFO", "PORTRAIT_STATS"]
        }
        return self._make_get_request("/api/v1/word-portrait/", "query", query_data)

    def get_corpus_stats(self, corpus: str = "MAIN") -> dict:
        corpus_data = {"type": self._safe_corpus(corpus)}
        return self._make_get_request("/api/v1/stats/", "corpus", corpus_data)

    def get_sketch_difference(self, lemma_1: str, lemma_2: str, corpus: str = "MAIN", pos: str = "A") -> dict:
        safe_pos = str(pos).strip().upper() if pos else "A"
        query_data = {
            "lemma_1": self._safe_string(lemma_1),
            "lemma_2": self._safe_string(lemma_2),
            "corpus": {"type": self._safe_corpus(corpus)},
            "pos": safe_pos
        }
        return self._make_get_request("/api/v1/word-portrait/sketch-difference", "query", query_data)

    def get_lex_gramm_search_form(self, corpus: str = "MAIN") -> dict:
        corpus_data = {"type": self._safe_corpus(corpus)}
        return self._make_get_request("/api/v1/lex-gramm/search-form", "corpus", corpus_data)

    def get_simple_concordance(self, lemma: str, corpus: str = "MAIN") -> dict:

        payload = {
            "corpus": {
                "type": self._safe_corpus(corpus)
            },
            "lexGramm": {
                "sectionValues": [
                    {
                        "subsectionValues": [
                            {
                                "conditionValues": [
                                    {
                                        "fieldName": "lex",
                                        "text": {
                                            "v": self._safe_string(lemma)
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        }

        return self._make_post_request("/api/v1/lex-gramm/concordance", payload)

    def get_corpus_config(self, corpus: str = "MAIN") -> dict:
        corpus_data = {"type": self._safe_corpus(corpus)}
        return self._make_get_request("/api/v1/config/", "corpus", corpus_data)

    def get_corpus_attributes(self, corpus: str = "MAIN") -> dict:
        corpus_data = {"type": self._safe_corpus(corpus)}
        return self._make_get_request("/api/v1/attrs/", "corpus", corpus_data)

    def get_attribute_values(self, attr_name: str, corpus: str = "MAIN") -> dict:
        corpus_data = {"type": self._safe_corpus(corpus)}
        return self._make_get_request(f"/api/v1/attrs/{self._safe_string(attr_name)}", "corpus", corpus_data)

    def check_auth(self) -> dict:
        url = f"{self.base_url}/api/v1/auth/check-authenticated/"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return {"is_authenticated": response.json()}