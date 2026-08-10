import json
from typing import Any

import requests
from langsmith import traceable  # НОВОЕ: Импорт трейсера для низкоуровневых запросов
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import Settings, get_settings


def _build_session(max_retries: int) -> requests.Session:
    session = requests.Session()
    retry_policy = Retry(
        total=max_retries,
        connect=max_retries,
        read=max_retries,
        status=max_retries,
        allowed_methods=None,
        status_forcelist=(429, 500, 502, 503, 504),
        backoff_factor=0.5,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry_policy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class NKRJAClient:
    def __init__(
        self,
        settings: Settings | None = None,
        session: Any | None = None,
    ):
        current = settings or get_settings()
        self.base_url = current.nkrja_base_url
        self.timeout = current.http_timeout_seconds
        self.session = session or _build_session(current.http_max_retries)
        self.headers = {
            "Authorization": f"Bearer {current.require_nkrja_api_key()}",
            "Content-Type": "application/json"
        }

    # НОВОЕ: Автоматически отправляет параметры GET-запроса и ответ в дашборд
    @traceable(run_type="tool", name="NKRJA_HTTP_GET")
    def _make_get_request(self, endpoint: str, param_name: str, payload: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        params = {param_name: json.dumps(payload, ensure_ascii=False)} if payload else {}
        response = self.session.get(
            url,
            headers=self.headers,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json() if response.text else {"status": "ok"}

    # НОВОЕ: Автоматически отправляет параметры POST-запроса и ответ в дашборд
    @traceable(run_type="tool", name="NKRJA_HTTP_POST")
    def _make_post_request(self, endpoint: str, payload: dict) -> dict:
        """вспомогательный метод для выполнения POST-запросов (требуется для конкорданса)"""
        url = f"{self.base_url}{endpoint}"
        response = self.session.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json() if response.text else {"status": "ok"}

    def _safe_corpus(self, corpus: str) -> str:
        if not corpus:
            return "MAIN"
        c = str(corpus).strip().upper()
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
        return str(text).strip() if text else ""

    def get_word_portrait(
            self,
            lemma: str,
            corpus: str,
            resultType: list,
            pos: str = None,
            seed: int = None,
            statFields: list = None,
            similarCategories: list = None
    ) -> dict:
        query_data = {
            "lemma": self._safe_string(lemma),
            "corpus": {"type": self._safe_corpus(corpus)},
            "resultType": resultType
        }
        if pos:
            query_data["pos"] = str(pos).strip().upper()
        if seed is not None:
            query_data["seed"] = seed
        if statFields:
            query_data["statFields"] = statFields
        if similarCategories:
            query_data["similarCategories"] = similarCategories

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
        response = self.session.get(
            url,
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return {"is_authenticated": response.json()}
