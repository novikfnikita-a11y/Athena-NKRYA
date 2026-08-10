"""Asynchronous, bounded HTTP client for the NKRJA API."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from app.config import Settings, get_settings
from tools.registry import normalize_corpus


class NKRJAError(RuntimeError):
    """Base class for safe, typed NKRJA client failures."""


class NKRJATimeoutError(NKRJAError):
    pass


class NKRJATransportError(NKRJAError):
    pass


class NKRJARateLimitError(NKRJAError):
    pass


class NKRJAResponseError(NKRJAError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class NKRJADecodeError(NKRJAError):
    pass


Sleep = Callable[[float], Awaitable[None]]


class NKRJAClient:
    """One reusable ``httpx.AsyncClient`` with retry and concurrency bounds."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.AsyncClient | None = None,
        *,
        session: httpx.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        current = settings or get_settings()
        supplied_client = client or session
        self.base_url = current.nkrja_base_url.rstrip("/")
        self.max_retries = current.http_max_retries
        self._sleep = sleep
        self._random = random_value
        self._semaphore = asyncio.Semaphore(current.max_concurrent_requests)
        self._owns_client = supplied_client is None
        self._closed = False
        self._client = supplied_client or httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {current.require_nkrja_api_key()}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(
                connect=current.http_timeout_seconds,
                read=current.http_timeout_seconds,
                write=current.http_timeout_seconds,
                pool=current.http_timeout_seconds,
            ),
        )
        if supplied_client is not None:
            # The caller owns transport lifecycle, but authentication still
            # belongs to this API boundary.
            token = current.require_nkrja_api_key()
            self._headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        else:
            self._headers = {}

    def _url(self, endpoint: str) -> str:
        path = endpoint.strip().lstrip("/")
        if self.base_url.endswith("/api") and path.startswith("api/"):
            path = path[4:]
        return f"{self.base_url}/{path}"

    @staticmethod
    def _retry_after_seconds(
        headers: Mapping[str, str],
        *,
        now: datetime | None = None,
    ) -> float | None:
        raw = headers.get("Retry-After")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(raw)
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            current = now or datetime.now(timezone.utc)
            return max(0.0, (retry_at - current).total_seconds())

    def _backoff(self, attempt: int) -> float:
        base = min(8.0, 0.5 * (2**attempt))
        return base + self._random() * min(1.0, base * 0.25)

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        require_object: bool = True,
    ) -> Any:
        if self._closed:
            raise NKRJATransportError("NKRJA client is closed.")
        url = self._url(endpoint)

        async with self._semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await self._client.request(
                        method,
                        url,
                        headers=self._headers or None,
                        params=params,
                        json=json_body,
                    )
                except httpx.TimeoutException as error:
                    if attempt < self.max_retries:
                        await self._sleep(self._backoff(attempt))
                        continue
                    raise NKRJATimeoutError("NKRJA request timed out.") from error
                except httpx.TransportError as error:
                    if attempt < self.max_retries:
                        await self._sleep(self._backoff(attempt))
                        continue
                    raise NKRJATransportError("NKRJA transport failed.") from error

                if response.status_code == 429:
                    delay = self._retry_after_seconds(response.headers)
                    if attempt < self.max_retries:
                        await self._sleep(
                            min(60.0, delay if delay is not None else self._backoff(attempt))
                        )
                        continue
                    raise NKRJARateLimitError(
                        "NKRJA rate limit remained active after bounded retries."
                    )

                if 500 <= response.status_code < 600:
                    if attempt < self.max_retries:
                        await self._sleep(self._backoff(attempt))
                        continue
                    raise NKRJAResponseError(
                        response.status_code,
                        f"NKRJA server returned HTTP {response.status_code} after retries.",
                    )

                if response.is_error:
                    raise NKRJAResponseError(
                        response.status_code,
                        f"NKRJA request failed with HTTP {response.status_code}.",
                    )
                if not response.content:
                    return {"status": "ok"}
                try:
                    payload = response.json()
                except (ValueError, json.JSONDecodeError) as error:
                    raise NKRJADecodeError("NKRJA returned invalid JSON.") from error
                if require_object and not isinstance(payload, dict):
                    raise NKRJADecodeError("NKRJA response root must be an object.")
                return payload

        raise AssertionError("unreachable retry loop")

    async def _make_get_request(
        self,
        endpoint: str,
        param_name: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        params = (
            {param_name: json.dumps(payload, ensure_ascii=False)} if payload else None
        )
        return await self._request("GET", endpoint, params=params)

    async def _make_post_request(
        self,
        endpoint: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return await self._request("POST", endpoint, json_body=payload)

    @staticmethod
    def _safe_corpus(corpus: str | None) -> str:
        return normalize_corpus(corpus).value

    @staticmethod
    def _safe_string(text: str | None) -> str:
        return str(text).strip() if text else ""

    async def get_word_portrait(
        self,
        lemma: str,
        corpus: str,
        resultType: list[str],
        pos: str | None = None,
        seed: int | None = None,
        statFields: list[str] | None = None,
        similarCategories: list[str] | None = None,
    ) -> dict[str, Any]:
        query_data: dict[str, Any] = {
            "lemma": self._safe_string(lemma),
            "corpus": {"type": self._safe_corpus(corpus)},
            "resultType": list(resultType),
        }
        if pos:
            query_data["pos"] = str(pos).strip().upper()
        if seed is not None:
            query_data["seed"] = seed
        if statFields:
            query_data["statFields"] = list(statFields)
        if similarCategories:
            query_data["similarCategories"] = list(similarCategories)
        return await self._make_get_request(
            "/api/v1/word-portrait/", "query", query_data
        )

    async def get_corpus_stats(self, corpus: str = "MAIN") -> dict[str, Any]:
        return await self._make_get_request(
            "/api/v1/stats/",
            "corpus",
            {"type": self._safe_corpus(corpus)},
        )

    async def get_sketch_difference(
        self,
        lemma_1: str,
        lemma_2: str,
        corpus: str = "MAIN",
        pos: str = "A",
    ) -> dict[str, Any]:
        query_data = {
            "lemma_1": self._safe_string(lemma_1),
            "lemma_2": self._safe_string(lemma_2),
            "corpus": {"type": self._safe_corpus(corpus)},
            "pos": str(pos or "A").strip().upper(),
        }
        return await self._make_get_request(
            "/api/v1/word-portrait/sketch-difference", "query", query_data
        )

    async def get_lex_gramm_search_form(
        self, corpus: str = "MAIN"
    ) -> dict[str, Any]:
        return await self._make_get_request(
            "/api/v1/lex-gramm/search-form",
            "corpus",
            {"type": self._safe_corpus(corpus)},
        )

    async def get_simple_concordance(
        self, lemma: str, corpus: str = "MAIN"
    ) -> dict[str, Any]:
        payload = {
            "corpus": {"type": self._safe_corpus(corpus)},
            "lexGramm": {
                "sectionValues": [
                    {
                        "subsectionValues": [
                            {
                                "conditionValues": [
                                    {
                                        "fieldName": "lex",
                                        "text": {"v": self._safe_string(lemma)},
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
        }
        return await self._make_post_request(
            "/api/v1/lex-gramm/concordance", payload
        )

    async def get_corpus_config(self, corpus: str = "MAIN") -> dict[str, Any]:
        return await self._make_get_request(
            "/api/v1/config/", "corpus", {"type": self._safe_corpus(corpus)}
        )

    async def get_corpus_attributes(
        self, corpus: str = "MAIN"
    ) -> dict[str, Any]:
        return await self._make_get_request(
            "/api/v1/attrs/", "corpus", {"type": self._safe_corpus(corpus)}
        )

    async def get_attribute_values(
        self, attr_name: str, corpus: str = "MAIN"
    ) -> dict[str, Any]:
        return await self._make_get_request(
            f"/api/v1/attrs/{self._safe_string(attr_name)}",
            "corpus",
            {"type": self._safe_corpus(corpus)},
        )

    async def check_auth(self) -> dict[str, Any]:
        payload = await self._request(
            "GET",
            "/api/v1/auth/check-authenticated/",
            require_object=False,
        )
        return {"is_authenticated": payload}

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "NKRJAClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()


__all__ = [
    "NKRJAClient",
    "NKRJADecodeError",
    "NKRJAError",
    "NKRJARateLimitError",
    "NKRJAResponseError",
    "NKRJATimeoutError",
    "NKRJATransportError",
]
