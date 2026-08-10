"""Contract tests for the lazy, settings-driven external clients."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.config import MissingSecretError, Settings


def _settings(**overrides: Any) -> Settings:
    values = {
        "APP_ENV": "test",
        "NKRJA_API_KEY": "nkrja-secret",
        "NKRJA_BASE_URL": "https://nkrja.example/api",
        "VSEGPT_API_KEY": "vsegpt-secret",
        "VSEGPT_BASE_URL": "https://vsegpt.example/v1",
        "VSEGPT_MODEL": "example/model",
        "HTTP_TIMEOUT_SECONDS": "17.5",
        "HTTP_MAX_RETRIES": "2",
        "LLM_MAX_OUTPUT_TOKENS": "777",
        "LLM_TEMPERATURE": "0.4",
    }
    values.update(overrides)
    return Settings.from_env(values)


def test_llm_factory_uses_typed_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from LLM import client as llm_module

    captured: dict[str, Any] = {}

    def fake_chat_openai(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm_module, "ChatOpenAI", fake_chat_openai)
    llm_module.get_llm(_settings())

    assert captured["model"] == "example/model"
    assert captured["base_url"] == "https://vsegpt.example/v1"
    assert captured["max_tokens"] == 777
    assert captured["temperature"] == 0.4
    assert captured["timeout"] == 17.5
    assert captured["max_retries"] == 2
    assert captured["api_key"].get_secret_value() == "vsegpt-secret"


def test_llm_factory_rejects_missing_secret_at_creation() -> None:
    from LLM.client import get_llm

    with pytest.raises(MissingSecretError, match="VSEGPT_API_KEY"):
        get_llm(_settings(VSEGPT_API_KEY=""))


@pytest.mark.asyncio
async def test_nkrja_client_uses_configured_transport_boundary() -> None:
    from tools.nkrja_client import NKRJAClient

    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handle)
    async with httpx.AsyncClient(transport=transport) as session:
        client = NKRJAClient(_settings(), client=session)
        await client.get_corpus_stats("MAIN")

    request = requests[0]
    assert request.method == "GET"
    assert str(request.url).startswith("https://nkrja.example/api/v1/stats/")
    assert request.headers["Authorization"] == "Bearer nkrja-secret"


def test_nkrja_client_rejects_missing_secret_at_creation() -> None:
    from tools.nkrja_client import NKRJAClient

    with pytest.raises(MissingSecretError, match="NKRJA_API_KEY"):
        NKRJAClient(_settings(NKRJA_API_KEY=""))
