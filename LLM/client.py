"""Factory for the configured VseGPT chat model."""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import Settings, get_settings


def get_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Build an LLM client only when a graph node actually needs one."""

    current = settings or get_settings()
    return ChatOpenAI(
        model=current.vsegpt_model,
        api_key=SecretStr(current.require_vsegpt_api_key()),
        base_url=current.vsegpt_base_url,
        max_tokens=current.llm_max_output_tokens,
        temperature=current.llm_temperature,
        timeout=current.http_timeout_seconds,
        max_retries=current.http_max_retries,
    )
