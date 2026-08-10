"""Lazy factories for replaceable chat and structured-output models.

The rest of Athena depends only on LangChain's ``invoke``/``ainvoke`` surface.
VseGPT is therefore a deployment choice, not a domain dependency: changing the
configured OpenAI-compatible base URL and model is enough to use another
provider or a self-hosted endpoint.
"""

from __future__ import annotations

from typing import Any, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

from app.config import Settings, get_settings


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


def get_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Build a provider-neutral chat client at the point of use."""

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


def get_structured_llm(
    schema: type[StructuredModel],
    settings: Settings | None = None,
    **kwargs: Any,
) -> Any:
    """Return LangChain's native structured mode for callers that support it.

    Research nodes intentionally use :func:`LLM.structured.ainvoke_structured`
    as their portable fallback because not every OpenAI-compatible endpoint
    implements native JSON Schema identically.  This factory still exposes the
    native mode for providers where it is reliable.
    """

    return get_llm(settings).with_structured_output(schema, **kwargs)


__all__ = ["get_llm", "get_structured_llm"]
