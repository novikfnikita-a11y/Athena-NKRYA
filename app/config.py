"""Lazy, typed runtime configuration for Athena-NKRYA.

Importing this module is intentionally safe without API keys.  Environment
variables (and an optional ``.env`` file) are read only when ``get_settings``
is first called.  A service must request its own secret immediately before its
client is created by calling the corresponding ``require_*`` method.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Mapping, cast
from urllib.parse import urlparse

from dotenv import load_dotenv


EnvironmentName = Literal["development", "test", "staging", "production"]


class ConfigurationError(ValueError):
    """Raised when a configured value has an invalid type or range."""


class MissingSecretError(ConfigurationError):
    """Raised when a client is about to be created without its API key."""


def _optional_value(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _value(
    environ: Mapping[str, str],
    name: str,
    default: str,
) -> str:
    return _optional_value(environ, name) or default


def _integer(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    raw_value = _optional_value(environ, name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ConfigurationError(
            f"{name} должен быть целым числом, "
            f"получено: {raw_value!r}"
        ) from error
    if value < minimum:
        raise ConfigurationError(
            f"{name} должен быть не меньше {minimum}, "
            f"получено: {value}"
        )
    return value


def _number(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    raw_value = _optional_value(environ, name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as error:
        raise ConfigurationError(
            f"{name} должен быть числом, "
            f"получено: {raw_value!r}"
        ) from error
    if value < minimum or (maximum is not None and value > maximum):
        upper_bound = f" и не больше {maximum}" if maximum is not None else ""
        raise ConfigurationError(
            f"{name} должен быть не меньше {minimum}{upper_bound}, "
            f"получено: {value}"
        )
    return value


def _boolean_value(raw_value: str | None, name: str, default: bool) -> bool:
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(
        f"{name} должен быть логическим значением "
        "true/false, "
        f"получено: {raw_value!r}"
    )


def _http_url(environ: Mapping[str, str], name: str, default: str) -> str:
    value = _value(environ, name, default).rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(
            f"{name} должен быть полным HTTP(S)-адресом, "
            f"получено: {value!r}"
        )
    return value


def _environment(environ: Mapping[str, str]) -> EnvironmentName:
    raw_value = _value(environ, "APP_ENV", "development").lower()
    aliases = {
        "dev": "development",
        "testing": "test",
        "stage": "staging",
        "prod": "production",
    }
    value = aliases.get(raw_value, raw_value)
    allowed = {"development", "test", "staging", "production"}
    if value not in allowed:
        raise ConfigurationError(
            "APP_ENV должен быть одним из development, test, staging, "
            f"production; получено: {raw_value!r}"
        )
    return cast(EnvironmentName, value)


def _required_secret(
    value: str | None,
    variable_name: str,
    client_name: str,
) -> str:
    if value:
        return value
    raise MissingSecretError(
        f"Переменная окружения {variable_name} обязательна "
        "для создания "
        f"клиента {client_name}."
    )


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable configuration snapshot built from environment variables."""

    environment: EnvironmentName

    nkrja_api_key: str | None = field(repr=False)
    nkrja_base_url: str
    vsegpt_api_key: str | None = field(repr=False)
    vsegpt_base_url: str
    vsegpt_model: str

    http_timeout_seconds: float
    http_max_retries: int
    max_concurrent_requests: int

    max_research_iterations: int
    evidence_context_budget_chars: int
    llm_max_output_tokens: int
    llm_temperature: float

    langsmith_tracing: bool
    langsmith_api_key: str | None = field(repr=False)
    langsmith_endpoint: str
    langsmith_project: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Create a validated snapshot without reading or mutating ``.env``."""

        source = os.environ if environ is None else environ
        tracing_raw = source.get("LANGSMITH_TRACING")
        tracing_name = "LANGSMITH_TRACING"
        if tracing_raw is None:
            tracing_raw = source.get("LANGCHAIN_TRACING_V2")
            tracing_name = "LANGCHAIN_TRACING_V2"

        return cls(
            environment=_environment(source),
            nkrja_api_key=_optional_value(source, "NKRJA_API_KEY"),
            nkrja_base_url=_http_url(
                source,
                "NKRJA_BASE_URL",
                "https://ruscorpora.ru",
            ),
            vsegpt_api_key=_optional_value(source, "VSEGPT_API_KEY"),
            vsegpt_base_url=_http_url(
                source,
                "VSEGPT_BASE_URL",
                "https://api.vsegpt.ru/v1",
            ),
            vsegpt_model=_value(
                source,
                "VSEGPT_MODEL",
                "deepseek/deepseek-chat:nojsonencode:x-title=NKRJA_Research_Agent",
            ),
            http_timeout_seconds=_number(
                source,
                "HTTP_TIMEOUT_SECONDS",
                30.0,
                minimum=0.1,
            ),
            http_max_retries=_integer(
                source,
                "HTTP_MAX_RETRIES",
                3,
                minimum=0,
            ),
            max_concurrent_requests=_integer(
                source,
                "MAX_CONCURRENT_REQUESTS",
                4,
                minimum=1,
            ),
            max_research_iterations=_integer(
                source,
                "MAX_RESEARCH_ITERATIONS",
                4,
                minimum=1,
            ),
            evidence_context_budget_chars=_integer(
                source,
                "EVIDENCE_CONTEXT_BUDGET_CHARS",
                300_000,
                minimum=1,
            ),
            llm_max_output_tokens=_integer(
                source,
                "LLM_MAX_OUTPUT_TOKENS",
                2_048,
                minimum=1,
            ),
            llm_temperature=_number(
                source,
                "LLM_TEMPERATURE",
                0.2,
                minimum=0.0,
                maximum=2.0,
            ),
            langsmith_tracing=_boolean_value(tracing_raw, tracing_name, False),
            langsmith_api_key=(
                _optional_value(source, "LANGSMITH_API_KEY")
                or _optional_value(source, "LANGCHAIN_API_KEY")
            ),
            langsmith_endpoint=_http_url(
                source,
                "LANGSMITH_ENDPOINT",
                _value(
                    source,
                    "LANGCHAIN_ENDPOINT",
                    "https://api.smith.langchain.com",
                ),
            ),
            langsmith_project=(
                _optional_value(source, "LANGSMITH_PROJECT")
                or _optional_value(source, "LANGCHAIN_PROJECT")
                or "nkrja-multiagent-agent"
            ),
        )

    def require_nkrja_api_key(self) -> str:
        """Return the NKRJA key or fail at the NKRJA client boundary."""

        return _required_secret(self.nkrja_api_key, "NKRJA_API_KEY", "НКРЯ")

    def require_vsegpt_api_key(self) -> str:
        """Return the VseGPT key or fail at the LLM client boundary."""

        return _required_secret(self.vsegpt_api_key, "VSEGPT_API_KEY", "VseGPT")

    def require_langsmith_api_key(self) -> str | None:
        """Require a tracing key only when LangSmith tracing is enabled."""

        if not self.langsmith_tracing:
            return None
        return _required_secret(
            self.langsmith_api_key,
            "LANGSMITH_API_KEY (или LANGCHAIN_API_KEY)",
            "LangSmith",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load ``.env`` once and return the process-wide settings snapshot."""

    load_dotenv(override=False)
    return Settings.from_env()


def clear_settings_cache() -> None:
    """Forget the cached snapshot after an intentional environment change."""

    get_settings.cache_clear()


def require_nkrja_api_key() -> str:
    """Client-factory helper retained at module level for concise imports."""

    return get_settings().require_nkrja_api_key()


def require_vsegpt_api_key() -> str:
    """Client-factory helper retained at module level for concise imports."""

    return get_settings().require_vsegpt_api_key()


if TYPE_CHECKING:
    # Compatibility names for modules that have not yet moved to get_settings().
    NKRJA_API_KEY: str | None
    VSEGPT_API_KEY: str | None


_LEGACY_SETTINGS = {
    "NKRJA_API_KEY": "nkrja_api_key",
    "VSEGPT_API_KEY": "vsegpt_api_key",
}


def __getattr__(name: str) -> object:
    """Serve legacy key imports without validating secrets during import."""

    setting_name = _LEGACY_SETTINGS.get(name)
    if setting_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(get_settings(), setting_name)


def __dir__() -> list[str]:
    return sorted([*globals(), *_LEGACY_SETTINGS])


__all__ = [
    "ConfigurationError",
    "EnvironmentName",
    "MissingSecretError",
    "NKRJA_API_KEY",
    "Settings",
    "VSEGPT_API_KEY",
    "clear_settings_cache",
    "get_settings",
    "require_nkrja_api_key",
    "require_vsegpt_api_key",
]
