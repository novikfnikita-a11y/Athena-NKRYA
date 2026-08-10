"""Shared deterministic and offline fixtures for the Athena test suite.

The production code currently reads ``.env`` files and validates API keys while
modules are imported.  Pytest imports this file before collecting test modules,
so the session hooks below establish a safe test environment early enough to
prevent those import-time side effects.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time as time_module
import uuid as uuid_module
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

import pytest


FIXED_TIME = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)

_TEST_ENVIRONMENT = {
    "NKRJA_API_KEY": "test-nkrja-key",
    "VSEGPT_API_KEY": "test-vsegpt-key",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGSMITH_TRACING": "false",
    "PYTHON_DOTENV_DISABLED": "1",
}

_EXTERNAL_TRACE_ENVIRONMENT = {
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_ENDPOINT",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_API_KEY",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
}

_MANAGED_ENVIRONMENT = set(_TEST_ENVIRONMENT) | _EXTERNAL_TRACE_ENVIRONMENT
_ORIGINAL_ENVIRONMENT: dict[str, str | None] = {}
_ORIGINAL_LOAD_DOTENV: Callable[..., Any] | None = None
_ORIGINAL_TIME = time_module.time
_ORIGINAL_UUID4 = uuid_module.uuid4
_ORIGINAL_CREATE_CONNECTION = socket.create_connection
_ORIGINAL_GETADDRINFO = socket.getaddrinfo
_ORIGINAL_SOCKET_CONNECT = socket.socket.connect
_ORIGINAL_SOCKET_CONNECT_EX = socket.socket.connect_ex
_SESSION_CONFIGURED = False


class FrozenClock:
    """Small controllable clock used by code that calls ``time.time()``."""

    def __init__(self, initial: datetime) -> None:
        self._initial = initial
        self.reset()

    def reset(self) -> None:
        self._timestamp = self._initial.timestamp()

    def time(self) -> float:
        return self._timestamp

    def now(self, tz: timezone | None = timezone.utc) -> datetime:
        value = datetime.fromtimestamp(self._timestamp, tz=timezone.utc)
        if tz is None:
            return value.replace(tzinfo=None)
        return value.astimezone(tz)

    def advance(self, seconds: float) -> None:
        self._timestamp += seconds


class DeterministicUUIDFactory:
    """Return unique but reproducible version-4 UUIDs."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._next_value = 1

    def __call__(self) -> uuid_module.UUID:
        value = uuid_module.UUID(int=self._next_value, version=4)
        self._next_value += 1
        return value


_CLOCK = FrozenClock(FIXED_TIME)
_UUIDS = DeterministicUUIDFactory()


def _disabled_load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
    """Act like an absent dotenv file without touching the filesystem."""

    return False


class NetworkAccessBlocked(RuntimeError):
    """Raised when an ordinary test attempts to open a real socket."""


def _reject_network(*_args: Any, **_kwargs: Any) -> None:
    raise NetworkAccessBlocked(
        "Real network access is disabled in tests. Use a fake transport or an "
        "explicit @pytest.mark.live test with ATHENA_RUN_LIVE_TESTS=1."
    )


def pytest_configure(config: pytest.Config) -> None:
    """Establish isolation before pytest imports any project test modules."""

    global _ORIGINAL_LOAD_DOTENV, _SESSION_CONFIGURED

    if _SESSION_CONFIGURED:
        return

    _ORIGINAL_ENVIRONMENT.update(
        {name: os.environ.get(name) for name in _MANAGED_ENVIRONMENT}
    )

    for name in _EXTERNAL_TRACE_ENVIRONMENT:
        os.environ.pop(name, None)
    os.environ.update(_TEST_ENVIRONMENT)

    try:
        import dotenv
    except ModuleNotFoundError:
        dotenv = None

    if dotenv is not None:
        _ORIGINAL_LOAD_DOTENV = dotenv.load_dotenv
        dotenv.load_dotenv = _disabled_load_dotenv

    # Install these before collection so modules using ``from uuid import
    # uuid4`` bind the deterministic callable as well.
    time_module.time = _CLOCK.time
    uuid_module.uuid4 = _UUIDS
    socket.create_connection = _reject_network
    socket.getaddrinfo = _reject_network
    socket.socket.connect = _reject_network
    socket.socket.connect_ex = _reject_network

    config.addinivalue_line(
        "markers",
        "live: test allowed to use real services only with ATHENA_RUN_LIVE_TESTS=1",
    )
    _SESSION_CONFIGURED = True


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore process state for callers embedding pytest."""

    global _SESSION_CONFIGURED

    del config

    time_module.time = _ORIGINAL_TIME
    uuid_module.uuid4 = _ORIGINAL_UUID4
    socket.create_connection = _ORIGINAL_CREATE_CONNECTION
    socket.getaddrinfo = _ORIGINAL_GETADDRINFO
    socket.socket.connect = _ORIGINAL_SOCKET_CONNECT
    socket.socket.connect_ex = _ORIGINAL_SOCKET_CONNECT_EX

    if _ORIGINAL_LOAD_DOTENV is not None:
        import dotenv

        dotenv.load_dotenv = _ORIGINAL_LOAD_DOTENV

    for name, value in _ORIGINAL_ENVIRONMENT.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    _SESSION_CONFIGURED = False


@pytest.fixture(autouse=True)
def isolated_test_runtime(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> Iterable[None]:
    """Reset deterministic values and prohibit network access per test."""

    _CLOCK.reset()
    _UUIDS.reset()

    for name in _EXTERNAL_TRACE_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    for name, value in _TEST_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)

    live_test = request.node.get_closest_marker("live") is not None
    live_enabled = os.environ.get("ATHENA_RUN_LIVE_TESTS") == "1"

    if live_test and not live_enabled:
        pytest.skip("Live services require ATHENA_RUN_LIVE_TESTS=1")

    if live_test:
        monkeypatch.setattr(socket, "create_connection", _ORIGINAL_CREATE_CONNECTION)
        monkeypatch.setattr(socket, "getaddrinfo", _ORIGINAL_GETADDRINFO)
        monkeypatch.setattr(socket.socket, "connect", _ORIGINAL_SOCKET_CONNECT)
        monkeypatch.setattr(socket.socket, "connect_ex", _ORIGINAL_SOCKET_CONNECT_EX)

    yield


@pytest.fixture
def frozen_clock() -> FrozenClock:
    """Return the clock already installed as ``time.time``."""

    return _CLOCK


@pytest.fixture
def deterministic_uuids() -> DeterministicUUIDFactory:
    """Return the UUID generator already installed as ``uuid.uuid4``."""

    return _UUIDS


@dataclass(frozen=True)
class FakeLLMResponse:
    content: Any


@dataclass(frozen=True)
class LLMCall:
    input: Any
    config: Any
    kwargs: Mapping[str, Any]
    is_async: bool


QueuedLLMResponse = Any | BaseException | Callable[[LLMCall], Any]


class FakeLLM:
    """Queue-based replacement supporting both sync and async LangChain calls."""

    def __init__(self, responses: Iterable[QueuedLLMResponse] = ()) -> None:
        self._responses = deque(responses)
        self.calls: list[LLMCall] = []

    def queue(self, *responses: QueuedLLMResponse) -> "FakeLLM":
        self._responses.extend(responses)
        return self

    def queue_json(self, *responses: Mapping[str, Any]) -> "FakeLLM":
        self._responses.extend(
            json.dumps(response, ensure_ascii=False) for response in responses
        )
        return self

    def invoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> FakeLLMResponse:
        return self._dispatch(input, config, kwargs, is_async=False)

    async def ainvoke(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ) -> FakeLLMResponse:
        return self._dispatch(input, config, kwargs, is_async=True)

    def bind(self, **_kwargs: Any) -> "FakeLLM":
        return self

    def bind_tools(self, *_args: Any, **_kwargs: Any) -> "FakeLLM":
        return self

    def with_structured_output(
        self,
        schema: type[Any],
        **kwargs: Any,
    ) -> "FakeStructuredLLM":
        return FakeStructuredLLM(
            model=self,
            schema=schema,
            include_raw=bool(kwargs.get("include_raw")),
        )

    def _dispatch(
        self,
        input: Any,
        config: Any,
        kwargs: Mapping[str, Any],
        *,
        is_async: bool,
    ) -> FakeLLMResponse:
        call = LLMCall(
            input=input,
            config=config,
            kwargs=dict(kwargs),
            is_async=is_async,
        )
        self.calls.append(call)

        if not self._responses:
            raise AssertionError(
                "FakeLLM received an unexpected call; queue a response first."
            )

        response = self._responses.popleft()
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            response = response(call)
        if hasattr(response, "content"):
            return response
        if isinstance(response, (dict, list)):
            response = json.dumps(response, ensure_ascii=False)
        return FakeLLMResponse(content=str(response))


@dataclass(frozen=True)
class FakeStructuredLLM:
    model: FakeLLM
    schema: type[Any]
    include_raw: bool = False

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raw = self.model.invoke(input, config=config, **kwargs)
        return self._parse(raw)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raw = await self.model.ainvoke(input, config=config, **kwargs)
        return self._parse(raw)

    def _parse(self, raw: FakeLLMResponse) -> Any:
        content = raw.content
        if isinstance(content, self.schema):
            parsed = content
        elif isinstance(content, str):
            parsed = self.schema.model_validate_json(content)
        else:
            parsed = self.schema.model_validate(content)

        if self.include_raw:
            return {"raw": raw, "parsed": parsed, "parsing_error": None}
        return parsed


@pytest.fixture
def fake_llm_factory() -> type[FakeLLM]:
    return FakeLLM


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    """Install one controllable fake behind every loaded ``get_llm`` alias."""

    from LLM import client as llm_client

    model = FakeLLM()
    original_get_llm = llm_client.get_llm

    def get_fake_llm(*_args: Any, **_kwargs: Any) -> FakeLLM:
        return model

    monkeypatch.setattr(llm_client, "get_llm", get_fake_llm)

    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith(
            ("app.", "LLM.", "planners.", "research.")
        ):
            continue
        if getattr(module, "get_llm", None) is original_get_llm:
            monkeypatch.setattr(module, "get_llm", get_fake_llm)

    return model


@dataclass(frozen=True)
class NKRJACall:
    action: str
    args: tuple[Any, ...]
    params: Mapping[str, Any]


QueuedNKRJAResponse = Any | BaseException | Callable[[NKRJACall], Any]


class FakeNKRJAClient:
    """Deterministic replacement for the currently synchronous NKRJA client."""

    def __init__(self) -> None:
        self._responses: dict[str, deque[QueuedNKRJAResponse]] = defaultdict(deque)
        self.calls: list[NKRJACall] = []

    def queue(self, action: str, *responses: QueuedNKRJAResponse) -> "FakeNKRJAClient":
        self._responses[action].extend(responses)
        return self

    def _dispatch(self, action: str, *args: Any, **params: Any) -> Any:
        call = NKRJACall(action=action, args=args, params=dict(params))
        self.calls.append(call)

        if not self._responses[action]:
            raise AssertionError(
                f"FakeNKRJAClient received unexpected call {action!r}; "
                "queue a response first."
            )

        response = self._responses[action].popleft()
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return response(call)
        return response

    def get_word_portrait(self, *args: Any, **params: Any) -> Any:
        return self._dispatch("get_word_portrait", *args, **params)

    def get_corpus_stats(self, *args: Any, **params: Any) -> Any:
        return self._dispatch("get_corpus_stats", *args, **params)

    def get_sketch_difference(self, *args: Any, **params: Any) -> Any:
        return self._dispatch("get_sketch_difference", *args, **params)

    def get_lex_gramm_search_form(self, *args: Any, **params: Any) -> Any:
        return self._dispatch("get_lex_gramm_search_form", *args, **params)

    def get_simple_concordance(self, *args: Any, **params: Any) -> Any:
        return self._dispatch("get_simple_concordance", *args, **params)

    def get_corpus_config(self, *args: Any, **params: Any) -> Any:
        return self._dispatch("get_corpus_config", *args, **params)

    def get_corpus_attributes(self, *args: Any, **params: Any) -> Any:
        return self._dispatch("get_corpus_attributes", *args, **params)

    def get_attribute_values(self, *args: Any, **params: Any) -> Any:
        return self._dispatch("get_attribute_values", *args, **params)

    def check_auth(self, *args: Any, **params: Any) -> Any:
        return self._dispatch("check_auth", *args, **params)


@pytest.fixture
def fake_nkrja_factory() -> type[FakeNKRJAClient]:
    return FakeNKRJAClient


@pytest.fixture
def fake_nkrja(monkeypatch: pytest.MonkeyPatch) -> FakeNKRJAClient:
    """Install one fake NKRJA client in the client module and orchestrator."""

    from tools import nkrja_client as nkrja_module

    client = FakeNKRJAClient()
    original_client_class = nkrja_module.NKRJAClient

    def build_fake_client(*_args: Any, **_kwargs: Any) -> FakeNKRJAClient:
        return client

    monkeypatch.setattr(nkrja_module, "NKRJAClient", build_fake_client)

    for module_name, module in tuple(sys.modules.items()):
        if module is None or not module_name.startswith(("app.", "research.", "tools.")):
            continue

        if getattr(module, "NKRJAClient", None) is original_client_class:
            monkeypatch.setattr(module, "NKRJAClient", build_fake_client)

        existing_client = getattr(module, "client", None)
        if isinstance(existing_client, original_client_class):
            monkeypatch.setattr(module, "client", client)

    return client
