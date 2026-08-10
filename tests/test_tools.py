"""Stage-2 contracts for tools, evidence and bounded API execution."""

from __future__ import annotations

import asyncio
import copy
import time
from typing import Any

import httpx
import pytest

from app.config import Settings
from research.fact_extractor import extract_facts
from state.models import EvidenceArtifact
from tools.budgets import BudgetExceeded, BudgetLimits, BudgetTracker, action_signature
from tools.compatibility import (
    CompatibilityOverride,
    CompatibilityStatus,
    check_compatibility,
)
from tools.evidence_compressor import (
    CompressionLimits,
    compress_response,
    raw_response_hash,
)
from tools.nkrja_client import (
    NKRJAClient,
    NKRJAResponseError,
    NKRJATimeoutError,
)
from tools.orchestrator import api_orchestrator_node_async
from tools.registry import ResultType, TOOL_CAPABILITIES, normalize_corpus


def _settings(**overrides: str) -> Settings:
    values = {
        "APP_ENV": "test",
        "NKRJA_API_KEY": "nkrja-secret",
        "NKRJA_BASE_URL": "https://nkrja.example/api",
        "VSEGPT_API_KEY": "vsegpt-secret",
        "HTTP_TIMEOUT_SECONDS": "0.5",
        "HTTP_MAX_RETRIES": "2",
        "MAX_CONCURRENT_REQUESTS": "2",
    }
    values.update(overrides)
    return Settings.from_env(values)


def _state(*actions: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "research_id": "research-1",
        "run_id": "run-1",
        "branch_id": "branch-1",
        "batch_id": "batch-1",
        "iteration_count": 0,
        "planned_actions": list(actions),
        "completed_actions": [],
        "evidence": [],
        "budgets": {
            "max_branches": 2,
            "max_actions_per_branch": 8,
            "max_requests_per_iteration": 4,
            "max_external_calls": 8,
            "max_wall_time_seconds": 10,
            "max_evidence_items": 20,
            "max_context_chars": 50_000,
        },
    }
    state.update(overrides)
    return state


def test_registry_normalizes_canonical_corpus_names_and_conditions() -> None:
    assert normalize_corpus("NEWSPAPER").value == "PAPER"
    assert normalize_corpus("educational").value == "SCHOOL"
    assert normalize_corpus("multimedia").value == "MULTI"

    capability = TOOL_CAPABILITIES["get_word_portrait"]
    assert capability.validate_params(
        {
            "lemma": "мать",
            "corpus": "MAIN",
            "resultType": ["PORTRAIT_STATS"],
        }
    ) == ("statFields is required for PORTRAIT_STATS",)


def test_compatibility_distinguishes_unsupported_unknown_and_supported() -> None:
    unsupported = check_compatibility("MAIN", ResultType.COGNATES)
    unknown = check_compatibility("BIRCHBARK", ResultType.WORD_INFO)
    supported = check_compatibility("MAIN", ResultType.FREQUENCY)

    assert unsupported.status is CompatibilityStatus.UNSUPPORTED
    assert "не реализует" in unsupported.reason
    assert unknown.status is CompatibilityStatus.UNKNOWN
    assert "не доказывает" in unknown.reason
    assert supported.status is CompatibilityStatus.SUPPORTED

    overridden = check_compatibility(
        "BIRCHBARK",
        ResultType.WORD_INFO,
        overrides={
            ("BIRCHBARK", "PORTRAIT_WORD_INFO"): CompatibilityOverride(
                status=CompatibilityStatus.SUPPORTED,
                reason="Повторно проверено на трёх леммах.",
                source="manual-test",
            )
        },
    )
    assert overridden.status is CompatibilityStatus.SUPPORTED
    assert overridden.source == "override:manual-test"

    still_unavailable = check_compatibility(
        "MAIN",
        ResultType.COGNATES,
        overrides={
            ("MAIN", "PORTRAIT_COGNATES"): CompatibilityOverride(
                status=CompatibilityStatus.SUPPORTED,
                reason="Ошибочное локальное наблюдение.",
                source="manual-test",
            )
        },
    )
    assert still_unavailable.status is CompatibilityStatus.UNSUPPORTED
    assert still_unavailable.source == "registry"


def test_action_signature_is_canonical_and_order_independent() -> None:
    first = action_signature(
        "get_word_portrait",
        {"corpus": "MAIN", "resultType": ["B", "A"], "lemma": " мать "},
    )
    second = action_signature(
        "get_word_portrait",
        {"lemma": "мать", "resultType": ["A", "B"], "corpus": "MAIN"},
    )
    assert first == second


@pytest.mark.asyncio
async def test_budget_reservations_are_atomic_under_concurrency() -> None:
    tracker = BudgetTracker(
        BudgetLimits(
            max_branches=1,
            max_actions_per_branch=3,
            max_requests_per_iteration=2,
            max_external_calls=2,
            max_wall_time_seconds=10,
            max_evidence_items=2,
            max_context_chars=100,
        )
    )
    results = await asyncio.gather(
        tracker.reserve_action(),
        tracker.reserve_action(),
        tracker.reserve_action(),
        return_exceptions=True,
    )
    assert sum(result is None for result in results) == 2
    errors = [result for result in results if isinstance(result, BudgetExceeded)]
    assert len(errors) == 1
    assert errors[0].limit == "max_requests_per_iteration"
    assert tracker.external_calls == 2
    with pytest.raises(BudgetExceeded, match="branch"):
        await tracker.reserve_branch()
    await tracker.reserve_context(100)
    with pytest.raises(BudgetExceeded) as context_error:
        await tracker.reserve_context(1)
    assert context_error.value.limit == "max_context_chars"


def test_compressor_preserves_numbers_examples_metadata_and_source_hash() -> None:
    raw = {
        "frequencyData": {"ipm": 12.5, "count": 42},
        "concordanceData": {
            "groups": [
                {
                    "docs": [
                        {
                            "info": {
                                "title": "Пример",
                                "source": {"docId": "doc-1"},
                                "docExplainInfo": {"items": []},
                            },
                            "snippetGroups": [
                                {
                                    "snippets": [
                                        {
                                            "sequences": [
                                                {
                                                    "words": [
                                                        {"text": "Это "},
                                                        {
                                                            "text": "слово",
                                                            "displayParams": {"hit": True},
                                                        },
                                                    ]
                                                }
                                            ]
                                        }
                                    ]
                                }
                            ],
                        }
                    ]
                }
            ]
        },
    }
    compressed = compress_response(
        "get_word_portrait",
        raw,
        params={
            "corpus": "MAIN",
            "lemma": "слово",
            "resultType": ["PORTRAIT_FREQUENCY", "PORTRAIT_CONCORDANCE"],
        },
        limits=CompressionLimits(max_items_per_collection=10, max_examples=2),
    )

    assert compressed["raw_response_hash"] == raw_response_hash(raw)
    assert compressed["data"]["portrait_frequency"]["ipm"] == 12.5
    example = compressed["data"]["portrait_concordance"][0]
    assert example["doc_title"] == "Пример"
    assert example["doc_id"] == "doc-1"
    assert "**слово**" in example["text"]


def test_compressor_reports_parser_level_truncation() -> None:
    raw = {
        "similarData": [
            {
                "category": "all",
                "values": [
                    {"word": f"слово-{index}", "weight": index}
                    for index in range(12)
                ],
            }
        ]
    }
    compressed = compress_response(
        "get_word_portrait",
        raw,
        params={
            "corpus": "MAIN",
            "lemma": "слово",
            "resultType": ["PORTRAIT_SIMILAR"],
            "similarCategories": ["all"],
        },
        limits=CompressionLimits(max_items_per_collection=2),
    )
    assert compressed["truncated"] is True
    assert len(compressed["data"]["portrait_similar"][0]["words"]) == 2


def test_fact_extraction_never_rewrites_numeric_values() -> None:
    evidence = EvidenceArtifact(
        research_id="research-1",
        run_id="run-1",
        branch_id="branch-1",
        batch_id="batch-1",
        action_id="action-1",
        evidence_id="evidence-1",
        tool="get_word_portrait",
        source="NKRJA",
        status="success",
        params={"corpus": "MAIN", "lemma": "мать"},
        payload={"data": {"portrait_frequency": {"ipm": 7.25, "count": 12}}},
    )
    result = extract_facts(evidence)

    by_metric = {fact.metric: fact for fact in result.facts}
    assert by_metric["portrait_frequency.ipm"].value == 7.25
    assert by_metric["portrait_frequency.ipm"].unit == "ipm"
    assert by_metric["portrait_frequency.count"].value == 12
    assert all(fact.evidence_id == "evidence-1" for fact in result.facts)
    assert result.interpretations == ()


def test_concordance_examples_become_provenance_backed_facts() -> None:
    compressed = compress_response(
        "get_simple_concordance",
        {
            "examples": [
                {
                    "text": "Это проверяемый пример употребления.",
                    "doc_title": "Тестовый документ",
                    "doc_id": "doc-1",
                    "date": "2020",
                }
            ]
        },
        params={"corpus": "MAIN", "lemma": "пример"},
    )
    evidence = EvidenceArtifact(
        research_id="research-1",
        run_id="run-1",
        branch_id="branch-1",
        batch_id="batch-1",
        action_id="action-1",
        evidence_id="evidence-concordance",
        tool="get_simple_concordance",
        source="NKRJA",
        status="success",
        params={"corpus": "MAIN", "lemma": "пример"},
        payload=compressed,
    )

    result = extract_facts(evidence)
    example = next(
        fact for fact in result.facts if fact.metric.endswith("observation")
    )

    assert example.value["text"] == "Это проверяемый пример употребления."
    assert example.value["doc_id"] == "doc-1"
    assert example.evidence_id == "evidence-concordance"
    assert example.tool == "get_simple_concordance"


@pytest.mark.asyncio
async def test_client_obeys_retry_after_for_429() -> None:
    calls = 0
    delays: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, request=request)
        return httpx.Response(200, json={"ok": True}, request=request)

    async def sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
        client = NKRJAClient(_settings(), client=transport, sleep=sleep, random_value=lambda: 0)
        result = await client.get_corpus_stats("MAIN")

    assert result == {"ok": True}
    assert calls == 2
    assert delays == [2.0]


@pytest.mark.asyncio
async def test_client_bounds_5xx_retries() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
        client = NKRJAClient(
            _settings(HTTP_MAX_RETRIES="2"),
            client=transport,
            sleep=lambda _delay: asyncio.sleep(0),
            random_value=lambda: 0,
        )
        with pytest.raises(NKRJAResponseError) as raised:
            await client.get_corpus_stats("MAIN")

    assert raised.value.status_code == 503
    assert calls == 3


@pytest.mark.asyncio
async def test_client_returns_typed_timeout_after_bounded_retries() -> None:
    calls = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("slow", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as transport:
        client = NKRJAClient(
            _settings(HTTP_MAX_RETRIES="1"),
            client=transport,
            sleep=lambda _delay: asyncio.sleep(0),
            random_value=lambda: 0,
        )
        with pytest.raises(NKRJATimeoutError):
            await client.get_corpus_stats("MAIN")

    assert calls == 2


class _AsyncNKRJAFake:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def get_word_portrait(self, **params: Any) -> dict[str, Any]:
        self.calls.append(("get_word_portrait", copy.deepcopy(params)))
        return {"frequencyData": {"ipm": 3.5, "count": 9}}

    async def get_corpus_stats(self, **params: Any) -> dict[str, Any]:
        self.calls.append(("get_corpus_stats", copy.deepcopy(params)))
        if params["corpus"] == "MAIN":
            raise RuntimeError("simulated isolated failure")
        return {"documents": 10}


@pytest.mark.asyncio
async def test_orchestrator_preserves_warning_success_ids_and_input_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tools.orchestrator.emit_trace", lambda **_kwargs: None)
    client = _AsyncNKRJAFake()
    params = {
        "lemma": "мать",
        "corpus": "MAIN",
        "resultType": ["PORTRAIT_COGNATES", "PORTRAIT_FREQUENCY"],
    }
    original = copy.deepcopy(params)
    update = await api_orchestrator_node_async(
        _state({"action_id": "action-1", "tool": "get_word_portrait", "params": params}),
        client=client,
    )

    assert params == original
    assert client.calls[0][1]["resultType"] == ["PORTRAIT_FREQUENCY"]
    assert [item["status"] for item in update["last_evidence_batch"]] == [
        "warning",
        "success",
    ]
    assert {item["action_id"] for item in update["last_evidence_batch"]} == {
        "action-1"
    }
    assert update["completed_actions"][0]["status"] == "succeeded"
    assert {fact["value"] for fact in update["facts"]} == {3.5, 9}
    assert update["evidence"] == update["last_evidence_batch"]


@pytest.mark.asyncio
async def test_orchestrator_isolates_action_failure_and_blocks_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tools.orchestrator.emit_trace", lambda **_kwargs: None)
    client = _AsyncNKRJAFake()
    main = {"tool": "get_corpus_stats", "params": {"corpus": "MAIN"}}
    spoken = {"tool": "get_corpus_stats", "params": {"corpus": "SPOKEN"}}
    update = await api_orchestrator_node_async(
        _state(main, spoken, copy.deepcopy(spoken)),
        client=client,
    )

    assert len(client.calls) == 2
    statuses = sorted(item["status"] for item in update["completed_actions"])
    assert statuses == ["failed", "skipped", "succeeded"]
    assert len(update["last_evidence_batch"]) == 3
    assert any(item["source"] == "System_Safeguard" for item in update["evidence"])

    completed_spoken = next(
        item for item in update["completed_actions"] if item["status"] == "succeeded"
    )
    repeated = await api_orchestrator_node_async(
        _state(spoken, completed_actions=[completed_spoken]),
        client=client,
    )
    assert len(client.calls) == 2
    assert repeated["completed_actions"][0]["status"] == "skipped"


@pytest.mark.asyncio
async def test_unsupported_action_finishes_without_creating_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_client() -> Any:
        raise AssertionError("client must not be created for local unsupported action")

    monkeypatch.setattr("tools.orchestrator.NKRJAClient", forbidden_client)
    update = await api_orchestrator_node_async(
        _state(
            {
                "tool": "get_word_portrait",
                "params": {
                    "lemma": "мать",
                    "corpus": "MAIN",
                    "resultType": ["PORTRAIT_COGNATES"],
                },
            }
        )
    )
    assert update["completed_actions"][0]["status"] == "unsupported"
    assert any(item["status"] == "unsupported" for item in update["evidence"])


@pytest.mark.asyncio
async def test_evidence_budget_keeps_warning_and_success_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tools.orchestrator.emit_trace", lambda **_kwargs: None)
    client = _AsyncNKRJAFake()
    state = _state(
        {
            "action_id": "action-budget",
            "tool": "get_word_portrait",
            "params": {
                "lemma": "мать",
                "corpus": "MAIN",
                "resultType": ["PORTRAIT_COGNATES", "PORTRAIT_FREQUENCY"],
            },
        }
    )
    state["budgets"]["max_evidence_items"] = 1
    update = await api_orchestrator_node_async(state, client=client)

    assert update["completed_actions"][0]["status"] == "skipped"
    assert len(update["evidence"]) == 1
    assert update["evidence"][0]["source"] == "System_Budget"
    assert update["facts"] == []


@pytest.mark.asyncio
async def test_conflicting_action_id_is_rejected_before_external_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tools.orchestrator.emit_trace", lambda **_kwargs: None)
    client = _AsyncNKRJAFake()
    update = await api_orchestrator_node_async(
        _state(
            {
                "action_id": "action-conflict",
                "tool": "get_corpus_stats",
                "params": {"corpus": "MAIN"},
            },
            {
                "action_id": "action-conflict",
                "tool": "get_corpus_stats",
                "params": {"corpus": "SPOKEN"},
            },
        ),
        client=client,
    )
    assert client.calls == []
    assert len(update["completed_actions"]) == 1
    assert update["completed_actions"][0]["status"] == "failed"
    assert update["completed_actions"][0]["action_id"] != "action-conflict"
    assert "уже связан" in update["completed_actions"][0]["message"]


@pytest.mark.asyncio
async def test_historical_action_id_conflict_cannot_replace_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tools.orchestrator.emit_trace", lambda **_kwargs: None)
    previous = {
        "research_id": "research-1",
        "run_id": "run-1",
        "branch_id": "branch-1",
        "batch_id": "batch-old",
        "action_id": "action-historical",
        "tool": "get_corpus_stats",
        "params": {"corpus": "MAIN"},
        "status": "succeeded",
        "iteration": 0,
        "signature": action_signature(
            "get_corpus_stats", {"corpus": "MAIN"}
        ),
    }
    client = _AsyncNKRJAFake()
    update = await api_orchestrator_node_async(
        _state(
            {
                "action_id": "action-historical",
                "tool": "get_corpus_stats",
                "params": {"corpus": "SPOKEN"},
            },
            completed_actions=[previous],
        ),
        client=client,
    )
    assert client.calls == []
    diagnostic = update["completed_actions"][0]
    assert diagnostic["action_id"] != previous["action_id"]
    assert diagnostic["status"] == "failed"


@pytest.mark.asyncio
async def test_wall_time_and_total_calls_persist_between_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tools.orchestrator.emit_trace", lambda **_kwargs: None)
    client = _AsyncNKRJAFake()
    state = _state(
        {"tool": "get_corpus_stats", "params": {"corpus": "SPOKEN"}},
        research_started_at=time.time() - 20,
        budget_usage={
            "branches": 1,
            "actions": 1,
            "external_calls": 1,
            "evidence_items": 0,
            "context_chars": 0,
        },
    )
    state["budgets"]["max_wall_time_seconds"] = 10
    state["budgets"]["max_external_calls"] = 1
    update = await api_orchestrator_node_async(state, client=client)

    assert client.calls == []
    assert update["completed_actions"][0]["status"] == "skipped"
    assert update["budget_usage"]["external_calls"] == 1
