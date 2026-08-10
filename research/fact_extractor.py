"""Deterministic numeric and qualitative facts with evidence provenance."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from state.models import EvidenceArtifact, EvidenceStatus, Fact
from tools.registry import normalize_corpus


@dataclass(frozen=True, slots=True)
class FactExtractionResult:
    facts: tuple[Fact, ...]
    # Interpretations are intentionally a separate channel.  This deterministic
    # extractor never manufactures them; later LLM nodes may add sourced text.
    interpretations: tuple[str, ...] = ()


_UNITS_BY_NAME = {
    "ipm": "ipm",
    "frequency": "occurrences",
    "freq": "occurrences",
    "count": "occurrences",
    "totalcount": "occurrences",
    "total_count": "occurrences",
    "total_count_in_corpus": "occurrences",
    "texts": "texts",
    "documents": "documents",
    "sentences": "sentences",
    "tokens": "tokens",
    "words": "tokens",
    "year": "year",
    "percentage": "percent",
    "percent": "percent",
}


def _walk_numbers(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[str, int | float]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield ".".join(path), value
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            yield from _walk_numbers(value[key], (*path, str(key)))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_numbers(item, (*path, str(index)))


_GROUPED_TEXT_KEYS = ("text", "word", "form")
def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only JSON-compatible values from a bounded compressed record."""

    result: dict[str, Any] = {}
    for key in sorted(value, key=str):
        item = value[key]
        if item is None or isinstance(item, (str, int, float, bool)):
            result[str(key)] = item
        elif isinstance(item, Mapping):
            result[str(key)] = _json_mapping(item)
        elif isinstance(item, (list, tuple)):
            result[str(key)] = [
                _json_mapping(child) if isinstance(child, Mapping) else child
                for child in item
                if child is None or isinstance(child, (str, int, float, bool, Mapping))
            ]
    return result


def _walk_text_observations(
    value: Any,
    path: tuple[str, ...] = (),
) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        grouped_key = next(
            (
                key
                for key in _GROUPED_TEXT_KEYS
                if isinstance(value.get(key), str) and value.get(key).strip()
            ),
            None,
        )
        if grouped_key is not None:
            metric = ".".join(path) or grouped_key
            yield f"{metric}.observation", _json_mapping(value)
            return
        for key in sorted(value, key=str):
            item = value[key]
            if isinstance(item, str) and item.strip():
                yield ".".join((*path, str(key))), item
            else:
                yield from _walk_text_observations(item, (*path, str(key)))
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_text_observations(item, (*path, str(index)))
        return
    if isinstance(value, str) and value.strip() and path:
        yield ".".join(path), value


def _unit_for(metric: str) -> str | None:
    leaf = metric.rsplit(".", 1)[-1].lower()
    return _UNITS_BY_NAME.get(leaf)


def _fact_id(evidence_id: str, metric: str) -> str:
    digest = hashlib.sha256(f"{evidence_id}:{metric}".encode("utf-8")).hexdigest()
    return f"fact-{digest}"


def extract_facts(
    artifact: EvidenceArtifact | Mapping[str, Any],
    *,
    max_facts: int = 200,
) -> FactExtractionResult:
    """Extract observable values without asking an LLM to rewrite them.

    Qualitative records are limited to bounded compressor fields such as
    concordance examples, collocates and word forms.  They remain ordinary
    provenance-backed ``Fact`` objects rather than free-form LLM deductions.
    """

    evidence = (
        artifact
        if isinstance(artifact, EvidenceArtifact)
        else EvidenceArtifact.model_validate(artifact)
    )
    if evidence.status is not EvidenceStatus.SUCCESS or max_facts < 1:
        return FactExtractionResult(())

    corpus_value = evidence.params.get("corpus")
    if not isinstance(corpus_value, str):
        return FactExtractionResult(())
    corpus = normalize_corpus(corpus_value).value
    lemma_value = evidence.params.get("lemma")
    lemma = lemma_value.strip() if isinstance(lemma_value, str) and lemma_value.strip() else None

    payload = evidence.payload
    if isinstance(payload, Mapping) and "data" in payload:
        payload = payload["data"]

    facts: list[Fact] = []
    qualitative_limit = min(max_facts, 50)
    for metric, value in _walk_text_observations(payload):
        if not metric:
            continue
        facts.append(
            Fact(
                research_id=evidence.research_id,
                run_id=evidence.run_id,
                branch_id=evidence.branch_id,
                batch_id=evidence.batch_id,
                action_id=evidence.action_id,
                evidence_id=evidence.evidence_id,
                fact_id=_fact_id(evidence.evidence_id, metric),
                corpus=corpus,
                lemma=lemma,
                metric=metric,
                value=value,
                unit=None,
                tool=evidence.tool,
            )
        )
        if len(facts) >= qualitative_limit:
            break

    for metric, value in _walk_numbers(payload):
        if not metric:
            continue
        facts.append(
            Fact(
                research_id=evidence.research_id,
                run_id=evidence.run_id,
                branch_id=evidence.branch_id,
                batch_id=evidence.batch_id,
                action_id=evidence.action_id,
                evidence_id=evidence.evidence_id,
                fact_id=_fact_id(evidence.evidence_id, metric),
                corpus=corpus,
                lemma=lemma,
                metric=metric,
                value=value,
                unit=_unit_for(metric),
                tool=evidence.tool,
            )
        )
        if len(facts) >= max_facts:
            break
    return FactExtractionResult(tuple(facts))


__all__ = ["FactExtractionResult", "extract_facts"]
