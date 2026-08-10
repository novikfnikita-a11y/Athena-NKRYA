"""Deterministic numeric fact extraction with evidence provenance."""

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
    """Extract numeric observations without asking an LLM to rewrite values."""

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
