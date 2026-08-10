"""Three-state compatibility decisions for corpus/result-type pairs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Mapping

from tools.registry import RESULT_TYPE_AVAILABILITY, Corpus, ResultType, normalize_corpus
from whitelist_generated import (
    RESULTTYPE_CORPUS_EMPTY,
    RESULTTYPE_CORPUS_ERRORS,
    RESULTTYPE_CORPUS_WHITELIST,
)


class CompatibilityStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CompatibilityDecision:
    corpus: Corpus
    result_type: ResultType
    status: CompatibilityStatus
    reason: str
    source: str


CompatibilityKey = tuple[Corpus, ResultType]


@dataclass(frozen=True, slots=True)
class CompatibilityOverride:
    status: CompatibilityStatus
    reason: str
    source: str
    verified_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("compatibility override requires a reason")
        if not self.source.strip():
            raise ValueError("compatibility override requires a source")


def _normalized_overrides(
    overrides: Mapping[tuple[str, str], CompatibilityOverride] | None,
) -> dict[CompatibilityKey, CompatibilityOverride]:
    result: dict[CompatibilityKey, CompatibilityOverride] = {}
    for (corpus, result_type), override in (overrides or {}).items():
        if not isinstance(override, CompatibilityOverride):
            raise TypeError("compatibility overrides must be CompatibilityOverride values")
        result[(normalize_corpus(corpus), ResultType(result_type))] = override
    return result


def check_compatibility(
    corpus: str | Corpus,
    result_type: str | ResultType,
    *,
    overrides: Mapping[tuple[str, str], CompatibilityOverride] | None = None,
) -> CompatibilityDecision:
    """Classify a pair without treating empty/error probe results as proof."""

    corpus_value = normalize_corpus(corpus)
    type_value = (
        result_type
        if isinstance(result_type, ResultType)
        else ResultType(str(result_type).strip().upper())
    )
    available, unavailable_reason = RESULT_TYPE_AVAILABILITY[type_value]
    if not available:
        return CompatibilityDecision(
            corpus_value,
            type_value,
            CompatibilityStatus.UNSUPPORTED,
            unavailable_reason or "Тип результата недоступен.",
            "registry",
        )

    override = _normalized_overrides(overrides).get((corpus_value, type_value))
    if override is not None:
        return CompatibilityDecision(
            corpus_value,
            type_value,
            override.status,
            override.reason,
            f"override:{override.source}",
        )

    supported = RESULTTYPE_CORPUS_WHITELIST.get(type_value.value, ())
    if corpus_value.value in supported:
        return CompatibilityDecision(
            corpus_value,
            type_value,
            CompatibilityStatus.SUPPORTED,
            "Комбинация подтверждена успешным воспроизводимым наблюдением.",
            "probe",
        )

    empty = RESULTTYPE_CORPUS_EMPTY.get(type_value.value, ())
    if corpus_value.value in empty:
        reason = "Пробный ответ был пустым; это не доказывает несовместимость."
    elif corpus_value.value in RESULTTYPE_CORPUS_ERRORS.get(type_value.value, ()):
        reason = "Пробный вызов завершился ошибкой; причина совместимости не установлена."
    else:
        reason = "Комбинация ещё не проверена полным воспроизводимым прогоном."
    return CompatibilityDecision(
        corpus_value,
        type_value,
        CompatibilityStatus.UNKNOWN,
        reason,
        "probe",
    )


__all__ = [
    "CompatibilityDecision",
    "CompatibilityOverride",
    "CompatibilityStatus",
    "check_compatibility",
]
