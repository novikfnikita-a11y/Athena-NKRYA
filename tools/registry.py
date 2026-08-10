"""Canonical registry for NKRJA corpora, result types and tool contracts.

Nothing outside this module should maintain its own spelling of a corpus or
its own list of tool parameters.  Runtime compatibility observations live in
``tools.compatibility`` and are deliberately kept separate from capability
declarations: an incomplete probe is not an API contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class Corpus(str, Enum):
    MAIN = "MAIN"
    SYNTAX = "SYNTAX"
    PAPER = "PAPER"
    REGIONAL = "REGIONAL"
    PARA = "PARA"
    MULTI = "MULTI"
    SCHOOL = "SCHOOL"
    DIALECT = "DIALECT"
    POETIC = "POETIC"
    SPOKEN = "SPOKEN"
    ACCENT = "ACCENT"
    MURCO = "MURCO"
    MULTIPARC_RUS = "MULTIPARC_RUS"
    MULTIPARC = "MULTIPARC"
    OLD_RUS = "OLD_RUS"
    BIRCHBARK = "BIRCHBARK"
    MID_RUS = "MID_RUS"
    ORTHLIB = "ORTHLIB"
    PANCHRON = "PANCHRON"
    KIDS = "KIDS"
    CLASSICS = "CLASSICS"
    BLOGS = "BLOGS"
    EPIGRAPHICA = "EPIGRAPHICA"
    GICR = "GICR"


class ResultType(str, Enum):
    WORD_INFO = "PORTRAIT_WORD_INFO"
    CONCORDANCE = "PORTRAIT_CONCORDANCE"
    STATS = "PORTRAIT_STATS"
    SKETCH = "PORTRAIT_SKETCH"
    FREQUENCY = "PORTRAIT_FREQUENCY"
    SIMILAR = "PORTRAIT_SIMILAR"
    MORPHEME = "PORTRAIT_MORPHEME"
    WORDFORMS = "PORTRAIT_WORDFORMS"
    COGNATES = "PORTRAIT_COGNATES"
    FIRST_MENTION = "PORTRAIT_FIRST_MENTION"
    MEANING = "PORTRAIT_MEANING"


@dataclass(frozen=True, slots=True)
class ConditionalParameter:
    name: str
    required_for_result_types: frozenset[ResultType]
    description: str


@dataclass(frozen=True, slots=True)
class ToolCapability:
    name: str
    description: str
    required_params: tuple[str, ...]
    optional_params: tuple[str, ...] = ()
    conditional_params: tuple[ConditionalParameter, ...] = ()

    def validate_params(self, params: Mapping[str, Any]) -> tuple[str, ...]:
        """Return all contract violations without mutating caller parameters."""

        errors = [
            f"missing required parameter: {name}"
            for name in self.required_params
            if name not in params or params[name] in (None, "", [])
        ]
        requested = {
            ResultType(value)
            for value in params.get("resultType", [])
            if value in RESULT_TYPE_VALUES
        }
        for condition in self.conditional_params:
            if requested & condition.required_for_result_types:
                if params.get(condition.name) in (None, "", []):
                    errors.append(
                        f"{condition.name} is required for "
                        + ", ".join(
                            sorted(
                                item.value
                                for item in requested
                                & condition.required_for_result_types
                            )
                        )
                    )
        allowed = {
            *self.required_params,
            *self.optional_params,
            *(item.name for item in self.conditional_params),
        }
        errors.extend(
            f"unknown parameter: {name}" for name in params if name not in allowed
        )
        return tuple(errors)


CORPUS_ALIASES: Mapping[str, Corpus] = MappingProxyType(
    {
        "ОСНОВНОЙ": Corpus.MAIN,
        "MAIN_CORPUS": Corpus.MAIN,
        "ГАЗЕТНЫЙ": Corpus.PAPER,
        "NEWSPAPER": Corpus.PAPER,
        "ОБУЧАЮЩИЙ": Corpus.SCHOOL,
        "EDUCATIONAL": Corpus.SCHOOL,
        "ШКОЛЬНЫЙ": Corpus.SCHOOL,
        "МУЛЬТИМЕДИЙНЫЙ": Corpus.MULTI,
        "MULTIMEDIA": Corpus.MULTI,
        "УСТНЫЙ": Corpus.SPOKEN,
        "ПОЭТИЧЕСКИЙ": Corpus.POETIC,
    }
)


RESULT_TYPE_AVAILABILITY: Mapping[ResultType, tuple[bool, str | None]] = (
    MappingProxyType(
        {
            result_type: (
                False,
                "Тип объявлен API, но текущий backend НКРЯ его не реализует.",
            )
            if result_type in {ResultType.COGNATES, ResultType.MEANING}
            else (True, None)
            for result_type in ResultType
        }
    )
)


_WORD_PORTRAIT_CONDITIONAL = (
    ConditionalParameter(
        name="statFields",
        required_for_result_types=frozenset({ResultType.STATS}),
        description="Поля группировки для PORTRAIT_STATS.",
    ),
    ConditionalParameter(
        name="similarCategories",
        required_for_result_types=frozenset({ResultType.SIMILAR}),
        description="Категории для PORTRAIT_SIMILAR; ['all'] без группировки.",
    ),
)


TOOL_CAPABILITIES: Mapping[str, ToolCapability] = MappingProxyType(
    {
        "get_word_portrait": ToolCapability(
            name="get_word_portrait",
            description="Комплексный анализ одной леммы в одном корпусе.",
            required_params=("lemma", "corpus", "resultType"),
            optional_params=("pos", "seed"),
            conditional_params=_WORD_PORTRAIT_CONDITIONAL,
        ),
        "get_corpus_stats": ToolCapability(
            name="get_corpus_stats",
            description="Общая статистика корпуса.",
            required_params=("corpus",),
        ),
        "get_sketch_difference": ToolCapability(
            name="get_sketch_difference",
            description="Сравнение синтаксических контекстов двух лемм.",
            required_params=("lemma_1", "lemma_2", "corpus", "pos"),
        ),
        "get_simple_concordance": ToolCapability(
            name="get_simple_concordance",
            description="Конкорданс одной леммы.",
            required_params=("lemma", "corpus"),
        ),
        "get_corpus_config": ToolCapability(
            name="get_corpus_config",
            description="Конфигурация корпуса.",
            required_params=("corpus",),
        ),
        "get_corpus_attributes": ToolCapability(
            name="get_corpus_attributes",
            description="Доступные метаатрибуты корпуса.",
            required_params=("corpus",),
        ),
        "get_attribute_values": ToolCapability(
            name="get_attribute_values",
            description="Значения одного метаатрибута корпуса.",
            required_params=("attr_name", "corpus"),
        ),
        "get_lex_gramm_search_form": ToolCapability(
            name="get_lex_gramm_search_form",
            description="Описание формы лексико-грамматического поиска.",
            required_params=("corpus",),
        ),
        "check_auth": ToolCapability(
            name="check_auth",
            description="Проверка авторизации НКРЯ.",
            required_params=(),
        ),
    }
)

# Transitional dictionary shape consumed by the current prompts/planners.
CAPABILITY_REGISTRY = MappingProxyType(
    {
        name: {
            "description": capability.description,
            "requires_params": list(capability.required_params),
            "optional_params": [
                *capability.optional_params,
                *(item.name for item in capability.conditional_params),
            ],
            "conditional_params": {
                item.name: sorted(
                    result_type.value
                    for result_type in item.required_for_result_types
                )
                for item in capability.conditional_params
            },
        }
        for name, capability in TOOL_CAPABILITIES.items()
    }
)

RESULT_TYPE_VALUES = frozenset(item.value for item in ResultType)
CORPUS_VALUES = frozenset(item.value for item in Corpus)
CORPUS_TYPE_ENUM_DESCRIPTION = (
    "СПРАВОЧНИК ДОСТУПНЫХ КОРПУСОВ (CorpusTypeEnum):\n"
    "Передавать строго одно из значений: "
    + ", ".join(item.value for item in Corpus)
)


def normalize_corpus(value: str | Corpus | None) -> Corpus:
    normalized = str(value.value if isinstance(value, Corpus) else value or "MAIN")
    normalized = normalized.strip().upper()
    normalized = CORPUS_ALIASES.get(normalized, normalized)
    try:
        return normalized if isinstance(normalized, Corpus) else Corpus(normalized)
    except ValueError as error:
        raise ValueError(f"unknown NKRJA corpus: {value!r}") from error


def get_registry_description() -> str:
    corpus_lines = ", ".join(item.value for item in Corpus)
    result_lines = []
    for result_type in ResultType:
        available, reason = RESULT_TYPE_AVAILABILITY[result_type]
        suffix = "доступен" if available else f"недоступен: {reason}"
        result_lines.append(f"- {result_type.value}: {suffix}")

    tool_lines = []
    for capability in TOOL_CAPABILITIES.values():
        required = ", ".join(capability.required_params) or "нет"
        optional = ", ".join(capability.optional_params) or "нет"
        conditions = "; ".join(
            f"{item.name} для "
            + ", ".join(sorted(value.value for value in item.required_for_result_types))
            for item in capability.conditional_params
        ) or "нет"
        tool_lines.append(
            f"Инструмент {capability.name}: {capability.description}\n"
            f"Обязательные: {required}. Опциональные: {optional}. "
            f"Условные: {conditions}."
        )

    return (
        "КОРПУСЫ НКРЯ:\n"
        + corpus_lines
        + "\n\nТИПЫ WORD PORTRAIT:\n"
        + "\n".join(result_lines)
        + "\n\nИНСТРУМЕНТЫ:\n"
        + "\n\n".join(tool_lines)
    )


__all__ = [
    "CAPABILITY_REGISTRY",
    "CORPUS_ALIASES",
    "CORPUS_TYPE_ENUM_DESCRIPTION",
    "CORPUS_VALUES",
    "RESULT_TYPE_AVAILABILITY",
    "RESULT_TYPE_VALUES",
    "TOOL_CAPABILITIES",
    "ConditionalParameter",
    "Corpus",
    "ResultType",
    "ToolCapability",
    "get_registry_description",
    "normalize_corpus",
]
