"""Strict, provider-independent contracts for every LLM decision.

No caller extracts a JSON-looking substring or silently coerces values.  A raw
response is validated with Pydantic's ``model_validate_json``.  One invalid
response may be repaired once; a second invalid response becomes a typed safe
failure that graph nodes can route to a terminal outcome.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypeVar

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    ValidationError,
    model_validator,
)

from state.models import (
    AggregatorStatus,
    ErrorKind,
    ExecutionStatus,
    ResearchMode,
)


class _LLMContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        allow_inf_nan=False,
    )


ZeroBasedIndex = Annotated[StrictInt, Field(ge=0)]
OneBasedStep = Annotated[StrictInt, Field(ge=1)]


class ClassificationOutput(_LLMContract):
    mode: Literal["research", "chat"]
    reasoning: str = Field(min_length=1)


class PlannerSubGoalOutput(_LLMContract):
    goal: str = Field(min_length=1)
    corpus: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    corpus_reasoning: str = Field(min_length=1)
    lemma: str | None = None
    hypotheses: tuple[str, ...] = ()
    research_plan: tuple[str, ...] = ()


class PlannerOutput(_LLMContract):
    mode: ResearchMode
    reasoning: str = Field(min_length=1)
    goal: str | None = None
    sub_goals: tuple[PlannerSubGoalOutput, ...] = ()
    error_kind: ErrorKind | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> "PlannerOutput":
        if self.mode in {ResearchMode.RESEARCH, ResearchMode.REPLAN}:
            if not self.goal or not self.sub_goals:
                raise ValueError("research/replan requires goal and sub_goals")
            if self.error_kind is not None or self.error_message is not None:
                raise ValueError("successful planning cannot contain error details")
        elif self.mode is ResearchMode.CHAT:
            if self.sub_goals:
                raise ValueError("chat cannot contain sub_goals")
            if self.error_kind is not None or self.error_message is not None:
                raise ValueError("chat cannot contain error details")
        else:
            if self.error_kind is None or not self.error_message:
                raise ValueError("error mode requires safe error details")
            if self.sub_goals:
                raise ValueError("error mode cannot contain sub_goals")
        return self


class ExecutionActionOutput(_LLMContract):
    tool: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    params: dict[str, JsonValue] = Field(default_factory=dict)
    depends_on: tuple[ZeroBasedIndex, ...] = ()

    @model_validator(mode="after")
    def validate_dependencies(self) -> "ExecutionActionOutput":
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("depends_on indices must be unique")
        if any(index < 0 for index in self.depends_on):
            raise ValueError("depends_on indices must be non-negative")
        return self


class ExecutionOutput(_LLMContract):
    status: ExecutionStatus
    reasoning: str = Field(min_length=1)
    actions: tuple[ExecutionActionOutput, ...] = ()
    error_kind: ErrorKind | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> "ExecutionOutput":
        if self.status is ExecutionStatus.EXECUTE:
            if not self.actions:
                raise ValueError("execute requires at least one action")
            if self.error_kind is not None or self.error_message is not None:
                raise ValueError("execute cannot contain error details")
        elif self.status is ExecutionStatus.COMPLETE:
            if self.actions:
                raise ValueError("complete cannot contain actions")
            if self.error_kind is not None or self.error_message is not None:
                raise ValueError("complete cannot contain error details")
        else:
            if self.actions:
                raise ValueError("error cannot contain actions")
            if self.error_kind is None or not self.error_message:
                raise ValueError("error requires safe error details")
        return self


class AggregatorOutput(_LLMContract):
    status: AggregatorStatus
    reasoning: str = Field(min_length=1)
    progress_made: StrictBool
    covered_plan_steps: tuple[OneBasedStep, ...] = ()
    missing_plan_steps: tuple[OneBasedStep, ...] = ()
    missing_information: tuple[str, ...] = ()
    deductions: tuple[str, ...] = ()
    error_kind: ErrorKind | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> "AggregatorOutput":
        if len(self.covered_plan_steps) != len(set(self.covered_plan_steps)):
            raise ValueError("covered_plan_steps must be unique")
        if len(self.missing_plan_steps) != len(set(self.missing_plan_steps)):
            raise ValueError("missing_plan_steps must be unique")
        if set(self.covered_plan_steps) & set(self.missing_plan_steps):
            raise ValueError("covered and missing plan steps cannot overlap")
        if self.status is AggregatorStatus.ERROR:
            if self.error_kind is None or not self.error_message:
                raise ValueError("error requires safe error details")
        elif self.error_kind is not None or self.error_message is not None:
            raise ValueError("non-error aggregation cannot contain error details")
        if self.status is AggregatorStatus.COMPLETE and self.missing_plan_steps:
            raise ValueError("complete cannot leave missing plan steps")
        return self


@dataclass(frozen=True, slots=True)
class StructuredTerminalResult:
    kind: ErrorKind
    message: str
    node: str
    retryable: bool
    attempts: int

    def as_safe_error(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "message": self.message,
            "node": self.node,
            "retryable": self.retryable,
        }


class StructuredOutputError(RuntimeError):
    """Raised after a model error or two invalid structured responses."""

    def __init__(self, result: StructuredTerminalResult) -> None:
        super().__init__(result.message)
        self.result = result


ContractT = TypeVar("ContractT", bound=BaseModel)


def _response_content(response: Any) -> str | BaseModel:
    content = getattr(response, "content", response)
    if isinstance(content, BaseModel):
        return content
    if isinstance(content, str):
        return content
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    raise TypeError("model response content must be JSON text or a Pydantic model")


def _validation_summary(error: Exception) -> str:
    if isinstance(error, ValidationError):
        compact = []
        for item in error.errors(include_url=False)[:8]:
            location = ".".join(str(part) for part in item.get("loc", ())) or "response"
            compact.append(f"{location}: {item.get('msg', 'invalid value')}")
        return "; ".join(compact)
    return str(error)[:800]


async def ainvoke_structured(
    llm: Any,
    messages: list[Any],
    schema: type[ContractT],
    *,
    node: str,
    config: Any = None,
) -> ContractT:
    """Invoke a model and strictly validate JSON, with exactly one repair."""

    current_messages = list(messages)
    for attempt in (1, 2):
        try:
            response = await llm.ainvoke(current_messages, config=config)
        except Exception as error:
            raise StructuredOutputError(
                StructuredTerminalResult(
                    kind=ErrorKind.MODEL,
                    message="Не удалось получить ответ модели.",
                    node=node,
                    retryable=True,
                    attempts=attempt,
                )
            ) from error

        content = _response_content(response)
        try:
            if isinstance(content, schema):
                return content
            if isinstance(content, BaseModel):
                content = content.model_dump_json()
            return schema.model_validate_json(content)
        except (ValidationError, ValueError, TypeError) as error:
            if attempt == 2:
                raise StructuredOutputError(
                    StructuredTerminalResult(
                        kind=ErrorKind.VALIDATION,
                        message="Модель дважды вернула ответ, не соответствующий контракту.",
                        node=node,
                        retryable=False,
                        attempts=2,
                    )
                ) from error

            invalid_text = (
                content.model_dump_json()
                if isinstance(content, BaseModel)
                else str(content)
            )
            repair = (
                "Исправь предыдущий ответ. Верни только один JSON-объект без "
                "markdown и комментариев, строго по JSON Schema ниже. Не добавляй "
                "поля и не превращай boolean в строки.\n\n"
                f"Ошибки валидации: {_validation_summary(error)}\n\n"
                f"JSON Schema: {json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
            )
            current_messages = [
                *messages,
                AIMessage(content=invalid_text[:12_000]),
                HumanMessage(content=repair),
            ]

    raise AssertionError("unreachable")


__all__ = [
    "AggregatorOutput",
    "ClassificationOutput",
    "ExecutionActionOutput",
    "ExecutionOutput",
    "PlannerOutput",
    "PlannerSubGoalOutput",
    "StructuredOutputError",
    "StructuredTerminalResult",
    "ainvoke_structured",
]
