"""Typed domain contracts for the Athena-NKRYA research lifecycle.

The models in this module are deliberately independent from LangGraph, the LLM
provider, and the NKRJA client.  Identifiers are required inputs rather than
generated defaults: the lifecycle layer owns ID creation, while these contracts
only validate that the same IDs are propagated through nested records.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StringConstraints,
    model_validator,
)


StableIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^\S+$",
    ),
]
NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
CorpusName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Z][A-Z0-9_]*$",
    ),
]
ToolName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]
NonNegativeInt = Annotated[int, Field(ge=0)]
JsonObject = dict[str, JsonValue]


class ResearchMode(str, Enum):
    """Planner decisions before routing the top-level graph."""

    RESEARCH = "research"
    CHAT = "chat"
    REPLAN = "replan"
    ERROR = "error"


# ``PlannerMode`` remains an explicit domain name for call sites that deal with
# planner output, while state/schema.py can use the more general name.
PlannerMode = ResearchMode


class ResearchStatus(str, Enum):
    """Lifecycle status of one research task, independent from its route."""

    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    PARTIAL = "partial"
    ERROR = "error"
    CANCELLED = "cancelled"


class PlannerRoute(str, Enum):
    """Static top-level routes available after planning."""

    RESEARCH = "research"
    CHAT = "chat"
    ERROR = "error"


class ExecutionStatus(str, Enum):
    """Explicit outcome of execution planning."""

    EXECUTE = "execute"
    COMPLETE = "complete"
    ERROR = "error"


class ActionStatus(str, Enum):
    """Lifecycle status of one deterministic tool action."""

    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"


class EvidenceStatus(str, Enum):
    """Outcome represented by an evidence artifact."""

    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    UNSUPPORTED = "unsupported"


class AggregatorStatus(str, Enum):
    """Explicit decision made after analysing a non-empty evidence batch."""

    COMPLETE = "complete"
    CONTINUE = "continue"
    REPLAN = "replan"
    ERROR = "error"


class BranchRoute(str, Enum):
    """Static routes available inside one corpus branch."""

    EXECUTE = "execute"
    COMPLETE = "complete"
    REPLAN = "replan"
    ERROR = "error"


class BranchStatus(str, Enum):
    """Terminal outcome of one isolated corpus branch."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    ERROR = "error"


class TerminationReason(str, Enum):
    """Machine-readable reason why research or a branch stopped."""

    GOAL_REACHED = "goal_reached"
    PLAN_COMPLETE = "plan_complete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_PROGRESS = "no_progress"
    NO_ACTIONS = "no_actions"
    ERROR = "error"
    CANCELLED = "cancelled"


class ErrorKind(str, Enum):
    """Safe error categories shared by state and the terminal error node."""

    MODEL = "model"
    API = "api"
    VALIDATION = "validation"
    BUDGET = "budget"
    CONFIGURATION = "configuration"
    INTERNAL = "internal"
    CANCELLED = "cancelled"


class _Contract(BaseModel):
    """Strict immutable base for records exchanged between graph nodes."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
        allow_inf_nan=False,
    )


class _ResearchScope(_Contract):
    research_id: StableIdentifier
    run_id: StableIdentifier


class _BranchScope(_ResearchScope):
    branch_id: StableIdentifier


class _BatchScope(_BranchScope):
    batch_id: StableIdentifier


class _ActionScope(_BatchScope):
    action_id: StableIdentifier


class _EvidenceScope(_ActionScope):
    evidence_id: StableIdentifier


def _ensure_unique(field_name: str, values: tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique identifiers")


class ResearchSubGoal(_BranchScope):
    """One corpus-bound unit of research dispatched to an isolated branch."""

    goal: NonEmptyText
    corpus: CorpusName
    corpus_reasoning: NonEmptyText
    lemma: NonEmptyText | None = None
    hypotheses: tuple[NonEmptyText, ...] = ()
    research_plan: tuple[NonEmptyText, ...] = ()


class PlannerDecision(_ResearchScope):
    """Validated planner decision for a new turn or a replan."""

    mode: PlannerMode
    reasoning: NonEmptyText
    goal: NonEmptyText | None = None
    sub_goals: tuple[ResearchSubGoal, ...] = ()
    error_kind: ErrorKind | None = None
    error_message: NonEmptyText | None = None

    @property
    def route(self) -> PlannerRoute:
        if self.mode is PlannerMode.CHAT:
            return PlannerRoute.CHAT
        if self.mode is PlannerMode.ERROR:
            return PlannerRoute.ERROR
        return PlannerRoute.RESEARCH

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        branch_ids = tuple(sub_goal.branch_id for sub_goal in self.sub_goals)
        _ensure_unique("sub_goals.branch_id", branch_ids)

        for sub_goal in self.sub_goals:
            if sub_goal.research_id != self.research_id:
                raise ValueError(
                    "every sub-goal must use the planner research_id"
                )
            if sub_goal.run_id != self.run_id:
                raise ValueError("every sub-goal must use the planner run_id")

        if self.mode in {PlannerMode.RESEARCH, PlannerMode.REPLAN}:
            if self.goal is None:
                raise ValueError("research and replan decisions require goal")
            if not self.sub_goals:
                raise ValueError(
                    "research and replan decisions require at least one sub-goal"
                )
            if self.error_kind is not None or self.error_message is not None:
                raise ValueError(
                    "successful planner decisions cannot contain error details"
                )
        elif self.mode is PlannerMode.CHAT:
            if self.sub_goals:
                raise ValueError("chat decisions cannot dispatch research sub-goals")
            if self.error_kind is not None or self.error_message is not None:
                raise ValueError("chat decisions cannot contain error details")
        else:
            if self.error_kind is None or self.error_message is None:
                raise ValueError(
                    "error decisions require error_kind and error_message"
                )
            if self.sub_goals:
                raise ValueError("error decisions cannot dispatch sub-goals")

        return self


class ActionCall(_ActionScope):
    """One immutable, traceable invocation planned for a tool."""

    tool: ToolName
    params: JsonObject = Field(default_factory=dict)
    status: ActionStatus = ActionStatus.PLANNED
    revision: NonNegativeInt = 0
    iteration: NonNegativeInt = 0
    depends_on_action_ids: tuple[StableIdentifier, ...] = ()
    signature: NonEmptyText | None = None
    message: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_action(self) -> Self:
        _ensure_unique("depends_on_action_ids", self.depends_on_action_ids)
        if self.action_id in self.depends_on_action_ids:
            raise ValueError("an action cannot depend on itself")
        if self.status in {
            ActionStatus.FAILED,
            ActionStatus.SKIPPED,
            ActionStatus.UNSUPPORTED,
        } and self.message is None:
            raise ValueError(f"{self.status.value} actions require a message")
        return self


class ExecutionPlan(_BatchScope):
    """Explicit execution decision for one branch iteration."""

    status: ExecutionStatus
    reasoning: NonEmptyText
    iteration: NonNegativeInt
    actions: tuple[ActionCall, ...] = ()
    error_kind: ErrorKind | None = None
    error_message: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        action_ids = tuple(action.action_id for action in self.actions)
        _ensure_unique("actions.action_id", action_ids)

        for action in self.actions:
            expected_scope = (
                self.research_id,
                self.run_id,
                self.branch_id,
                self.batch_id,
            )
            action_scope = (
                action.research_id,
                action.run_id,
                action.branch_id,
                action.batch_id,
            )
            if action_scope != expected_scope:
                raise ValueError(
                    "every action must use the execution plan scope identifiers"
                )
            if action.iteration != self.iteration:
                raise ValueError(
                    "every action must use the execution plan iteration"
                )
            if action.status is not ActionStatus.PLANNED:
                raise ValueError("execution plans may contain only planned actions")

        if self.status is ExecutionStatus.EXECUTE:
            if not self.actions:
                raise ValueError("execute status requires at least one action")
            if self.error_kind is not None or self.error_message is not None:
                raise ValueError("execute status cannot contain error details")
        elif self.status is ExecutionStatus.COMPLETE:
            if self.actions:
                raise ValueError("complete status cannot contain actions")
            if self.error_kind is not None or self.error_message is not None:
                raise ValueError("complete status cannot contain error details")
        else:
            if self.actions:
                raise ValueError("error status cannot contain actions")
            if self.error_kind is None or self.error_message is None:
                raise ValueError("error status requires error_kind and error_message")

        return self


class EvidenceArtifact(_EvidenceScope):
    """One result, warning, error, or API limitation tied to an action."""

    tool: ToolName
    source: NonEmptyText
    status: EvidenceStatus
    revision: NonNegativeInt = 0
    params: JsonObject = Field(default_factory=dict)
    payload: JsonValue | None = None
    message: NonEmptyText | None = None
    raw_response_hash: StableIdentifier | None = None

    @model_validator(mode="after")
    def validate_artifact(self) -> Self:
        if self.status in {
            EvidenceStatus.WARNING,
            EvidenceStatus.ERROR,
            EvidenceStatus.UNSUPPORTED,
        } and self.message is None:
            raise ValueError(f"{self.status.value} evidence requires a message")
        return self


class Fact(_EvidenceScope):
    """A typed observation with direct provenance to one evidence artifact."""

    fact_id: StableIdentifier
    revision: NonNegativeInt = 0
    corpus: CorpusName
    metric: NonEmptyText
    value: JsonValue
    tool: ToolName
    lemma: NonEmptyText | None = None
    unit: NonEmptyText | None = None


class AggregatorDecision(_BatchScope):
    """Validated branch decision based only on the current evidence batch."""

    status: AggregatorStatus
    reasoning: NonEmptyText
    iteration: NonNegativeInt
    progress_made: StrictBool
    evidence_ids: tuple[StableIdentifier, ...] = ()
    fact_ids: tuple[StableIdentifier, ...] = ()
    covered_plan_steps: tuple[NonNegativeInt, ...] = ()
    missing_plan_steps: tuple[NonNegativeInt, ...] = ()
    missing_information: tuple[NonEmptyText, ...] = ()
    termination_reason: TerminationReason | None = None
    error_kind: ErrorKind | None = None
    error_message: NonEmptyText | None = None

    @property
    def route(self) -> BranchRoute:
        return {
            AggregatorStatus.CONTINUE: BranchRoute.EXECUTE,
            AggregatorStatus.COMPLETE: BranchRoute.COMPLETE,
            AggregatorStatus.REPLAN: BranchRoute.REPLAN,
            AggregatorStatus.ERROR: BranchRoute.ERROR,
        }[self.status]

    @model_validator(mode="after")
    def validate_aggregation(self) -> Self:
        _ensure_unique("evidence_ids", self.evidence_ids)
        _ensure_unique("fact_ids", self.fact_ids)
        _ensure_unique("covered_plan_steps", self.covered_plan_steps)
        _ensure_unique("missing_plan_steps", self.missing_plan_steps)

        overlap = set(self.covered_plan_steps) & set(self.missing_plan_steps)
        if overlap:
            raise ValueError(
                "covered_plan_steps and missing_plan_steps cannot overlap"
            )

        if self.status is AggregatorStatus.ERROR:
            if self.error_kind is None or self.error_message is None:
                raise ValueError("error status requires error_kind and error_message")
        elif self.error_kind is not None or self.error_message is not None:
            raise ValueError("non-error status cannot contain error details")

        if self.status in {
            AggregatorStatus.COMPLETE,
            AggregatorStatus.ERROR,
        }:
            if self.termination_reason is None:
                raise ValueError(
                    "terminal aggregator status requires termination_reason"
                )
        elif self.termination_reason is not None:
            raise ValueError(
                "non-terminal aggregator status cannot contain termination_reason"
            )

        return self


class BranchResult(_BranchScope):
    """Order-independent terminal summary returned by one corpus branch."""

    status: BranchStatus
    revision: NonNegativeInt = 0
    corpus: CorpusName
    goal: NonEmptyText
    iteration_count: NonNegativeInt
    termination_reason: TerminationReason
    batch_ids: tuple[StableIdentifier, ...] = ()
    action_ids: tuple[StableIdentifier, ...] = ()
    evidence_ids: tuple[StableIdentifier, ...] = ()
    fact_ids: tuple[StableIdentifier, ...] = ()
    warnings: tuple[NonEmptyText, ...] = ()
    error_kind: ErrorKind | None = None
    error_message: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _ensure_unique("batch_ids", self.batch_ids)
        _ensure_unique("action_ids", self.action_ids)
        _ensure_unique("evidence_ids", self.evidence_ids)
        _ensure_unique("fact_ids", self.fact_ids)

        if self.status is BranchStatus.ERROR:
            if self.error_kind is None or self.error_message is None:
                raise ValueError("error branch results require error details")
        elif self.status is BranchStatus.COMPLETE:
            if self.error_kind is not None or self.error_message is not None:
                raise ValueError(
                    "complete branch results cannot contain error details"
                )
        elif (self.error_kind is None) != (self.error_message is None):
            raise ValueError(
                "partial branch error_kind and error_message must be set together"
            )

        return self


__all__ = [
    "ActionCall",
    "ActionStatus",
    "AggregatorDecision",
    "AggregatorStatus",
    "BranchResult",
    "BranchRoute",
    "BranchStatus",
    "ErrorKind",
    "EvidenceArtifact",
    "EvidenceStatus",
    "ExecutionPlan",
    "ExecutionStatus",
    "Fact",
    "PlannerDecision",
    "PlannerMode",
    "PlannerRoute",
    "ResearchMode",
    "ResearchStatus",
    "ResearchSubGoal",
    "TerminationReason",
]
