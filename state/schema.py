"""LangGraph state schema with explicit lifecycle and isolation boundaries."""

from __future__ import annotations

from typing import Annotated, Any, Literal, NotRequired, TypedDict

from state.models import (
    ActionCall,
    AggregatorStatus,
    BranchResult,
    ErrorKind,
    EvidenceArtifact,
    ExecutionStatus,
    Fact,
    PlannerRoute,
    ResearchMode,
    ResearchStatus,
    TerminationReason,
)
from state.reducers import (
    merge_actions,
    merge_branch_results,
    merge_evidence,
    merge_facts,
    merge_unique_text,
)


class SafeError(TypedDict):
    kind: ErrorKind | str
    message: str
    node: NotRequired[str]
    retryable: NotRequired[bool]


class ResearchBudgets(TypedDict):
    max_iterations: int
    max_branches: int
    max_actions_per_branch: int
    max_requests_per_iteration: int
    max_external_calls: int
    max_wall_time_seconds: float
    max_evidence_items: int
    max_context_chars: int


class ResearchBudgetUsage(TypedDict):
    branches: int
    actions: int
    external_calls: int
    evidence_items: int
    context_chars: int


class ConversationTurn(TypedDict):
    turn_id: str
    research_id: str
    question: str
    answer: str


class ResearchSnapshot(TypedDict, total=False):
    research_id: str
    run_id: str
    turn_id: str
    question: str
    goal: str
    status: ResearchStatus | str
    final_response: str
    facts: list[Fact | dict[str, Any] | str]
    evidence: list[EvidenceArtifact | dict[str, Any]]
    termination_reason: TerminationReason | str | None


class ResearchState(TypedDict, total=False):
    # Conversation scope: survives multiple user turns in one thread.
    thread_id: str
    conversation_history: list[ConversationTurn]
    research_archive: dict[str, ResearchSnapshot]
    selected_context_research_id: str | None
    context_facts: list[Fact | dict[str, Any] | str]
    context_evidence: list[EvidenceArtifact | dict[str, Any]]

    # Current user turn and current research scope.
    turn_id: str
    research_id: str
    run_id: str
    active_research_question: str
    research_question: str
    final_response: str
    mode: ResearchMode | Literal["research", "chat", "replan", "error"]
    planner_route: PlannerRoute | str
    research_status: ResearchStatus | str

    # Current plan. These values are replaced on replan, not accumulated.
    goal: str
    research_plan: list[str]
    current_step_index: int
    recommended_corpus: str
    corpus_reasoning: str
    hypotheses: list[str]
    open_questions: list[str]

    # Current single-branch compatibility scope. Stage 4 will fan this out.
    branch_id: str
    batch_id: str
    iteration_count: int
    research_started_at: float
    budgets: ResearchBudgets
    budget_usage: ResearchBudgetUsage

    # Truly cumulative research-local channels. Lifecycle resets them explicitly.
    evidence: Annotated[
        list[EvidenceArtifact | dict[str, Any]],
        merge_evidence,
    ]
    facts: Annotated[list[Fact | dict[str, Any]], merge_facts]
    deductions: Annotated[list[str], merge_unique_text]
    completed_actions: Annotated[
        list[ActionCall | dict[str, Any]],
        merge_actions,
    ]
    branch_results: Annotated[
        list[BranchResult | dict[str, Any]],
        merge_branch_results,
    ]

    # Ephemeral batch data and explicit routing decisions.
    planned_actions: list[ActionCall | dict[str, Any]]
    last_evidence_batch: list[EvidenceArtifact | dict[str, Any]]
    execution_status: ExecutionStatus | str | None
    aggregator_status: AggregatorStatus | str | None
    aggregator_reasoning: str
    missing_information: list[str]
    needs_replanning: bool
    is_goal_reached: bool
    confidence: float
    pagination_context: dict[str, Any]

    # Terminal and diagnostic state.
    termination_reason: TerminationReason | str | None
    error: SafeError | None


__all__ = [
    "ConversationTurn",
    "ResearchBudgets",
    "ResearchBudgetUsage",
    "ResearchSnapshot",
    "ResearchState",
    "SafeError",
]
