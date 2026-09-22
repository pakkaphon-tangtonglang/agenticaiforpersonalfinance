"""State schemas for LangGraph agents and structured output models."""

from collections.abc import Callable
from decimal import Decimal
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from langgraph.graph.message import add_messages


class TaxAgentState(TypedDict):
    """State for the Tax Agent LangGraph graph.

    Attributes:
        messages: Conversation messages (LangGraph manages append via add_messages).
        user_id: UUID of the user for cross-agent DB operations.
        db_session_factory: Optional session factory for DB access (injected into tools).

    Example:
        >>> state: TaxAgentState = {
        ...     "messages": [], "user_id": "",
        ...     "db_session_factory": None,
        ... }
    """

    messages: Annotated[list[Any], add_messages]
    user_id: str
    db_session_factory: Callable[[], Session] | None


class ExpenseAgentState(TypedDict):
    """State for the Expense Agent LangGraph graph.

    Attributes:
        messages: Conversation messages (LangGraph manages append via add_messages).
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access (injected into tools).

    Example:
        >>> state: ExpenseAgentState = {
        ...     "messages": [], "user_id": "abc-123",
        ...     "db_session_factory": None,
        ... }
    """

    messages: Annotated[list[Any], add_messages]
    user_id: str
    db_session_factory: Callable[[], Session] | None


class OrchestratorDecision(BaseModel):
    """Structured output from the Router Agent.

    Attributes:
        intent: Classified intent of the user query.
        confidence: Confidence score (0.0 to 1.0).

    Example:
        >>> decision = OrchestratorDecision(intent="tax", confidence=Decimal("0.95"))
    """

    intent: Literal[
        "tax",
        "asset_monitoring",
        "expense",
        "planning",
        "recommendation",
        "report",
        "general",
        "unknown",
        # Produced only by the ablation confidence-threshold toggle in
        # classify_query, never by the LLM or the production router path.
        "clarify",
    ] = Field(description="The classified intent of the user query.")
    confidence: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Confidence score of the classification.",
    )


class AssetMonitoringAgentState(TypedDict):
    """State for the Investment Agent LangGraph graph.

    Attributes:
        messages: Conversation messages (LangGraph manages append via add_messages).
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access (injected into tools).

    Example:
        >>> state: AssetMonitoringAgentState = {
        ...     "messages": [], "user_id": "abc-123",
        ...     "db_session_factory": None,
        ... }
    """

    messages: Annotated[list[Any], add_messages]
    user_id: str
    db_session_factory: Callable[[], Session] | None


class PlanningAgentState(TypedDict):
    """State for the Planning Agent LangGraph graph.

    Attributes:
        messages: Conversation messages (LangGraph manages append via add_messages).
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access (injected into tools).

    Example:
        >>> state: PlanningAgentState = {
        ...     "messages": [], "user_id": "abc-123",
        ...     "db_session_factory": None,
        ... }
    """

    messages: Annotated[list[Any], add_messages]
    user_id: str
    db_session_factory: Callable[[], Session] | None


class RecommendationAgentState(TypedDict):
    """State for the Recommendation Agent LangGraph graph.

    Attributes:
        messages: Conversation messages (LangGraph manages append via add_messages).
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access (injected into tools).

    Example:
        >>> state: RecommendationAgentState = {
        ...     "messages": [],
        ...     "user_id": "abc-123", "db_session_factory": None,
        ... }
    """

    messages: Annotated[list[Any], add_messages]
    user_id: str
    db_session_factory: Callable[[], Session] | None


class ReportAgentState(TypedDict):
    """State for the Report Agent LangGraph graph.

    Attributes:
        messages: Conversation messages (LangGraph manages append via add_messages).
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access (injected into tools).

    Example:
        >>> state: ReportAgentState = {
        ...     "messages": [],
        ...     "user_id": "abc-123", "db_session_factory": None,
        ... }
    """

    messages: Annotated[list[Any], add_messages]
    user_id: str
    db_session_factory: Callable[[], Session] | None
