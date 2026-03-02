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
        tax_result: The computed TaxCalculationResult dict, if available.
        user_id: UUID of the user for cross-agent DB operations.
        db_session_factory: Optional session factory for DB access (injected into tools).

    Example:
        >>> state: TaxAgentState = {
        ...     "messages": [], "tax_result": None, "user_id": "",
        ...     "db_session_factory": None,
        ... }
    """

    messages: Annotated[list[Any], add_messages]
    tax_result: dict[str, Any] | None
    user_id: str
    db_session_factory: Callable[[], Session] | None


class ExpenseAgentState(TypedDict):
    """State for the Expense Agent LangGraph graph.

    Attributes:
        messages: Conversation messages (LangGraph manages append via add_messages).
        expense_result: The computed expense result dict, if available.
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access (injected into tools).

    Example:
        >>> state: ExpenseAgentState = {
        ...     "messages": [], "expense_result": None, "user_id": "abc-123",
        ...     "db_session_factory": None,
        ... }
    """

    messages: Annotated[list[Any], add_messages]
    expense_result: dict[str, Any] | None
    user_id: str
    db_session_factory: Callable[[], Session] | None


class RouterDecision(BaseModel):
    """Structured output from the Router Agent.

    Attributes:
        intent: Classified intent of the user query.
        confidence: Confidence score (0.0 to 1.0).

    Example:
        >>> decision = RouterDecision(intent="tax", confidence=Decimal("0.95"))
    """

    intent: Literal["tax", "investment", "expense", "planning", "general", "unknown"] = Field(
        description="The classified intent of the user query."
    )
    confidence: Decimal = Field(
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Confidence score of the classification.",
    )


class InvestmentAgentState(TypedDict):
    """State for the Investment Agent LangGraph graph.

    Attributes:
        messages: Conversation messages (LangGraph manages append via add_messages).
        investment_result: The computed portfolio result dict, if available.
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access (injected into tools).

    Example:
        >>> state: InvestmentAgentState = {
        ...     "messages": [], "investment_result": None, "user_id": "abc-123",
        ...     "db_session_factory": None,
        ... }
    """

    messages: Annotated[list[Any], add_messages]
    investment_result: dict[str, Any] | None
    user_id: str
    db_session_factory: Callable[[], Session] | None


class PlanningAgentState(TypedDict):
    """State for the Planning Agent LangGraph graph.

    Attributes:
        messages: Conversation messages (LangGraph manages append via add_messages).
        planning_result: The computed planning result dict, if available.
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access (injected into tools).

    Example:
        >>> state: PlanningAgentState = {
        ...     "messages": [], "planning_result": None, "user_id": "abc-123",
        ...     "db_session_factory": None,
        ... }
    """

    messages: Annotated[list[Any], add_messages]
    planning_result: dict[str, Any] | None
    user_id: str
    db_session_factory: Callable[[], Session] | None


class ExtractedTaxParameters(BaseModel):
    """Parameters extracted from user's natural language tax query.

    Attributes:
        tax_year: The tax year to calculate for.
        gross_income: Annual gross income in THB.
        deductions_by_type: Mapping of deduction type to amount.
        withholding_tax_paid: Total withholding tax already paid.
        missing_fields: Fields the user did not provide.

    Example:
        >>> params = ExtractedTaxParameters(gross_income=Decimal("1200000"))
    """

    tax_year: int | None = Field(default=None)
    gross_income: Decimal | None = Field(default=None)
    deductions_by_type: dict[str, Decimal] = Field(default_factory=dict)
    withholding_tax_paid: Decimal = Field(default=Decimal("0"))
    missing_fields: list[str] = Field(default_factory=list)
