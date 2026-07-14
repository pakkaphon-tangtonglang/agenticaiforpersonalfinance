"""Tests for agent state schemas and Pydantic models."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from finance_ai.agents.schemas import (
    ExpenseAgentState,
    OrchestratorDecision,
)


class TestOrchestratorDecision:
    """Tests for OrchestratorDecision model."""

    def test_valid_tax_intent(self) -> None:
        """OrchestratorDecision accepts valid tax intent with confidence."""
        decision = OrchestratorDecision(intent="tax", confidence=Decimal("0.95"))
        assert decision.intent == "tax"
        assert decision.confidence == Decimal("0.95")

    def test_valid_asset_monitoring_intent(self) -> None:
        """OrchestratorDecision accepts valid asset_monitoring intent."""
        decision = OrchestratorDecision(intent="asset_monitoring", confidence=Decimal("0.8"))
        assert decision.intent == "asset_monitoring"

    def test_valid_general_intent(self) -> None:
        """OrchestratorDecision accepts valid general intent."""
        decision = OrchestratorDecision(intent="general", confidence=Decimal("0.5"))
        assert decision.intent == "general"

    def test_valid_unknown_intent(self) -> None:
        """OrchestratorDecision accepts unknown intent."""
        decision = OrchestratorDecision(intent="unknown", confidence=Decimal("0.0"))
        assert decision.intent == "unknown"

    def test_valid_expense_intent(self) -> None:
        """OrchestratorDecision accepts expense intent."""
        decision = OrchestratorDecision(intent="expense", confidence=Decimal("0.9"))
        assert decision.intent == "expense"

    def test_invalid_intent_raises(self) -> None:
        """OrchestratorDecision rejects invalid intent values."""
        with pytest.raises(ValidationError):
            OrchestratorDecision(intent="budgeting", confidence=Decimal("0.5"))  # type: ignore[arg-type]

    def test_confidence_below_zero_raises(self) -> None:
        """OrchestratorDecision rejects confidence below 0."""
        with pytest.raises(ValidationError):
            OrchestratorDecision(intent="tax", confidence=Decimal("-0.1"))

    def test_confidence_above_one_raises(self) -> None:
        """OrchestratorDecision rejects confidence above 1."""
        with pytest.raises(ValidationError):
            OrchestratorDecision(intent="tax", confidence=Decimal("1.1"))

    def test_confidence_boundary_zero(self) -> None:
        """OrchestratorDecision accepts confidence of exactly 0."""
        decision = OrchestratorDecision(intent="tax", confidence=Decimal("0"))
        assert decision.confidence == Decimal("0")

    def test_confidence_boundary_one(self) -> None:
        """OrchestratorDecision accepts confidence of exactly 1."""
        decision = OrchestratorDecision(intent="tax", confidence=Decimal("1"))
        assert decision.confidence == Decimal("1")


class TestExpenseAgentState:
    """Tests for ExpenseAgentState TypedDict."""

    def test_creates_with_all_fields(self) -> None:
        """ExpenseAgentState can be instantiated with all fields."""
        state: ExpenseAgentState = {
            "messages": [],
            "user_id": "test-user-123",
            "db_session_factory": None,
        }
        assert state["messages"] == []
        assert state["user_id"] == "test-user-123"
