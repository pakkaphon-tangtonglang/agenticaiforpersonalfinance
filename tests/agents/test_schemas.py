"""Tests for agent state schemas and Pydantic models."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from finance_ai.agents.schemas import (
    ExpenseAgentState,
    ExtractedTaxParameters,
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


class TestExtractedTaxParameters:
    """Tests for ExtractedTaxParameters model."""

    def test_defaults(self) -> None:
        """ExtractedTaxParameters has correct default values."""
        params = ExtractedTaxParameters()
        assert params.tax_year is None
        assert params.gross_income is None
        assert params.deductions_by_type == {}
        assert params.withholding_tax_paid == Decimal("0")
        assert params.missing_fields == []

    def test_with_all_fields(self) -> None:
        """ExtractedTaxParameters populates all fields correctly."""
        params = ExtractedTaxParameters(
            tax_year=2024,
            gross_income=Decimal("1200000"),
            deductions_by_type={"personal_allowance": Decimal("60000"), "rmf": Decimal("100000")},
            withholding_tax_paid=Decimal("120000"),
            missing_fields=["spouse_status"],
        )
        assert params.tax_year == 2024
        assert params.gross_income == Decimal("1200000")
        assert len(params.deductions_by_type) == 2
        assert params.withholding_tax_paid == Decimal("120000")
        assert "spouse_status" in params.missing_fields

    def test_partial_fields(self) -> None:
        """ExtractedTaxParameters works with partial fields."""
        params = ExtractedTaxParameters(
            gross_income=Decimal("500000"),
            missing_fields=["tax_year"],
        )
        assert params.gross_income == Decimal("500000")
        assert params.tax_year is None
        assert params.missing_fields == ["tax_year"]


class TestExpenseAgentState:
    """Tests for ExpenseAgentState TypedDict."""

    def test_creates_with_all_fields(self) -> None:
        """ExpenseAgentState can be instantiated with all fields."""
        state: ExpenseAgentState = {
            "messages": [],
            "expense_result": None,
            "user_id": "test-user-123",
            "db_session_factory": None,
        }
        assert state["messages"] == []
        assert state["expense_result"] is None
        assert state["user_id"] == "test-user-123"

    def test_with_expense_result(self) -> None:
        """ExpenseAgentState stores expense result dict."""
        result = {"total_amount": "400.00", "transaction_count": 3}
        state: ExpenseAgentState = {
            "messages": [],
            "expense_result": result,
            "user_id": "test-user-456",
            "db_session_factory": None,
        }
        assert state["expense_result"] is not None
        assert state["expense_result"]["total_amount"] == "400.00"
