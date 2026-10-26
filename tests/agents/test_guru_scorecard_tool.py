"""Tests for the get_guru_scorecard tool."""

from typing import Any
from unittest.mock import MagicMock, patch

from langchain_core.utils.function_calling import convert_to_openai_tool

from finance_ai.agents.guru_scorecard_tool import get_guru_scorecard

GATHERED_DATA: dict[str, Any] = {
    "income_monthly": {"total_income": "50000"},
    "expense": {"total_amount": "40000", "category_breakdown": []},
    "goals": {"goals": []},
    "portfolio": {"total_value": "0", "holdings": []},
}


def _invoke(guru_id: str) -> dict[str, Any]:
    """Call the tool with injected state values, as ToolNode does."""
    result: dict[str, Any] = get_guru_scorecard.func(  # type: ignore[attr-defined]
        guru_id=guru_id, user_id="u1", db_session_factory=MagicMock()
    )
    return result


class TestGetGuruScorecard:
    """Tests for get_guru_scorecard."""

    @patch("finance_ai.agents.guru_scorecard_tool.get_user_risk_level", return_value=3)
    @patch("finance_ai.agents.guru_scorecard_tool.gather_all_financial_data")
    @patch("finance_ai.agents.guru_scorecard_tool.get_tool_session")
    def test_returns_active_gurus_checks(
        self, _mock_session: MagicMock, mock_gather: MagicMock, _mock_risk: MagicMock
    ) -> None:
        """The scorecard is built for the guru in state from the user's data."""
        mock_gather.return_value = GATHERED_DATA

        result = _invoke("coach_noom")

        assert result["guru_id"] == "coach_noom"
        savings = next(check for check in result["checks"] if check["key"] == "savings_rate")
        assert savings["value"] == "20.00%"

    def test_missing_guru_returns_error(self) -> None:
        """Without a guru in state the tool refuses."""
        assert "error" in _invoke("")

    def test_only_injected_arguments(self) -> None:
        """The model sees no arguments: everything comes from graph state."""
        llm_schema = convert_to_openai_tool(get_guru_scorecard)["function"]["parameters"]

        assert not llm_schema.get("properties")
