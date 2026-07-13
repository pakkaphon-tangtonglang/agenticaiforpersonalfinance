"""Tests for report tool wrappers."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from finance_ai.agents.report_tools import (
    REPORT_TOOLS,
    _serialize_report,
    _serialize_section,
    generate_financial_report_tool,
    get_financial_summary,
)


class TestReportToolsList:
    """Tests for REPORT_TOOLS export."""

    def test_contains_two_tools(self) -> None:
        """Should export exactly 2 tools."""
        assert len(REPORT_TOOLS) == 2

    def test_contains_expected_tools(self) -> None:
        """Should contain report and summary tools."""
        names = [t.name for t in REPORT_TOOLS]
        assert "generate_financial_report_tool" in names
        assert "get_financial_summary" in names


class TestSerializeSection:
    """Tests for _serialize_section helper."""

    def test_converts_decimals_to_strings(self) -> None:
        """Should convert Decimal values to strings."""
        mock_section = MagicMock()
        mock_section.model_dump.return_value = {
            "total_income": Decimal("50000"),
            "count": 5,
            "name": "test",
        }
        result = _serialize_section(mock_section)
        assert result["total_income"] == "50000"
        assert result["count"] == 5
        assert result["name"] == "test"


class TestSerializeReport:
    """Tests for _serialize_report helper."""

    def test_serializes_full_report(self) -> None:
        """Should convert report to dict with all sections."""
        mock_report = MagicMock()
        mock_report.user_id = "user-1"
        mock_report.generated_at = "2026-03-10"
        mock_report.report_type = "monthly"
        mock_report.year = 2026
        mock_report.month = 3
        mock_report.health_score = 85
        mock_report.highlights = []

        for section_name in [
            "monthly_overview",
            "expense_breakdown",
            "investment_portfolio",
            "goal_progress",
            "tax_status",
        ]:
            section = MagicMock()
            section.model_dump.return_value = {"value": Decimal("100")}
            setattr(mock_report, section_name, section)

        result = _serialize_report(mock_report)
        assert result["action"] == "financial_report"
        assert result["health_score"] == 85
        assert "monthly_overview" in result
        assert "highlights" in result


class TestGenerateFinancialReportTool:
    """Tests for generate_financial_report_tool @tool."""

    @patch("finance_ai.tools.report_service.generate_financial_report")
    def test_calls_service(self, mock_gen: MagicMock) -> None:
        """Should call service and return serialized dict."""
        mock_report = MagicMock()
        mock_report.user_id = "u1"
        mock_report.generated_at = "2026-03-10"
        mock_report.report_type = "monthly"
        mock_report.year = 2026
        mock_report.month = 3
        mock_report.health_score = 90
        mock_report.highlights = []
        for section_name in [
            "monthly_overview",
            "expense_breakdown",
            "investment_portfolio",
            "goal_progress",
            "tax_status",
        ]:
            section = MagicMock()
            section.model_dump.return_value = {}
            setattr(mock_report, section_name, section)
        mock_gen.return_value = mock_report

        result = generate_financial_report_tool.invoke({"year": "2026", "month": "3"})
        assert result["action"] == "financial_report"


class TestGetFinancialSummary:
    """Tests for get_financial_summary @tool."""

    @patch("finance_ai.tools.report_service.generate_financial_report")
    def test_returns_summary(self, mock_gen: MagicMock) -> None:
        """Should return quick summary dict."""
        mock_report = MagicMock()
        mock_report.user_id = "u1"
        mock_report.generated_at = "2026-03-10"
        mock_report.report_type = "monthly"
        mock_report.year = 2026
        mock_report.month = 3
        mock_report.health_score = 75
        mock_report.highlights = []
        overview = MagicMock()
        overview.model_dump.return_value = {
            "total_income": Decimal("50000"),
            "total_expenses": Decimal("35000"),
            "net_savings": Decimal("15000"),
            "savings_rate": Decimal("0.30"),
        }
        mock_report.monthly_overview = overview
        for section_name in [
            "expense_breakdown",
            "investment_portfolio",
            "goal_progress",
            "tax_status",
        ]:
            section = MagicMock()
            section.model_dump.return_value = {}
            setattr(mock_report, section_name, section)
        mock_gen.return_value = mock_report

        result = get_financial_summary.invoke({})
        assert result["action"] == "financial_summary"
        assert result["health_score"] == 75
