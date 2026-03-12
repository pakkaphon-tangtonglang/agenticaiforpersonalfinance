"""Tests for finance_ai.ui.export_service — PDF and CSV export."""

from decimal import Decimal

import pytest

from finance_ai.tools.report_models import (
    ExpenseBreakdownSection,
    FinancialReport,
    GoalProgressSection,
    InvestmentPortfolioSection,
    MonthlyOverviewSection,
    TaxStatusSection,
)
from finance_ai.ui.export_service import (
    format_thai_currency,
    generate_report_pdf,
    generate_transactions_csv,
)


def _make_report() -> FinancialReport:
    """Create a sample FinancialReport for testing."""
    return FinancialReport(
        user_id="test-user",
        generated_at="2026-03-10T00:00:00",
        report_type="monthly",
        year=2026,
        month=3,
        monthly_overview=MonthlyOverviewSection(
            total_income=Decimal("50000"),
            total_expenses=Decimal("35000"),
            net_savings=Decimal("15000"),
            savings_rate=Decimal("0.30"),
        ),
        expense_breakdown=ExpenseBreakdownSection(
            total_amount=Decimal("35000"),
            transaction_count=10,
            categories=[
                {"category": "food", "amount": "15000", "percentage": "42.86"},
                {"category": "transport", "amount": "5000", "percentage": "14.29"},
            ],
        ),
        investment_portfolio=InvestmentPortfolioSection(
            total_value=Decimal("500000"),
            total_cost=Decimal("450000"),
            total_gain_loss=Decimal("50000"),
            gain_loss_percentage=Decimal("11.11"),
            holding_count=2,
        ),
        goal_progress=GoalProgressSection(
            total_goals=2,
            active_goals=1,
            completed_goals=1,
            overall_percentage=Decimal("60"),
        ),
        tax_status=TaxStatusSection(
            status="not_filed",
            gross_income=Decimal("600000"),
            total_deductions=Decimal("160000"),
            total_tax=Decimal("29000"),
            effective_rate=Decimal("4.83"),
            tax_year=2026,
        ),
        health_score=72,
    )


class TestFormatThaiCurrency:
    """Tests for format_thai_currency."""

    def test_decimal_input(self) -> None:
        """Decimal value formats correctly."""
        result = format_thai_currency(Decimal("50000"))
        assert result == "50,000.00 บาท"

    def test_float_input(self) -> None:
        """Float value formats correctly."""
        result = format_thai_currency(1234.5)
        assert result == "1,234.50 บาท"

    def test_string_input(self) -> None:
        """String value formats correctly."""
        result = format_thai_currency("99999.99")
        assert result == "99,999.99 บาท"

    def test_zero(self) -> None:
        """Zero formats correctly."""
        result = format_thai_currency(Decimal("0"))
        assert result == "0.00 บาท"


class TestGenerateReportPdf:
    """Tests for generate_report_pdf."""

    def test_returns_bytes(self) -> None:
        """PDF generation returns bytes-like object."""
        report = _make_report()
        result = generate_report_pdf(report)
        assert isinstance(result, (bytes, bytearray))

    def test_pdf_header(self) -> None:
        """PDF starts with PDF magic bytes."""
        report = _make_report()
        result = generate_report_pdf(report)
        assert result[:5] == b"%PDF-"

    def test_non_empty(self) -> None:
        """PDF has meaningful content (> 1KB)."""
        report = _make_report()
        result = generate_report_pdf(report)
        assert len(result) > 1000

    def test_default_report(self) -> None:
        """Minimal report generates without error."""
        report = FinancialReport(
            user_id="test",
            generated_at="2026-01-01T00:00:00",
        )
        result = generate_report_pdf(report)
        assert result[:5] == b"%PDF-"


class TestGenerateTransactionsCsv:
    """Tests for generate_transactions_csv."""

    def test_has_bom(self) -> None:
        """CSV starts with UTF-8 BOM."""
        result = generate_transactions_csv([])
        assert result.startswith("\ufeff")

    def test_header_row(self) -> None:
        """CSV has Thai header row."""
        result = generate_transactions_csv([])
        lines = result.strip().split("\n")
        assert "วันที่" in lines[0]
        assert "จำนวนเงิน" in lines[0]

    def test_data_rows(self) -> None:
        """CSV includes transaction data rows."""
        transactions = [
            {
                "date": "2026-03-01",
                "type": "expense",
                "category": "food",
                "description": "ข้าวมันไก่",
                "amount": "50.00",
            },
            {
                "date": "2026-03-02",
                "type": "expense",
                "category": "transport",
                "description": "BTS",
                "amount": "44.00",
            },
        ]
        result = generate_transactions_csv(transactions)
        lines = result.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_category_translation(self) -> None:
        """Category keys are translated to Thai."""
        transactions = [
            {"date": "2026-03-01", "category": "food", "amount": "100"},
        ]
        result = generate_transactions_csv(transactions)
        assert "อาหาร" in result

    def test_empty_transactions(self) -> None:
        """Empty list produces header-only CSV."""
        result = generate_transactions_csv([])
        lines = result.strip().split("\n")
        assert len(lines) == 1
