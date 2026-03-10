"""Tests for report constants."""

from decimal import Decimal

from finance_ai.tools.report_constants import (
    DEFAULT_REPORT_TYPE,
    GOAL_NEAR_COMPLETE_THRESHOLD,
    HIGHLIGHT_TYPES,
    MONTHS_PER_YEAR,
    REPORT_SECTIONS,
    REPORT_TYPES,
    SAVINGS_RATE_GOOD_THRESHOLD,
)


class TestReportSections:
    """Tests for REPORT_SECTIONS constant."""

    def test_has_seven_sections(self) -> None:
        """Should define exactly 7 report sections."""
        assert len(REPORT_SECTIONS) == 7

    def test_contains_monthly_overview(self) -> None:
        """Must include monthly_overview section."""
        assert "monthly_overview" in REPORT_SECTIONS

    def test_contains_highlights(self) -> None:
        """Must include highlights section."""
        assert "highlights" in REPORT_SECTIONS

    def test_all_labels_are_thai(self) -> None:
        """All labels should be non-empty strings."""
        for label in REPORT_SECTIONS.values():
            assert isinstance(label, str)
            assert len(label) > 0


class TestReportTypes:
    """Tests for REPORT_TYPES constant."""

    def test_has_monthly_and_annual(self) -> None:
        """Should support monthly and annual report types."""
        assert "monthly" in REPORT_TYPES
        assert "annual" in REPORT_TYPES


class TestHighlightTypes:
    """Tests for HIGHLIGHT_TYPES constant."""

    def test_has_three_types(self) -> None:
        """Should define achievement, warning, and info."""
        assert len(HIGHLIGHT_TYPES) == 3
        assert "achievement" in HIGHLIGHT_TYPES
        assert "warning" in HIGHLIGHT_TYPES
        assert "info" in HIGHLIGHT_TYPES


class TestThresholds:
    """Tests for threshold constants."""

    def test_savings_rate_threshold_is_decimal(self) -> None:
        """Savings threshold should be Decimal 0.20."""
        assert isinstance(SAVINGS_RATE_GOOD_THRESHOLD, Decimal)
        assert SAVINGS_RATE_GOOD_THRESHOLD == Decimal("0.20")

    def test_goal_near_complete_threshold(self) -> None:
        """Goal threshold should be Decimal 80.00."""
        assert isinstance(GOAL_NEAR_COMPLETE_THRESHOLD, Decimal)
        assert GOAL_NEAR_COMPLETE_THRESHOLD == Decimal("80.00")

    def test_months_per_year(self) -> None:
        """Should be 12."""
        assert MONTHS_PER_YEAR == 12

    def test_default_report_type(self) -> None:
        """Default type should be monthly."""
        assert DEFAULT_REPORT_TYPE == "monthly"
