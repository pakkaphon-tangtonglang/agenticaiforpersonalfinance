"""Tests for cross-agent constants."""

from finance_ai.tools.cross_agent_constants import (
    CROSS_AGENT_DOMAINS,
    SUMMARY_TYPES,
)


class TestCrossAgentDomains:
    """Tests for CROSS_AGENT_DOMAINS mapping."""

    def test_has_all_domains(self) -> None:
        """Verify all expected domains are present."""
        expected = {"tax", "expense", "investment", "planning", "income"}
        assert set(CROSS_AGENT_DOMAINS.keys()) == expected

    def test_values_are_thai(self) -> None:
        """All domain labels should be non-empty Thai strings."""
        for label in CROSS_AGENT_DOMAINS.values():
            assert isinstance(label, str)
            assert len(label) > 0


class TestSummaryTypes:
    """Tests for SUMMARY_TYPES mapping."""

    def test_has_all_summary_types(self) -> None:
        """Verify all expected summary types are present."""
        expected = {
            "expense_summary",
            "portfolio_summary",
            "goals_summary",
            "income_summary",
            "tax_filing_summary",
        }
        assert set(SUMMARY_TYPES.keys()) == expected

    def test_values_are_thai(self) -> None:
        """All summary type labels should be non-empty Thai strings."""
        for label in SUMMARY_TYPES.values():
            assert isinstance(label, str)
            assert len(label) > 0
