"""Tests for UI application constants."""

from finance_ai.ui.app_constants import (
    FEATURE_CARDS,
    INTENT_CONFIG,
    MULTI_AGENT_QUERIES,
    RECOMMENDATION_QUERIES,
    REPORT_QUERIES,
    SAMPLE_QUERIES,
)


class TestAppConstants:
    """Tests for app-level UI constants."""

    def test_intent_config_has_required_intents(self) -> None:
        """INTENT_CONFIG contains all specialist intents."""
        required_intents = {
            "tax",
            "expense",
            "asset_monitoring",
            "planning",
            "recommendation",
            "report",
            "general",
            "unknown",
        }
        assert required_intents.issubset(set(INTENT_CONFIG.keys()))
        for config in INTENT_CONFIG.values():
            assert "label" in config
            assert "icon" in config
            assert "color" in config

    def test_sample_queries_are_non_empty(self) -> None:
        """Sample query lists contain entries with short and full text."""
        for query_list in (
            SAMPLE_QUERIES,
            MULTI_AGENT_QUERIES,
            REPORT_QUERIES,
            RECOMMENDATION_QUERIES,
        ):
            assert query_list
            for query in query_list:
                assert query["short"]
                assert query["full"]

    def test_feature_cards_have_content(self) -> None:
        """FEATURE_CARDS list contains icon, title and description."""
        assert FEATURE_CARDS
        for card in FEATURE_CARDS:
            assert card["icon"]
            assert card["title"]
            assert card["desc"]
