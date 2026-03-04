"""Tests for recommendation tool wrappers."""

from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from finance_ai.agents.recommendation_tools import (
    RECOMMENDATION_TOOLS,
    generate_financial_recommendations,
    get_financial_health_score,
)
from finance_ai.database.models.user import User


class TestToolList:
    """Tests for RECOMMENDATION_TOOLS list."""

    def test_has_two_tools(self) -> None:
        """Should contain exactly 2 tools."""
        assert len(RECOMMENDATION_TOOLS) == 2

    def test_contains_expected_tools(self) -> None:
        """Should contain the recommendation and health score tools."""
        names = [t.name for t in RECOMMENDATION_TOOLS]
        assert "generate_financial_recommendations" in names
        assert "get_financial_health_score" in names


class TestGenerateFinancialRecommendations:
    """Tests for generate_financial_recommendations tool."""

    def test_returns_report_structure(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Should return a valid report dict."""
        result = generate_financial_recommendations.invoke(
            {
                "year": "2026",
                "month": "3",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            },
        )
        assert "total_recommendations" in result
        assert "health_score" in result
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)

    def test_defaults_to_current_period(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Empty year/month should default to current period."""
        result = generate_financial_recommendations.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            },
        )
        assert "health_score" in result

    def test_recommendations_have_category_label(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Each recommendation should have a category_label."""
        result = generate_financial_recommendations.invoke(
            {
                "year": "2026",
                "month": "3",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            },
        )
        for rec in result["recommendations"]:
            assert "category_label" in rec
            assert isinstance(rec["category_label"], str)


class TestGetFinancialHealthScore:
    """Tests for get_financial_health_score tool."""

    def test_returns_score_structure(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Should return health score and top issues."""
        result = get_financial_health_score.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            },
        )
        assert "health_score" in result
        assert "total_recommendations" in result
        assert "top_issues" in result
        assert isinstance(result["health_score"], int)
        assert 0 <= result["health_score"] <= 100

    def test_top_issues_limited_to_three(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Top issues should be at most 3 items."""
        result = get_financial_health_score.invoke(
            {
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            },
        )
        assert len(result["top_issues"]) <= 3
