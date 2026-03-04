"""LangGraph tool wrappers for proactive financial recommendations.

Tools gather all user financial data, run rule-based analysis,
and return structured recommendations for the LLM to present.
"""

from datetime import date
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.session_helper import get_tool_session


@tool
def generate_financial_recommendations(
    year: str = "",
    month: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Generate proactive financial recommendations.

    Analyzes all user financial data (expenses, income, tax,
    investments, goals) and returns prioritized recommendations.

    Args:
        year: Year to analyze (default current year).
        month: Month to analyze (default current month).
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with recommendation report including health score.
    """
    today = date.today()
    parsed_year = int(year) if year else today.year
    parsed_month = int(month) if month else today.month
    return _fetch_recommendations(db_session_factory, user_id, parsed_year, parsed_month)


def _fetch_recommendations(
    db_session_factory: Any,
    user_id: str,
    year: int,
    month: int,
) -> dict[str, Any]:
    """Fetch recommendations via the service layer.

    Args:
        db_session_factory: Session factory or None.
        user_id: UUID of the user.
        year: Year to analyze.
        month: Month to analyze.

    Returns:
        Serialized RecommendationReport dict.
    """
    from finance_ai.tools.recommendation_service import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        generate_recommendations,
    )

    with get_tool_session(db_session_factory) as session:
        report = generate_recommendations(session, user_id, year, month)
    return _serialize_report(report)


def _serialize_report(report: Any) -> dict[str, Any]:
    """Convert RecommendationReport to serializable dict.

    Args:
        report: RecommendationReport instance.

    Returns:
        Dict suitable for LLM consumption.
    """
    return {
        "user_id": report.user_id,
        "generated_at": report.generated_at,
        "total_recommendations": report.total_recommendations,
        "health_score": report.health_score,
        "recommendations": [
            {
                "category": r.category,
                "category_label": _get_category_label(r.category),
                "priority": r.priority,
                "title": r.title,
                "description": r.description,
                "action_items": r.action_items,
                "estimated_impact": r.estimated_impact,
            }
            for r in report.recommendations
        ],
    }


def _get_category_label(category: str) -> str:
    """Get Thai label for a recommendation category.

    Args:
        category: Category key.

    Returns:
        Thai label string.
    """
    from finance_ai.tools.recommendation_constants import (  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
        RECOMMENDATION_CATEGORIES,
    )

    return RECOMMENDATION_CATEGORIES.get(category, category)


@tool
def get_financial_health_score(
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Calculate overall financial health score (0-100).

    Analyzes all financial domains and returns a score.
    Higher score means healthier financial situation.

    Args:
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with health score and recommendation count.
    """
    today = date.today()
    report = _fetch_recommendations(db_session_factory, user_id, today.year, today.month)
    return {
        "health_score": report["health_score"],
        "total_recommendations": report["total_recommendations"],
        "top_issues": [r["title"] for r in report["recommendations"][:3]],
    }


RECOMMENDATION_TOOLS = [
    generate_financial_recommendations,
    get_financial_health_score,
]
