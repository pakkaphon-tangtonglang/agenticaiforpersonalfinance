"""Guru scorecard tool: deterministic checks on the user's own finances.

The numbers are computed in code (guru_scorecard_service); the guru only
explains them, so it cannot invent the user's savings rate or allocation.
"""

from datetime import date
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.recommendation_agent import get_user_risk_level
from finance_ai.agents.session_helper import get_tool_session
from finance_ai.tools.guru_scorecard_service import build_guru_scorecard
from finance_ai.tools.recommendation_service import gather_all_financial_data


@tool
def get_guru_scorecard(
    guru_id: Annotated[str, InjectedState("guru_id")] = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Get the active guru's checklist computed from the user's real finances.

    Call this before giving personal advice. Explain each check's status
    (good / warn / bad / missing) in the guru's lens; never change the
    numbers. For "missing" checks, ask the user for that information.

    Args:
        guru_id: Active guru (injected from graph state).
        user_id: UUID of the user (injected from graph state).
        db_session_factory: Session factory (injected from graph state).

    Returns:
        Dict with guru_id and a list of checks, or 'error'.
    """
    if not guru_id:
        return {"error": "No guru selected. guru_id is missing from the conversation state."}
    today = date.today()
    with get_tool_session(db_session_factory) as session:
        data = gather_all_financial_data(session, user_id, today.year, today.month)
    risk_level = get_user_risk_level(user_id, db_session_factory)
    return build_guru_scorecard(guru_id, data, risk_level).model_dump()
