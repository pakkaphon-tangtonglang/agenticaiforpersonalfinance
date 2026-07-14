"""Orchestrator Agent that classifies user queries and routes to specialized agents.

Supports routing to Tax, Expense, Asset Monitoring, Planning, Recommendation,
and Report agents. General/unknown intents are handled by general chat.
"""

import json
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.orm import Session

from finance_ai.agents.prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    get_date_context,
)
from finance_ai.agents.schemas import OrchestratorDecision
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DECISION = OrchestratorDecision(intent="unknown", confidence=Decimal("0"))


def parse_orchestrator_response(content: Any) -> OrchestratorDecision:
    """Parse the LLM's JSON response into a OrchestratorDecision.

    Handles markdown code fences that LLMs sometimes wrap JSON in.

    Args:
        content: Raw response content from the LLM.

    Returns:
        Parsed OrchestratorDecision, or default 'unknown' on failure.

    Example:
        >>> parse_orchestrator_response('{"intent": "tax", "confidence": 0.95}')
        OrchestratorDecision(intent='tax', confidence=Decimal('0.95'))
    """
    try:
        text = str(content).strip()
        text = _strip_code_fence(text)
        data = json.loads(text)
        return OrchestratorDecision(**data)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        logger.warning("Failed to parse router response: %s", content)
        return DEFAULT_DECISION


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fence wrapping from text.

    Args:
        text: Text that may be wrapped in ```json ... ```.

    Returns:
        Text with code fences removed if present.

    Example:
        >>> _strip_code_fence('```json\\n{"a": 1}\\n```')
        '{"a": 1}'
    """
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return text


def _build_messages(
    query: str,
    chat_history: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Build message list from chat history and current query.

    Args:
        query: The user's current query.
        chat_history: Optional previous messages as (role, content) tuples.

    Returns:
        Combined message list for graph invocation.

    Example:
        >>> _build_messages("hello", [("user", "hi"), ("assistant", "hey")])
        [('user', 'hi'), ('assistant', 'hey'), ('user', 'hello')]
    """
    history = list(chat_history or [])
    history.append(("user", query))
    return history


def classify_query(
    query: str,
    chat_model: BaseChatModel | None = None,
) -> OrchestratorDecision:
    """Classify a user query into an intent category.

    Args:
        query: The user's natural language query.
        chat_model: Optional ChatModel override for testing.

    Returns:
        OrchestratorDecision with intent and confidence.

    Example:
        >>> decision = classify_query("คำนวณภาษีปี 2024")
    """
    if chat_model is None:
        from finance_ai.agents.llm_factory import create_chat_model

        chat_model = create_chat_model()
    messages = [
        SystemMessage(content=get_date_context() + ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]
    response = chat_model.invoke(messages)
    return parse_orchestrator_response(response.content)


def execute_tax_agent(
    query: str,
    chat_model: BaseChatModel | None = None,
    user_id: str = "",
    db_session_factory: Callable[[], Session] | None = None,
    chat_history: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Execute the Tax Agent for a tax-related query.

    Args:
        query: The user's tax-related query.
        chat_model: Optional ChatModel override.
        user_id: UUID of the user for cross-agent DB operations.
        db_session_factory: Optional session factory for DB access.
        chat_history: Optional previous messages for context.

    Returns:
        Dict with intent='tax' and the agent's response.

    Example:
        >>> result = execute_tax_agent("คำนวณภาษี เงินเดือน 1 ล้าน")
    """
    from finance_ai.agents.graph_cache import get_compiled_graph  # noqa: PLC0415

    graph = get_compiled_graph("tax", chat_model)
    result = graph.invoke(
        {
            "messages": _build_messages(query, chat_history),
            "user_id": user_id,
            "db_session_factory": db_session_factory,
        }
    )
    last_message = result["messages"][-1]
    return {"intent": "tax", "response": last_message.content}


def execute_expense_agent(
    query: str,
    chat_model: BaseChatModel | None = None,
    user_id: str = "",
    db_session_factory: Callable[[], Session] | None = None,
    chat_history: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Execute the Expense Agent for an expense-related query.

    Args:
        query: The user's expense-related query.
        chat_model: Optional ChatModel override.
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access.
        chat_history: Optional previous messages for context.

    Returns:
        Dict with intent='expense' and the agent's response.

    Example:
        >>> result = execute_expense_agent("จ่ายค่ากาแฟ 80 บาท")
    """
    from finance_ai.agents.graph_cache import get_compiled_graph  # noqa: PLC0415

    graph = get_compiled_graph("expense", chat_model)
    result = graph.invoke(
        {
            "messages": _build_messages(query, chat_history),
            "user_id": user_id,
            "db_session_factory": db_session_factory,
        }
    )
    last_message = result["messages"][-1]
    return {"intent": "expense", "response": last_message.content}


def execute_asset_monitoring_agent(
    query: str,
    chat_model: BaseChatModel | None = None,
    user_id: str = "",
    db_session_factory: Callable[[], Session] | None = None,
    chat_history: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Execute the Asset Monitoring Agent for an asset-related query.

    Args:
        query: The user's asset monitoring query.
        chat_model: Optional ChatModel override.
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access.
        chat_history: Optional previous messages for context.

    Returns:
        Dict with intent='asset_monitoring' and the agent's response.

    Example:
        >>> result = execute_asset_monitoring_agent("ดูพอร์ตของฉัน")
    """
    from finance_ai.agents.graph_cache import get_compiled_graph  # noqa: PLC0415

    graph = get_compiled_graph("asset_monitoring", chat_model)
    result = graph.invoke(
        {
            "messages": _build_messages(query, chat_history),
            "user_id": user_id,
            "db_session_factory": db_session_factory,
        }
    )
    last_message = result["messages"][-1]
    return {"intent": "asset_monitoring", "response": last_message.content}


def execute_planning_agent(
    query: str,
    chat_model: BaseChatModel | None = None,
    user_id: str = "",
    db_session_factory: Callable[[], Session] | None = None,
    chat_history: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Execute the Planning Agent for a planning-related query.

    Args:
        query: The user's planning-related query.
        chat_model: Optional ChatModel override.
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access.
        chat_history: Optional previous messages for context.

    Returns:
        Dict with intent='planning' and the agent's response.

    Example:
        >>> result = execute_planning_agent("อยากออมเงิน 100,000 บาท")
    """
    from finance_ai.agents.graph_cache import get_compiled_graph  # noqa: PLC0415

    graph = get_compiled_graph("planning", chat_model)
    result = graph.invoke(
        {
            "messages": _build_messages(query, chat_history),
            "user_id": user_id,
            "db_session_factory": db_session_factory,
        }
    )
    last_message = result["messages"][-1]
    return {"intent": "planning", "response": last_message.content}


def execute_recommendation_agent(
    query: str,
    chat_model: BaseChatModel | None = None,
    user_id: str = "",
    db_session_factory: Callable[[], Session] | None = None,
    chat_history: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Execute the Recommendation Agent for a recommendation query.

    Args:
        query: The user's recommendation-related query.
        chat_model: Optional ChatModel override.
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access.
        chat_history: Optional previous messages for context.

    Returns:
        Dict with intent='recommendation' and the agent's response.

    Example:
        >>> result = execute_recommendation_agent("วิเคราะห์การเงินของฉัน")
    """
    from finance_ai.agents.graph_cache import get_compiled_graph  # noqa: PLC0415

    graph = get_compiled_graph("recommendation", chat_model)
    result = graph.invoke(
        {
            "messages": _build_messages(query, chat_history),
            "user_id": user_id,
            "db_session_factory": db_session_factory,
        }
    )
    last_message = result["messages"][-1]
    return {"intent": "recommendation", "response": last_message.content}


def execute_report_agent(
    query: str,
    chat_model: BaseChatModel | None = None,
    user_id: str = "",
    db_session_factory: Callable[[], Session] | None = None,
    chat_history: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Execute the Report Agent for a financial report query.

    Args:
        query: The user's report-related query.
        chat_model: Optional ChatModel override.
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access.
        chat_history: Optional previous messages for context.

    Returns:
        Dict with intent='report' and the agent's response.

    Example:
        >>> result = execute_report_agent("สร้างรายงานการเงิน")
    """
    from finance_ai.agents.graph_cache import get_compiled_graph  # noqa: PLC0415

    graph = get_compiled_graph("report", chat_model)
    result = graph.invoke(
        {
            "messages": _build_messages(query, chat_history),
            "user_id": user_id,
            "db_session_factory": db_session_factory,
        }
    )
    last_message = result["messages"][-1]
    return {"intent": "report", "response": last_message.content}


def execute_general_chat(
    query: str,
    chat_model: BaseChatModel | None = None,
    user_id: str = "",
    db_session_factory: Callable[[], Session] | None = None,
    chat_history: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Handle general conversation using LLM directly.

    Args:
        query: The user's message.
        chat_model: Optional ChatModel override.
        user_id: Unused, kept for consistent signature.
        db_session_factory: Unused, kept for consistent signature.
        chat_history: Optional previous messages for context.

    Returns:
        Dict with intent='general_chat' and the LLM's response.

    Example:
        >>> result = execute_general_chat("สวัสดีครับ")
    """
    if chat_model is None:
        from finance_ai.agents.llm_factory import create_chat_model  # noqa: PLC0415

        chat_model = create_chat_model()
    messages: list[SystemMessage | HumanMessage] = [
        SystemMessage(content=get_date_context() + GENERAL_CHAT_SYSTEM_PROMPT),
    ]
    for role, content in chat_history or []:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(SystemMessage(content=content))
    messages.append(HumanMessage(content=query))
    response = chat_model.invoke(messages)
    return {"intent": "general_chat", "response": response.content}


def build_unsupported_response(decision: OrchestratorDecision) -> dict[str, Any]:
    """Build a response for unsupported intents.

    Args:
        decision: The router's classification decision.

    Returns:
        Dict with intent and a Thai message about the limitation.

    Example:
        >>> build_unsupported_response(OrchestratorDecision(intent="unknown", confidence=Decimal("0")))
    """
    return {
        "intent": decision.intent,
        "response": (
            "ขออภัย ขณะนี้ระบบรองรับเฉพาะคำถามเกี่ยวกับภาษี"
            " ค่าใช้จ่าย การลงทุน วางแผนการเงิน คำแนะนำการเงิน"
            " และรายงานการเงินเท่านั้น"
        ),
    }


def orchestrate_query(
    query: str,
    chat_model: BaseChatModel | None = None,
    user_id: str = "",
    db_session_factory: Callable[[], Session] | None = None,
    chat_history: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Route a user query to the appropriate agent.

    Classifies the query intent and dispatches to the matching agent.
    Passes chat_history for conversation context.

    Args:
        query: The user's natural language query.
        chat_model: Optional ChatModel override for testing.
        user_id: UUID of the user for DB-backed agents.
        db_session_factory: Optional session factory for DB access.
        chat_history: Optional previous messages for context.

    Returns:
        Dict with 'intent' and 'response' from the target agent.

    Example:
        >>> result = orchestrate_query("คำนวณภาษี เงินเดือน 1 ล้าน")
    """
    decision = classify_query(query, chat_model)
    logger.info("Routed query to: %s (confidence: %s)", decision.intent, decision.confidence)
    args = (query, chat_model, user_id, db_session_factory, chat_history)
    agent_map: dict[str, Callable[..., dict[str, Any]]] = {
        "tax": execute_tax_agent,
        "expense": execute_expense_agent,
        "asset_monitoring": execute_asset_monitoring_agent,
        "planning": execute_planning_agent,
        "general": execute_planning_agent,
        "recommendation": execute_recommendation_agent,
        "report": execute_report_agent,
        "unknown": execute_general_chat,
    }
    agent_fn = agent_map.get(decision.intent)
    if agent_fn is not None:
        return agent_fn(*args)
    return execute_general_chat(*args)
