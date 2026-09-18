"""Orchestrator Agent that classifies user queries and routes to specialized agents.

Supports routing to Tax, Expense, Asset Monitoring, Planning, Recommendation,
and Report agents. General/unknown intents are handled by general chat.
"""

import json
import re
from collections.abc import Callable
from decimal import Decimal
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy.orm import Session

from finance_ai.agents.prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
    ORCHESTRATOR_SYSTEM_PROMPT,
    get_date_context,
)
from finance_ai.agents.schemas import OrchestratorDecision
from finance_ai.core.logging import get_logger
from finance_ai.tools.symbol_search_service import search_asset_symbols

logger = get_logger(__name__)

DEFAULT_DECISION = OrchestratorDecision(intent="unknown", confidence=Decimal("0"))

# Minimum confidence required to route without asking the user back
ROUTER_CONFIDENCE_THRESHOLD = Decimal("0.7")

# Maximum number of recent history messages shown to the router LLM
_ROUTER_HISTORY_LIMIT = 6

# Uppercase ticker-like tokens (PTT, AAPL, KBANK.BK); lowercase words ignored
_ASSET_TOKEN_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{1,9}$")

# Common uppercase words that are not asset tickers
_NON_ASSET_TOKENS = {"USD", "THB", "OK", "ATM", "SMS", "HTTP", "WWW"}


def _content_to_text(content: Any) -> str:
    """Convert LLM message content into plain text.

    Handles string content plus Gemini-style content-block lists
    ([{'type': 'text', 'text': '...'}, ...]) returned by Gemini 3+.

    Args:
        content: Raw response content from the LLM.

    Returns:
        Concatenated text blocks, or str(content) as fallback.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if parts:
            return "".join(parts)
    return str(content)


def parse_orchestrator_response(content: Any) -> OrchestratorDecision:
    """Parse the LLM's JSON response into a OrchestratorDecision.

    Handles markdown code fences and Gemini-style content-block lists.

    Args:
        content: Raw response content from the LLM.

    Returns:
        Parsed OrchestratorDecision, or default 'unknown' on failure.

    Example:
        >>> parse_orchestrator_response('{"intent": "tax", "confidence": 0.95}')
        OrchestratorDecision(intent='tax', confidence=Decimal('0.95'))
    """
    try:
        text = _content_to_text(content).strip()
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
    chat_history: list[tuple[str, str]] | None = None,
) -> OrchestratorDecision:
    """Classify a user query into an intent category.

    Uses recent chat history for context and resolves possible asset
    mentions (ticker-like tokens) before classification.

    Args:
        query: The user's natural language query.
        chat_model: Optional ChatModel override for testing.
        chat_history: Optional recent (role, content) messages for context.

    Returns:
        OrchestratorDecision with intent and confidence.

    Example:
        >>> decision = classify_query("คำนวณภาษีปี 2024")
    """
    if chat_model is None:
        from finance_ai.agents.llm_factory import create_chat_model

        chat_model = create_chat_model()
    asset_hint = _resolve_asset_hint(query)
    messages = _build_router_messages(query, chat_history, asset_hint)
    response = chat_model.invoke(messages)
    return parse_orchestrator_response(response.content)


def _extract_asset_candidate_tokens(query: str) -> list[str]:
    """Extract uppercase ticker-like tokens from a query.

    Args:
        query: The user's query text.

    Returns:
        Up to 2 unique candidate tokens (e.g. "PTT", "AAPL").
    """
    tokens = re.findall(r"[A-Za-z0-9.\-]+", query)
    candidates: list[str] = []
    for token in tokens:
        if not _ASSET_TOKEN_PATTERN.match(token):
            continue
        if token in _NON_ASSET_TOKENS or token in candidates:
            continue
        candidates.append(token)
    return candidates[:2]


def _resolve_asset_hint(query: str) -> str | None:
    """Resolve a possible asset mention into a routing hint.

    Searches Yahoo Finance for uppercase ticker-like tokens in the query.
    Thai-only or lowercase-only queries skip the search entirely.

    Args:
        query: The user's query text.

    Returns:
        Thai hint string naming the resolved asset, or None.

    Example:
        >>> _resolve_asset_hint("จ่ายค่ากาแฟ 80 บาท") is None
        True
    """
    for token in _extract_asset_candidate_tokens(query):
        matches = search_asset_symbols(token)
        if matches:
            best = matches[0]
            return f"ผู้ใช้อาจกล่าวถึงหลักทรัพย์: {best.symbol} ({best.name})"
    return None


def _build_router_messages(
    query: str,
    chat_history: list[tuple[str, str]] | None = None,
    asset_hint: str | None = None,
) -> list[BaseMessage]:
    """Build the router LLM message list.

    Args:
        query: The user's current query.
        chat_history: Optional recent messages for context.
        asset_hint: Optional resolved asset symbol hint.

    Returns:
        System prompt, trimmed history, optional hint, current query.
    """
    messages: list[BaseMessage] = [
        SystemMessage(content=get_date_context() + ORCHESTRATOR_SYSTEM_PROMPT)
    ]
    for role, content in (chat_history or [])[-_ROUTER_HISTORY_LIMIT:]:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    if asset_hint:
        messages.append(SystemMessage(content=asset_hint))
    messages.append(HumanMessage(content=query))
    return messages


def _build_clarify_response() -> dict[str, Any]:
    """Build the Thai clarify-back response for low-confidence routing.

    Returns:
        Dict with intent='clarify' and a Thai question listing options.
    """
    return {
        "intent": "clarify",
        "response": (
            "ไม่แน่ใจว่าเข้าใจถูกไหมคะ ช่วยเลือกหัวข้อให้ชัดขึ้นได้ไหมคะ\n"
            "• บันทึกรายรับ–รายจ่าย เช่น จ่ายค่ากาแฟ 80 บาท\n"
            "• วางแผนเป้าหมาย เช่น อยากออมเงิน 100,000 บาท\n"
            "• หุ้น/กองทุน เช่น ดูราคา PTT\n"
            "• ภาษี เช่น คำนวณภาษีปี 2024\n"
            "หรือพิมพ์รายละเอียดเพิ่มอีกนิดก็ได้คะ"
        ),
    }


def _should_clarify(decision: OrchestratorDecision) -> bool:
    """Check whether a routing decision is too unsure to act on.

    Unknown intent (non-finance chatter) never asks back — it goes to
    general chat, which can answer anything.

    Args:
        decision: The router's classification decision.

    Returns:
        True when confidence is below the routing threshold.
    """
    if decision.intent == "unknown":
        return False
    return decision.confidence < ROUTER_CONFIDENCE_THRESHOLD


def _invoke_intent(
    intent: str,
    query: str,
    chat_model: BaseChatModel | None,
    user_id: str,
    db_session_factory: Callable[[], Session] | None,
    chat_history: list[tuple[str, str]] | None,
) -> dict[str, Any]:
    """Build, invoke, and return the response for a single specialist agent.

    Args:
        intent: Agent intent key (tax, expense, etc.).
        query: The user's query.
        chat_model: Optional ChatModel override.
        user_id: UUID of the user for DB operations.
        db_session_factory: Optional session factory for DB access.
        chat_history: Optional previous messages for context.

    Returns:
        Dict with the intent and the agent's last message content.

    Raises:
        ValueError: If the intent has no registered graph builder.
    """
    from finance_ai.agents.graph_cache import get_compiled_graph  # noqa: PLC0415

    graph = get_compiled_graph(intent, chat_model)
    if graph is None:
        raise ValueError(f"Unknown agent intent: {intent}")
    result = graph.invoke(
        {
            "messages": _build_messages(query, chat_history),
            "user_id": user_id,
            "db_session_factory": db_session_factory,
        }
    )
    last_message = result["messages"][-1]
    return {"intent": intent, "response": last_message.content}


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
    return _invoke_intent("tax", query, chat_model, user_id, db_session_factory, chat_history)


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
    return _invoke_intent("expense", query, chat_model, user_id, db_session_factory, chat_history)


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
    return _invoke_intent(
        "asset_monitoring", query, chat_model, user_id, db_session_factory, chat_history
    )


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
    return _invoke_intent("planning", query, chat_model, user_id, db_session_factory, chat_history)


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
    return _invoke_intent(
        "recommendation", query, chat_model, user_id, db_session_factory, chat_history
    )


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
    return _invoke_intent("report", query, chat_model, user_id, db_session_factory, chat_history)


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


def _build_agent_map() -> dict[str, Callable[..., dict[str, Any]]]:
    """Map intent keys to their executor functions.

    Returns:
        Dict from intent string to executor callable.
    """
    return {
        "tax": execute_tax_agent,
        "expense": execute_expense_agent,
        "asset_monitoring": execute_asset_monitoring_agent,
        "planning": execute_planning_agent,
        "general": execute_general_chat,
        "recommendation": execute_recommendation_agent,
        "report": execute_report_agent,
        "unknown": execute_general_chat,
    }


def orchestrate_query(
    query: str,
    chat_model: BaseChatModel | None = None,
    user_id: str = "",
    db_session_factory: Callable[[], Session] | None = None,
    chat_history: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Route a user query to the appropriate agent.

    Classifies the query intent (with chat history and asset-hint context)
    and dispatches to the matching agent. Low-confidence classifications
    ask the user for clarification instead of guessing. Passes
    chat_history for conversation context.

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
    decision = classify_query(query, chat_model, chat_history)
    logger.info("Routed query to: %s (confidence: %s)", decision.intent, decision.confidence)
    if _should_clarify(decision):
        logger.info("Low confidence - asking user for clarification")
        return _build_clarify_response()
    agent_map = _build_agent_map()
    default_fn: Callable[..., dict[str, Any]] = execute_general_chat
    agent_fn = agent_map.get(decision.intent, default_fn)
    result = agent_fn(query, chat_model, user_id, db_session_factory, chat_history)
    result["response"] = _content_to_text(result.get("response", ""))
    return result
