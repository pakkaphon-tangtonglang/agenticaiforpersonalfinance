"""Orchestrator Agent that classifies user queries and routes to specialized agents.

Supports routing to Tax, Expense, Asset Monitoring, Planning, Recommendation,
and Report agents. General/unknown intents are handled by general chat.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError
from sqlalchemy.orm import Session

from finance_ai.agents.llm_error_messages import describe_llm_error
from finance_ai.agents.llm_factory import router_chat_model
from finance_ai.agents.multi_intent import (
    TaskSection,
    combine_sections,
    merge_tasks_by_agent,
    run_tasks,
    section_title,
    split_by_confidence,
)
from finance_ai.agents.prompts import (
    GENERAL_CHAT_SYSTEM_PROMPT,
    ORCHESTRATOR_HISTORY_INSTRUCTION,
    ORCHESTRATOR_PREVIOUS_TOPICS_HINT,
    build_orchestrator_system_prompt,
    get_date_context,
)
from finance_ai.agents.schemas import OrchestratorDecision, RoutedTask
from finance_ai.agents.tool_trace import ToolTrace, traces_from_messages
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


@dataclass(frozen=True)
class RouterAblationConfig:
    """Feature toggles for router ablation experiments.

    Defaults reproduce production behavior exactly. Evaluation runs
    flip one flag at a time to isolate each feature's contribution.
    The confidence threshold defaults to False because production
    applies it in orchestrate_query, not in classify_query.

    Attributes:
        include_chat_history: Pass recent messages to the router LLM.
        include_few_shot_examples: Include the prompt's example block.
        resolve_asset_hint: Pre-search uppercase tokens on Yahoo.
        apply_confidence_threshold: Downgrade low-confidence to clarify.
    """

    include_chat_history: bool = True
    include_few_shot_examples: bool = True
    resolve_asset_hint: bool = True
    apply_confidence_threshold: bool = False


DEFAULT_ABLATION_CONFIG = RouterAblationConfig()


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
        raw_tasks = data.pop("tasks", [])
        decision = OrchestratorDecision(**data)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError):
        logger.warning("Failed to parse router response: %s", content)
        return DEFAULT_DECISION
    return decision.model_copy(update={"tasks": parse_routed_tasks(raw_tasks)})


def parse_routed_tasks(raw_tasks: Any) -> list[RoutedTask]:
    """Validate each part of a multi-topic reply on its own.

    One malformed part is dropped instead of discarding the whole decision.

    Args:
        raw_tasks: The ``tasks`` value from the router JSON.

    Returns:
        Valid tasks in the router's order ([] when absent or not a list).

    Example:
        >>> raw = [{"intent": "tax", "sub_query": "ภาษี", "confidence": 0.9}]
        >>> parse_routed_tasks(raw)[0].intent
        'tax'
    """
    if not isinstance(raw_tasks, list):
        return []
    tasks: list[RoutedTask] = []
    for item in raw_tasks:
        try:
            tasks.append(RoutedTask.model_validate(item))
        except ValidationError:
            logger.warning("Dropped invalid router task: %s", item)
    return tasks


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
    config: RouterAblationConfig | None = None,
    previous_intents: list[str] | None = None,
) -> OrchestratorDecision:
    """Classify a user query into an intent category.

    Uses recent chat history for context and resolves possible asset
    mentions (ticker-like tokens) before classification. Ablation
    runs can disable each helper via config.

    Args:
        query: The user's natural language query.
        chat_model: Optional ChatModel override for testing.
        chat_history: Optional recent (role, content) messages for context.
        config: Optional ablation toggles; None uses production defaults.
        previous_intents: Topics of the previous reply when it covered several.

    Returns:
        OrchestratorDecision with intent, confidence and any multi-topic tasks.

    Example:
        >>> decision = classify_query("คำนวณภาษีปี 2024")
    """
    if chat_model is None:
        from finance_ai.agents.llm_factory import create_chat_model

        chat_model = create_chat_model()
    settings = config or DEFAULT_ABLATION_CONFIG
    asset_hint = _resolve_asset_hint(query) if settings.resolve_asset_hint else None
    messages = _build_router_messages(query, chat_history, asset_hint, settings)
    topics_hint = previous_topics_hint(previous_intents)
    if topics_hint:
        messages.insert(-1, SystemMessage(content=topics_hint))
    response = router_chat_model(chat_model).invoke(messages)
    decision = parse_orchestrator_response(response.content)
    return _apply_confidence_threshold(decision, settings)


def previous_topics_hint(previous_intents: list[str] | None) -> str | None:
    """Describe the previous reply's topics when it answered several.

    Lets a follow-up such as "แล้วเรื่องภาษีล่ะ" route to the right agent.
    Single-topic replies add nothing, so single-topic routing is unchanged.

    Args:
        previous_intents: Agents that answered the previous message.

    Returns:
        Thai hint for the router, or None.

    Example:
        >>> previous_topics_hint(["tax"]) is None
        True
    """
    if not previous_intents or len(previous_intents) < 2:
        return None
    topics = ", ".join(f"{section_title(intent)} ({intent})" for intent in previous_intents)
    return ORCHESTRATOR_PREVIOUS_TOPICS_HINT.format(topics=topics)


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


def _history_messages(
    chat_history: list[tuple[str, str]] | None,
) -> list[BaseMessage]:
    """Convert trimmed (role, content) history into LangChain messages.

    Args:
        chat_history: Optional recent messages.

    Returns:
        Up to _ROUTER_HISTORY_LIMIT HumanMessage/AIMessage objects
        followed by a closing instruction SystemMessage. The instruction
        (placed last, right before the current query) tells the model the
        history is context only, keeping it in classifier (JSON) mode
        instead of answering the conversation directly.
    """
    trimmed = (chat_history or [])[-_ROUTER_HISTORY_LIMIT:]
    if not trimmed:
        return []
    messages: list[BaseMessage] = []
    for role, content in trimmed:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    messages.append(SystemMessage(content=ORCHESTRATOR_HISTORY_INSTRUCTION))
    return messages


def _build_router_messages(
    query: str,
    chat_history: list[tuple[str, str]] | None = None,
    asset_hint: str | None = None,
    config: RouterAblationConfig | None = None,
) -> list[BaseMessage]:
    """Build the router LLM message list.

    Args:
        query: The user's current query.
        chat_history: Optional recent messages for context.
        asset_hint: Optional resolved asset symbol hint.
        config: Optional ablation toggles; None uses production defaults.

    Returns:
        System prompt, optional history, optional hint, current query.
    """
    settings = config or DEFAULT_ABLATION_CONFIG
    system_prompt = build_orchestrator_system_prompt(settings.include_few_shot_examples)
    messages: list[BaseMessage] = [SystemMessage(content=get_date_context() + system_prompt)]
    if settings.include_chat_history:
        messages.extend(_history_messages(chat_history))
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


def _apply_confidence_threshold(
    decision: OrchestratorDecision,
    config: RouterAblationConfig,
) -> OrchestratorDecision:
    """Downgrade low-confidence decisions to 'clarify' when enabled.

    Mirrors orchestrate_query's clarify-back rule for ablation runs.

    Args:
        decision: The parsed router decision.
        config: Ablation toggles for the current run.

    Returns:
        The original decision, or a 'clarify' decision when the
        threshold is enabled and confidence is below the threshold.
    """
    if config.apply_confidence_threshold and _should_clarify(decision):
        # Keep the parts: a multi-topic message is judged per part, not by the first one.
        return decision.model_copy(update={"intent": "clarify"})
    return decision


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
        Dict with the intent, the agent's last message content and its tool calls.

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
    return {
        "intent": intent,
        "response": last_message.text,
        "tools": traces_from_messages(list(result["messages"])),
    }


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
    return {"intent": "general_chat", "response": response.text}


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
    previous_intents: list[str] | None = None,
    allow_multi_intent: bool = True,
) -> dict[str, Any]:
    """Route a user query to the appropriate agent (or agents).

    Classifies the query intent (with chat history and asset-hint context)
    and dispatches to the matching agent. Low-confidence classifications
    ask the user for clarification instead of guessing. Passes
    chat_history for conversation context. A message about several topics
    runs one agent per topic and returns one combined reply.

    Args:
        query: The user's natural language query.
        chat_model: Optional ChatModel override for testing.
        user_id: UUID of the user for DB-backed agents.
        db_session_factory: Optional session factory for DB access.
        chat_history: Optional previous messages for context.
        previous_intents: Topics of the previous reply when it covered several.
        allow_multi_intent: False sends every message to one agent, as before
            multi-intent routing (used by the architecture study baseline).

    Returns:
        Dict with 'intent' and 'response'; multi-topic replies also carry
        'intents' (every agent that answered, in order).

    Example:
        >>> result = orchestrate_query("คำนวณภาษี เงินเดือน 1 ล้าน")
    """
    try:
        context = RunContext(chat_model, user_id, db_session_factory, chat_history)
        return _route_query(query, context, previous_intents, allow_multi_intent)
    except Exception as exc:  # noqa: BLE001
        logger.error("Query orchestration failed: %s", exc, exc_info=True)
        return {"intent": "error", "response": describe_llm_error(exc)}


@dataclass(frozen=True)
class RunContext:
    """What every agent in one request shares.

    Example:
        >>> RunContext(None, "user-1", None, None).user_id
        'user-1'
    """

    chat_model: BaseChatModel | None
    user_id: str
    db_session_factory: Callable[[], Session] | None
    chat_history: list[tuple[str, str]] | None


def _route_query(
    query: str,
    context: RunContext,
    previous_intents: list[str] | None = None,
    allow_multi_intent: bool = True,
) -> dict[str, Any]:
    """Classify a query and run the matching agent or agents (errors propagate).

    Returns:
        Dict with 'intent' and 'response' (plus 'intents' for several topics).
    """
    decision = classify_query(
        query, context.chat_model, context.chat_history, previous_intents=previous_intents
    )
    logger.info("Routed query to: %s (confidence: %s)", decision.intent, decision.confidence)
    tasks = merge_tasks_by_agent(decision.tasks) if allow_multi_intent else []
    if len(tasks) >= 2:
        return _route_multi_intent(query, tasks, context)
    if _should_clarify(decision):
        logger.info("Low confidence - asking user for clarification")
        return _build_clarify_response()
    agent_fn: Callable[..., dict[str, Any]] = (
        _build_agent_map().get(decision.intent) or execute_general_chat
    )
    result = agent_fn(
        query,
        context.chat_model,
        context.user_id,
        context.db_session_factory,
        context.chat_history,
    )
    result["response"] = _content_to_text(result.get("response", ""))
    result.pop("tools", None)
    return result


def _route_multi_intent(query: str, tasks: list[RoutedTask], context: RunContext) -> dict[str, Any]:
    """Answer every confident part, then join the answers into one reply.

    Args:
        query: The user's whole message.
        tasks: Merged tasks (two or more agents).
        context: Shared request context.

    Returns:
        Dict with the first 'intent', every answering agent in 'intents',
        and the combined 'response'.
    """
    confident, unsure = split_by_confidence(tasks)
    logger.info("Multi-intent message: %s", [task.intent for task in tasks])
    if not confident:
        return _build_clarify_response()
    sections = run_tasks(
        confident, query, lambda intent, question: run_agent(intent, question, context)
    )
    return _multi_intent_result(sections, combine_sections(sections, unsure))


def run_agent(intent: str, question: str, context: RunContext) -> tuple[str, list[ToolTrace]]:
    """Run one agent (general chat for unknown intents) and return its answer and tool calls.

    Args:
        intent: Agent key from the router.
        question: Question for the agent.
        context: Shared request context.

    Returns:
        (answer text, tool calls).

    Example:
        >>> run_agent("tax", "ภาษีปีนี้", context)  # doctest: +SKIP
    """
    agent_fn: Callable[..., dict[str, Any]] = _build_agent_map().get(intent) or execute_general_chat
    result = agent_fn(
        question,
        context.chat_model,
        context.user_id,
        context.db_session_factory,
        context.chat_history,
    )
    return _content_to_text(result.get("response", "")), list(result.get("tools", []))


def _multi_intent_result(sections: list[TaskSection], response: str) -> dict[str, Any]:
    """Build the orchestrator result for a multi-topic reply."""
    intents = [section.intent for section in sections]
    return {"intent": intents[0], "intents": intents, "response": response}
