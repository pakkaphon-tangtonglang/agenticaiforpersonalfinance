"""Shared inputs and outputs for the four architecture designs."""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from sqlalchemy.orm import Session

from finance_ai.agents.router_agent import (
    execute_asset_monitoring_agent,
    execute_expense_agent,
    execute_general_chat,
    execute_planning_agent,
    execute_recommendation_agent,
    execute_report_agent,
    execute_tax_agent,
)

GENERAL_AGENT = "general"
CONFIDENCE_THRESHOLD = Decimal("0.7")

_EXECUTORS: dict[str, Callable[..., dict[str, Any]]] = {
    "tax": execute_tax_agent,
    "expense": execute_expense_agent,
    "asset_monitoring": execute_asset_monitoring_agent,
    "planning": execute_planning_agent,
    "recommendation": execute_recommendation_agent,
    "report": execute_report_agent,
    GENERAL_AGENT: execute_general_chat,
}


class DesignError(RuntimeError):
    """Raised when a design cannot produce an answer (e.g. provider failure)."""


@dataclass(frozen=True)
class DesignContext:
    """Everything a design needs to answer one task.

    Attributes:
        chat_model: Model used for routing and agents (same for all designs).
        user_id: Demo user id.
        db_session_factory: Session factory for the per-run database copy.
        task_id: Task id (for logs).
        task_index: Task position; peer-to-peer rotates its entry agent with it.
    """

    chat_model: BaseChatModel
    user_id: str
    db_session_factory: Optional[Callable[[], Session]]
    task_id: str = ""
    task_index: int = 0


@dataclass(frozen=True)
class DesignOutcome:
    """What a design returns: the answer and the agents it went through."""

    answer: str
    route: list[str]


def run_specialist(agent: str, query: str, context: DesignContext) -> str:
    """Run one production specialist agent (or general chat) on a query.

    Args:
        agent: Agent key; unknown keys fall back to general chat.
        query: User question.
        context: Model, user and database for the run.

    Returns:
        The agent's answer text.

    Example:
        >>> run_specialist("tax", "คำนวณภาษี", context)  # doctest: +SKIP
    """
    executor: Callable[..., dict[str, Any]] = _EXECUTORS.get(agent) or execute_general_chat
    result = executor(query, context.chat_model, context.user_id, context.db_session_factory)
    return str(result.get("response", ""))


def content_to_text(content: Any) -> str:
    """Flatten model reply content (string or content blocks) into text.

    Args:
        content: ``message.content`` from any provider.

    Returns:
        Plain text.

    Example:
        >>> content_to_text([{"type": "text", "text": "hi"}])
        'hi'
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def parse_json_reply(content: Any) -> dict[str, Any]:
    """Parse the first JSON object in a model reply.

    Args:
        content: Model reply content (string or content blocks).

    Returns:
        Parsed dict, or an empty dict when no valid JSON object is found.

    Example:
        >>> parse_json_reply('```json\\n{"group": "operations"}\\n```')
        {'group': 'operations'}
    """
    match = re.search(r"\{.*?\}", content_to_text(content), re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_confidence(value: Any) -> Decimal:
    """Turn a confidence value from a model reply into a Decimal in [0, 1].

    Args:
        value: Raw value (number or string).

    Returns:
        Decimal confidence; 0 when missing or invalid.

    Example:
        >>> parse_confidence("0.85")
        Decimal('0.85')
    """
    try:
        confidence = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")
    return min(max(confidence, Decimal("0")), Decimal("1"))
