"""Shared helpers for router study tests."""

from collections.abc import Iterator
from typing import Any, Optional
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from finance_ai.agents import router_agent
from finance_ai.evaluation.router_thinking.questions import Destination, RoutingQuestion
from finance_ai.evaluation.router_thinking.runner import RouterCallRow


def fake_router_model(
    reply: str = '{"intent": "tax", "confidence": 0.9}', thinking_tokens: int = 0
) -> Runnable[Any, Any]:
    """A router model that always gives the same JSON reply with usage metadata."""
    usage: dict[str, Any] = {"input_tokens": 1200, "output_tokens": 15, "total_tokens": 1215}
    if thinking_tokens:
        usage["output_token_details"] = {"reasoning": thinking_tokens}
    return RunnableLambda(lambda _messages: AIMessage(content=reply, usage_metadata=usage))


def make_question(
    question_id: str = "tax_01", expected: Destination = "tax", query: str = "ภาษีปีนี้"
) -> RoutingQuestion:
    """A routing question."""
    return RoutingQuestion(question_id=question_id, query=query, expected=expected)


def make_row(  # noqa: PLR0913
    setup: str = "flash_minimal",
    question_id: str = "tax_01",
    correct: bool = True,
    latency: float = 1.0,
    source: str = "new",
    expected: str = "tax",
    error: Optional[str] = None,
) -> RouterCallRow:
    """A logged router call."""
    return RouterCallRow(
        setup=setup,
        model="gemini-3.5-flash",
        thinking_level="minimal",
        round_index=0,
        question_id=question_id,
        source=source,
        expected=expected,
        predicted_intent=expected if correct else "expense",
        destination=expected if correct else "expense",
        correct=correct,
        latency_seconds=latency,
        input_tokens=1200,
        output_tokens=15,
        error=error,
    )


@pytest.fixture(autouse=True)
def no_yahoo() -> Iterator[None]:
    """Never call Yahoo Finance from tests."""
    with patch.object(router_agent, "_resolve_asset_hint", lambda _query: None):
        yield
