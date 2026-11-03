"""Tests for per-run LLM call, token and tool-call counting."""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from finance_ai.evaluation.architectures.run_metrics import (
    RunMetrics,
    collect_run_metrics,
)


@tool
def lookup_price(symbol: str) -> str:
    """Return a fixed price for a symbol."""
    return f"{symbol}: 32.50"


def _fake_model(*replies: AIMessage) -> GenericFakeChatModel:
    """Build a fake chat model that returns the given replies in order."""
    return GenericFakeChatModel(messages=iter(replies))


class TestCollectRunMetrics:
    """collect_run_metrics counts what happens inside its block."""

    def test_counts_llm_calls_and_tokens(self) -> None:
        """Counts llm calls and tokens."""
        usage: Any = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}
        model = _fake_model(
            AIMessage(content="a", usage_metadata=usage),
            AIMessage(content="b", usage_metadata=usage),
        )
        with collect_run_metrics() as metrics:
            model.invoke("hi")
            model.invoke("again")
        assert metrics.llm_calls == 2
        assert metrics.input_tokens == 240
        assert metrics.output_tokens == 60

    def test_counts_calls_without_usage_metadata(self) -> None:
        """Counts calls without usage metadata."""
        model = _fake_model(AIMessage(content="no usage"))
        with collect_run_metrics() as metrics:
            model.invoke("hi")
        assert metrics.llm_calls == 1
        assert metrics.input_tokens == 0

    def test_records_tool_calls_with_arguments(self) -> None:
        """Records tool calls with arguments."""
        with collect_run_metrics() as metrics:
            lookup_price.invoke({"symbol": "PTT.BK"})
        assert [call.name for call in metrics.tool_calls] == ["lookup_price"]
        assert metrics.tool_calls[0].args == {"symbol": "PTT.BK"}

    def test_nothing_counted_outside_the_block(self) -> None:
        """Nothing counted outside the block."""
        model = _fake_model(AIMessage(content="x"), AIMessage(content="y"))
        with collect_run_metrics() as metrics:
            model.invoke("inside")
        model.invoke("outside")
        assert metrics.llm_calls == 1

    def test_nested_blocks_are_independent(self) -> None:
        """Nested blocks are independent."""
        model = _fake_model(AIMessage(content="x"), AIMessage(content="y"))
        with collect_run_metrics() as outer:
            model.invoke("outer")
            with collect_run_metrics() as inner:
                model.invoke("inner")
        assert inner.llm_calls == 1
        assert outer.llm_calls == 1


class TestRunMetrics:
    """RunMetrics helpers."""

    def test_tool_names_keeps_order(self) -> None:
        """Tool names keeps order."""
        metrics = RunMetrics()
        metrics.record_tool_call("a", {})
        metrics.record_tool_call("b", {"x": 1})
        assert metrics.tool_names() == ["a", "b"]

    def test_starts_empty(self) -> None:
        """Starts empty."""
        metrics = RunMetrics()
        assert (metrics.llm_calls, metrics.input_tokens, metrics.output_tokens) == (0, 0, 0)
        assert not metrics.tool_calls
