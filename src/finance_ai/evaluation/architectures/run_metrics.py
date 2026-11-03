"""Count LLM calls, tokens and tool calls made inside one study run.

Uses LangChain's configure hook (the same mechanism as
``get_usage_metadata_callback``), so every chat model and tool invoked
inside the ``with`` block reports here without changing production code.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.tracers.context import register_configure_hook
from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    """One tool call made during a run.

    Example:
        >>> ToolCallRecord(name="calculate_thai_tax", args={"gross_income": 600000}).name
        'calculate_thai_tax'
    """

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    """Counters for one run of one task through one architecture.

    Example:
        >>> metrics = RunMetrics()
        >>> metrics.record_tool_call("add_expense", {"amount": "80"})
        >>> metrics.tool_names()
        ['add_expense']
    """

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[ToolCallRecord] = []

    def record_tool_call(self, name: str, args: dict[str, Any]) -> None:
        """Append a tool call.

        Args:
            name: Tool name.
            args: Arguments the tool received.
        """
        self.tool_calls.append(ToolCallRecord(name=name, args=args))

    def tool_names(self) -> list[str]:
        """Return the called tool names in call order.

        Returns:
            Tool names, duplicates kept.
        """
        return [call.name for call in self.tool_calls]


class RunMetricsHandler(BaseCallbackHandler):
    """Callback handler that fills a RunMetrics object."""

    def __init__(self, metrics: RunMetrics) -> None:
        """Store the metrics object to update.

        Args:
            metrics: Counters to fill.
        """
        self.metrics = metrics

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Count one model call and its token usage.

        Args:
            response: Model result; chat generations carry usage_metadata.
            **kwargs: Unused callback arguments.
        """
        self.metrics.llm_calls += 1
        input_tokens, output_tokens = _usage_from_result(response)
        self.metrics.input_tokens += input_tokens
        self.metrics.output_tokens += output_tokens

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Record a tool call with its arguments.

        Args:
            serialized: Tool description; holds the tool name.
            input_str: Raw input text (used only when inputs is missing).
            inputs: Parsed tool arguments.
            **kwargs: Unused callback arguments.
        """
        name = str((serialized or {}).get("name") or kwargs.get("name") or "unknown")
        args = dict(inputs) if inputs is not None else {"input": input_str}
        self.metrics.record_tool_call(name, args)


def _usage_from_result(response: LLMResult) -> tuple[int, int]:
    """Sum input and output tokens over a model result.

    Args:
        response: Model result.

    Returns:
        (input_tokens, output_tokens); zeros when the provider sends none.
    """
    input_tokens = output_tokens = 0
    for generations in response.generations:
        for generation in generations:
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None)
            if usage:
                input_tokens += int(usage.get("input_tokens", 0))
                output_tokens += int(usage.get("output_tokens", 0))
    return input_tokens, output_tokens


_ACTIVE_HANDLER: ContextVar[Optional[RunMetricsHandler]] = ContextVar(
    "architecture_run_metrics", default=None
)
register_configure_hook(_ACTIVE_HANDLER, inheritable=True)


@contextmanager
def collect_run_metrics() -> Iterator[RunMetrics]:
    """Collect LLM and tool counters for everything run inside the block.

    Yields:
        RunMetrics filled while the block runs.

    Example:
        >>> with collect_run_metrics() as metrics:
        ...     pass
        >>> metrics.llm_calls
        0
    """
    metrics = RunMetrics()
    token = _ACTIVE_HANDLER.set(RunMetricsHandler(metrics))
    try:
        yield metrics
    finally:
        _ACTIVE_HANDLER.reset(token)
