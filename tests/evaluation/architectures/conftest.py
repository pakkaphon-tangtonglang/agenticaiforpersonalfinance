"""Shared fakes for the architecture design tests (no network)."""

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from finance_ai.evaluation.architectures.design_context import DesignContext
from finance_ai.evaluation.architectures.run_metrics import RunMetrics
from finance_ai.evaluation.architectures.run_result import ArchitectureRun, DesignName


class ToolFakeChatModel(GenericFakeChatModel):
    """Fake chat model that also accepts bind_tools (returns itself)."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ToolFakeChatModel":
        """Ignore tools and return the same fake."""
        return self


def fake_model(*replies: str | AIMessage) -> ToolFakeChatModel:
    """Build a fake model replying with the given texts or messages in order."""
    messages = [
        reply if isinstance(reply, AIMessage) else AIMessage(content=reply) for reply in replies
    ]
    return ToolFakeChatModel(messages=iter(messages))


@pytest.fixture(name="make_context")
def fixture_make_context() -> Callable[..., DesignContext]:
    """Return a factory for DesignContext objects around a fake model."""

    def make(*replies: str | AIMessage, task_index: int = 0) -> DesignContext:
        return DesignContext(
            chat_model=fake_model(*replies),
            user_id="demo-user",
            db_session_factory=None,
            task_id="task",
            task_index=task_index,
        )

    return make


def make_run(  # pylint: disable=too-many-arguments
    answer: str = "",
    route: list[str] | None = None,
    tools: list[str] | None = None,
    design: DesignName = "hub_spoke",
    *,
    task_id: str = "x",
    latency: float = 0.0,
    error: str | None = None,
) -> ArchitectureRun:
    """Build a run record with the given answer, route, tool calls and latency."""
    metrics = RunMetrics(llm_calls=2, input_tokens=100, output_tokens=20)
    for name in tools or []:
        metrics.record_tool_call(name, {})
    return ArchitectureRun(
        task_id=task_id,
        design=design,
        round_index=0,
        answer=answer,
        route=route or [],
        metrics=metrics,
        latency_seconds=latency,
        error=error,
    )
