"""Short summaries of an agent's tool calls, shown as tool chips in the chat.

The stream sees each tool call twice: once as the model's request
(name + args, possibly split across chunks) and once as the tool's
reply (ToolMessage). ToolCallTracker joins the two into a ToolTrace.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from pydantic import BaseModel, Field

MAX_PREVIEW_LINES = 3
MAX_VALUE_LENGTH = 120
FALLBACK_TOOL_NAME = "tool"
EMPTY_VALUES: tuple[Any, ...] = (None, "", [], {})


class ToolTrace(BaseModel):
    """One finished tool call, ready for the chat UI.

    Attributes:
        call_id: Tool call id from the model.
        name: Tool name, e.g. "calculate_thai_tax".
        args: Model-supplied arguments as display strings.
        preview: First lines of the tool's reply.
        failed: True when the tool returned an error.

    Example:
        >>> ToolTrace(call_id="c1", name="get_stock_price").failed
        False
    """

    call_id: str = ""
    name: str = FALLBACK_TOOL_NAME
    args: dict[str, str] = Field(default_factory=dict)
    preview: list[str] = Field(default_factory=list)
    failed: bool = False


def _shorten(text: str) -> str:
    """Cut text to MAX_VALUE_LENGTH characters, ending with an ellipsis."""
    if len(text) <= MAX_VALUE_LENGTH:
        return text
    return text[: MAX_VALUE_LENGTH - 1] + "…"


def _display_value(value: Any) -> str:
    """Turn one argument or JSON value into a short display string."""
    if isinstance(value, (list, dict)):
        return _shorten(json.dumps(value, ensure_ascii=False))
    return _shorten(str(value))


def format_argument_values(args: dict[str, Any]) -> dict[str, str]:
    """Format tool arguments for display, dropping empty ones.

    Args:
        args: Raw tool-call arguments.

    Returns:
        Argument name → short display string.

    Example:
        >>> format_argument_values({"amount": 120, "note": ""})
        {'amount': '120'}
    """
    return {key: _display_value(value) for key, value in args.items() if value not in EMPTY_VALUES}


def _json_scalar_lines(data: dict[str, Any]) -> list[str]:
    """List `key: value` lines for the simple fields of a JSON object."""
    return [
        f"{key}: {_display_value(value)}"
        for key, value in data.items()
        if isinstance(value, (str, int, float, bool))
    ]


def summarize_tool_result(text: str) -> list[str]:
    """Pick the first few readable lines of a tool's reply.

    Args:
        text: The tool's reply text (plain text or JSON).

    Returns:
        At most MAX_PREVIEW_LINES short lines.

    Example:
        >>> summarize_tool_result('{"total": "100.00"}')
        ['total: 100.00']
    """
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    lines = _json_scalar_lines(data) if isinstance(data, dict) else []
    if not lines:
        lines = [_shorten(line.strip()) for line in text.splitlines() if line.strip()]
    return lines[:MAX_PREVIEW_LINES]


class ToolCallTracker:
    """Join tool-call requests with tool replies during one stream.

    Example:
        >>> tracker = ToolCallTracker()
        >>> tracker.trace_for(ToolMessage(content="ok", tool_call_id="c1")).name
        'tool'
    """

    def __init__(self) -> None:
        """Start with no pending chunks and no known calls."""
        self._pending_chunk: AIMessageChunk | None = None
        self._calls_by_id: dict[str, dict[str, Any]] = {}

    def observe(self, message: Any) -> None:
        """Record tool-call requests from a streamed message.

        Args:
            message: Any item from the LangGraph message stream.
        """
        if isinstance(message, AIMessageChunk):
            if message.tool_call_chunks:
                pending = self._pending_chunk
                self._pending_chunk = message if pending is None else pending + message
            return
        if isinstance(message, AIMessage):
            self._remember_calls(message.tool_calls)

    def trace_for(self, message: ToolMessage) -> ToolTrace:
        """Build the trace for a finished tool call.

        Args:
            message: The tool's reply.

        Returns:
            ToolTrace with the call's name, args and reply preview.
        """
        self._flush_pending_chunk()
        call = self._calls_by_id.get(message.tool_call_id, {})
        return ToolTrace(
            call_id=message.tool_call_id,
            name=call.get("name") or message.name or FALLBACK_TOOL_NAME,
            args=format_argument_values(call.get("args") or {}),
            preview=summarize_tool_result(message.text),
            failed=message.status == "error",
        )

    def _flush_pending_chunk(self) -> None:
        """Parse joined chunks into complete tool calls."""
        if self._pending_chunk is not None:
            self._remember_calls(self._pending_chunk.tool_calls)
            self._pending_chunk = None

    def _remember_calls(self, tool_calls: list[Any]) -> None:
        """Store tool calls by id."""
        for call in tool_calls:
            if call.get("id"):
                self._calls_by_id[call["id"]] = dict(call)
