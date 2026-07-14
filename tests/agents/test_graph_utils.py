"""Tests for shared graph utilities."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from finance_ai.agents.graph_utils import should_continue


class TestShouldContinue:
    """Tests for the shared should_continue function."""

    def test_returns_tools_when_tool_calls_exist(self) -> None:
        """Return 'tools' when last message has tool_calls."""
        msg = AIMessage(content="", tool_calls=[{"id": "c1", "name": "tool", "args": {}}])
        state: dict[str, Any] = {"messages": [msg]}
        assert should_continue(state) == "tools"

    def test_returns_end_when_no_tool_calls(self) -> None:
        """Return 'end' when last message has no tool_calls."""
        msg = AIMessage(content="Hello")
        state: dict[str, Any] = {"messages": [msg]}
        assert should_continue(state) == "end"

    def test_returns_end_when_tool_calls_empty(self) -> None:
        """Return 'end' when tool_calls list is empty."""
        msg = AIMessage(content="", tool_calls=[])
        state: dict[str, Any] = {"messages": [msg]}
        assert should_continue(state) == "end"

    def test_returns_end_when_no_tool_calls_attr(self) -> None:
        """Return 'end' when message lacks tool_calls attribute."""
        msg = MagicMock(spec=[])
        state: dict[str, Any] = {"messages": [msg]}
        assert should_continue(state) == "end"
