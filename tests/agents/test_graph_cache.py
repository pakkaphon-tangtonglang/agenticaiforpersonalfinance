"""Tests for graph caching module."""

from unittest.mock import MagicMock

from langchain_core.language_models.chat_models import BaseChatModel

from finance_ai.agents.graph_cache import get_compiled_graph


class TestGetCompiledGraph:
    """Tests for graph caching."""

    def test_returns_compiled_graph(self) -> None:
        """Return a compiled graph for valid intent."""
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.bind_tools.return_value = mock_model
        graph = get_compiled_graph("tax", mock_model)
        assert graph is not None

    def test_caches_same_graph(self) -> None:
        """Return same graph instance on second call with same intent."""
        mock_model = MagicMock(spec=BaseChatModel)
        mock_model.bind_tools.return_value = mock_model
        graph1 = get_compiled_graph("expense", mock_model)
        graph2 = get_compiled_graph("expense", mock_model)
        assert graph1 is graph2

    def test_returns_none_for_unknown_intent(self) -> None:
        """Return None for unrecognized intent."""
        mock_model = MagicMock(spec=BaseChatModel)
        result = get_compiled_graph("nonexistent", mock_model)
        assert result is None
