"""Tests for RAG agent tool."""

from unittest.mock import MagicMock, patch

import pytest

from finance_ai.rag.document_models import RetrievalResult


@pytest.fixture
def sample_results() -> list[RetrievalResult]:
    """Create sample retrieval results."""
    return [
        RetrievalResult(
            content="ค่าลดหย่อนส่วนตัว 60,000 บาท",
            source_file="tax_deductions_guide.md",
            domain="tax",
            relevance_score=0.92,
        ),
    ]


@pytest.fixture
def mock_retriever(sample_results: list[RetrievalResult]) -> MagicMock:
    """Create a mock FinanceRetriever."""
    mock = MagicMock()
    mock.retrieve.return_value = sample_results
    mock.retrieve_as_context.return_value = (
        "[แหล่งที่มา: tax_deductions_guide.md]\nค่าลดหย่อนส่วนตัว 60,000 บาท"
    )
    return mock


class TestSearchFinanceKnowledge:
    """Tests for the search_finance_knowledge tool."""

    @patch("finance_ai.agents.rag_tool.get_retriever")
    def test_returns_results_and_context(
        self, mock_get_retriever: MagicMock, mock_retriever: MagicMock
    ) -> None:
        """Test that the tool returns results and context."""
        from finance_ai.agents.rag_tool import search_finance_knowledge

        mock_get_retriever.return_value = mock_retriever
        result = search_finance_knowledge.invoke({"query": "ค่าลดหย่อน", "domain": "tax"})
        assert len(result["citations"]) == 1
        assert result["citations"][0]["index"] == 1
        assert "[1]" in result["context"]
        assert "ค่าลดหย่อนส่วนตัว" in result["context"]
        assert "tax_deductions_guide" not in str(result)

    @patch("finance_ai.agents.rag_tool.get_retriever")
    def test_passes_domain_filter(
        self, mock_get_retriever: MagicMock, mock_retriever: MagicMock
    ) -> None:
        """Test that domain filter is passed to retriever."""
        from finance_ai.agents.rag_tool import search_finance_knowledge

        mock_get_retriever.return_value = mock_retriever
        search_finance_knowledge.invoke({"query": "ภาษี", "domain": "tax"})
        mock_retriever.retrieve.assert_called_once_with("ภาษี", domain="tax")

    @patch("finance_ai.agents.rag_tool.get_retriever")
    def test_empty_domain_passes_none(
        self, mock_get_retriever: MagicMock, mock_retriever: MagicMock
    ) -> None:
        """Test that empty domain string is treated as None."""
        from finance_ai.agents.rag_tool import search_finance_knowledge

        mock_get_retriever.return_value = mock_retriever
        search_finance_knowledge.invoke({"query": "ข้อมูล", "domain": ""})
        mock_retriever.retrieve.assert_called_once_with("ข้อมูล", domain=None)

    @patch("finance_ai.agents.rag_tool.get_retriever")
    def test_empty_results(self, mock_get_retriever: MagicMock) -> None:
        """Test handling of empty retrieval results."""
        from finance_ai.agents.rag_tool import search_finance_knowledge

        mock_ret = MagicMock()
        mock_ret.retrieve.return_value = []
        mock_ret.retrieve_as_context.return_value = ""
        mock_get_retriever.return_value = mock_ret

        result = search_finance_knowledge.invoke({"query": "nonexistent", "domain": ""})
        assert result["citations"] == []
        assert result["context"] == ""

    @patch("finance_ai.agents.rag_tool.get_retriever")
    def test_default_domain_is_empty_string(
        self, mock_get_retriever: MagicMock, mock_retriever: MagicMock
    ) -> None:
        """Test that calling without domain defaults to empty string (None)."""
        from finance_ai.agents.rag_tool import search_finance_knowledge

        mock_get_retriever.return_value = mock_retriever
        search_finance_knowledge.invoke({"query": "test"})
        mock_retriever.retrieve.assert_called_once_with("test", domain=None)

    @patch("finance_ai.agents.rag_tool.get_retriever")
    def test_numbers_continue_after_earlier_search_in_turn(
        self, mock_get_retriever: MagicMock, mock_retriever: MagicMock
    ) -> None:
        """A second search in the same turn gets the next citation number."""
        from langchain_core.messages import HumanMessage

        from finance_ai.agents.citation_sources import build_citations
        from finance_ai.agents.rag_tool import search_finance_knowledge
        from tests.agents.test_citation_sources import _result, _tool_message

        mock_get_retriever.return_value = mock_retriever
        earlier = build_citations([_result("other.md")], [])
        messages = [HumanMessage(content="q"), _tool_message(earlier, "search_finance_knowledge")]

        result = search_finance_knowledge.func(  # type: ignore[attr-defined]
            query="ภาษี", messages=messages
        )

        assert result["citations"][0]["index"] == 2
