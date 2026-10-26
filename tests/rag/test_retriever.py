"""Tests for RAG retriever."""

from unittest.mock import MagicMock

import pytest

from finance_ai.rag.document_models import RetrievalResult
from finance_ai.rag.retriever import FinanceRetriever, format_results_as_context


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock FinanceVectorStore."""
    return MagicMock()


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
        RetrievalResult(
            content="ขั้นบันไดภาษี 0-150,000 ยกเว้น",
            source_file="tax_thai_personal_income.md",
            domain="tax",
            relevance_score=0.85,
        ),
    ]


class TestRetrieve:
    """Tests for the retrieve method."""

    def test_returns_results_from_vector_store(
        self,
        mock_vector_store: MagicMock,
        sample_results: list[RetrievalResult],
    ) -> None:
        """Test that retrieve delegates to vector store search."""
        mock_vector_store.search.return_value = sample_results
        retriever = FinanceRetriever(vector_store=mock_vector_store, top_k=3)
        results = retriever.retrieve("ค่าลดหย่อน")
        assert len(results) == 2
        assert results[0].content == "ค่าลดหย่อนส่วนตัว 60,000 บาท"

    def test_passes_domain_filter(self, mock_vector_store: MagicMock) -> None:
        """Test that domain filter is passed to vector store."""
        mock_vector_store.search.return_value = []
        retriever = FinanceRetriever(vector_store=mock_vector_store)
        retriever.retrieve("query", domain="tax")
        mock_vector_store.search.assert_called_once_with(
            query="query", top_k=3, domain_filter="tax"
        )

    def test_none_domain_passes_none(self, mock_vector_store: MagicMock) -> None:
        """Test that None domain passes None to vector store."""
        mock_vector_store.search.return_value = []
        retriever = FinanceRetriever(vector_store=mock_vector_store)
        retriever.retrieve("query")
        mock_vector_store.search.assert_called_once_with(query="query", top_k=3, domain_filter=None)

    def test_uses_configured_top_k(self, mock_vector_store: MagicMock) -> None:
        """Test that top_k from constructor is used."""
        mock_vector_store.search.return_value = []
        retriever = FinanceRetriever(vector_store=mock_vector_store, top_k=5)
        retriever.retrieve("query")
        mock_vector_store.search.assert_called_once_with(query="query", top_k=5, domain_filter=None)


class TestRetrieveAsContext:
    """Tests for the retrieve_as_context method."""

    def test_formats_results_with_source(
        self,
        mock_vector_store: MagicMock,
        sample_results: list[RetrievalResult],
    ) -> None:
        """Test that results are formatted with source attribution."""
        mock_vector_store.search.return_value = sample_results
        retriever = FinanceRetriever(vector_store=mock_vector_store)
        context = retriever.retrieve_as_context("ค่าลดหย่อน")
        assert "[แหล่งที่มา: tax_deductions_guide.md]" in context
        assert "ค่าลดหย่อนส่วนตัว 60,000 บาท" in context
        assert "[แหล่งที่มา: tax_thai_personal_income.md]" in context

    def test_empty_results_return_empty_string(self, mock_vector_store: MagicMock) -> None:
        """Test that empty results return an empty string."""
        mock_vector_store.search.return_value = []
        retriever = FinanceRetriever(vector_store=mock_vector_store)
        context = retriever.retrieve_as_context("nonexistent")
        assert context == ""

    def test_passes_domain_filter(self, mock_vector_store: MagicMock) -> None:
        """Test that domain is passed through to retrieve."""
        mock_vector_store.search.return_value = []
        retriever = FinanceRetriever(vector_store=mock_vector_store)
        retriever.retrieve_as_context("query", domain="investment")
        mock_vector_store.search.assert_called_once_with(
            query="query", top_k=3, domain_filter="investment"
        )


class TestRetrieveByMetadata:
    """Tests for metadata-filtered retrieval used by the guru tool."""

    def test_passes_metadata_filter(
        self, mock_vector_store: MagicMock, sample_results: list[RetrievalResult]
    ) -> None:
        """The metadata filter is forwarded to the vector store."""
        mock_vector_store.search.return_value = sample_results
        retriever = FinanceRetriever(vector_store=mock_vector_store)

        results = retriever.retrieve_by_metadata("query", {"guru_id": "buffett"})

        assert results == sample_results
        mock_vector_store.search.assert_called_once_with(
            query="query", top_k=3, metadata_filter={"guru_id": "buffett"}
        )


class TestFormatResultsAsContext:
    """Tests for the public context formatter."""

    def test_includes_source_url_when_present(self) -> None:
        """Guru results show their public URL next to the file name."""
        result = RetrievalResult(
            content="Be fearful",
            source_file="letter_2008.md",
            domain="guru",
            relevance_score=0.4,
            source_url="https://example.com/2008.pdf",
        )

        context = format_results_as_context([result])

        assert context == "[แหล่งที่มา: letter_2008.md | https://example.com/2008.pdf]\nBe fearful"
