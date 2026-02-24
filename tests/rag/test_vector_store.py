"""Tests for RAG vector store."""

from unittest.mock import MagicMock, patch

import pytest

from finance_ai.rag.document_models import DocumentChunk, DocumentMetadata, RetrievalResult
from finance_ai.rag.vector_store import FinanceVectorStore


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Create a mock embeddings instance."""
    return MagicMock()


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Create sample chunks for testing."""
    metadata = DocumentMetadata(source_file="tax_guide.md", domain="tax", title="คู่มือภาษี")
    return [
        DocumentChunk(
            chunk_id="chunk_0",
            content="ภาษีเงินได้บุคคลธรรมดา",
            metadata=metadata,
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="chunk_1",
            content="ค่าลดหย่อนส่วนตัว 60,000 บาท",
            metadata=metadata,
            chunk_index=1,
        ),
    ]


class TestFinanceVectorStoreInit:
    """Tests for FinanceVectorStore initialization."""

    @patch("finance_ai.rag.vector_store.Chroma")
    def test_creates_chroma_instance(
        self, mock_chroma: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        """Test that Chroma is initialized with correct parameters."""
        FinanceVectorStore(
            embeddings=mock_embeddings,
            persist_directory="./test_db",
            collection_name="test_collection",
        )
        mock_chroma.assert_called_once_with(
            collection_name="test_collection",
            embedding_function=mock_embeddings,
            persist_directory="./test_db",
        )


class TestAddChunks:
    """Tests for adding chunks to the vector store."""

    @patch("finance_ai.rag.vector_store.Chroma")
    def test_adds_chunks_to_store(
        self,
        mock_chroma: MagicMock,
        mock_embeddings: MagicMock,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        """Test that chunks are added with correct texts, metadatas, and ids."""
        store = FinanceVectorStore(embeddings=mock_embeddings)
        store.add_chunks(sample_chunks)
        mock_instance = mock_chroma.return_value
        mock_instance.add_texts.assert_called_once()
        call_kwargs = mock_instance.add_texts.call_args
        assert len(call_kwargs.kwargs["texts"]) == 2
        assert len(call_kwargs.kwargs["metadatas"]) == 2
        assert len(call_kwargs.kwargs["ids"]) == 2

    @patch("finance_ai.rag.vector_store.Chroma")
    def test_returns_chunk_count(
        self,
        mock_chroma: MagicMock,
        mock_embeddings: MagicMock,
        sample_chunks: list[DocumentChunk],
    ) -> None:
        """Test that add_chunks returns the number of chunks added."""
        store = FinanceVectorStore(embeddings=mock_embeddings)
        count = store.add_chunks(sample_chunks)
        assert count == 2

    @patch("finance_ai.rag.vector_store.Chroma")
    def test_empty_chunks_returns_zero(
        self, mock_chroma: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        """Test that adding empty list returns 0."""
        store = FinanceVectorStore(embeddings=mock_embeddings)
        count = store.add_chunks([])
        assert count == 0


class TestSearch:
    """Tests for searching the vector store."""

    @patch("finance_ai.rag.vector_store.Chroma")
    def test_search_returns_retrieval_results(
        self, mock_chroma: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        """Test that search returns properly formatted RetrievalResult list."""
        mock_doc = MagicMock()
        mock_doc.page_content = "ภาษีเงินได้"
        mock_doc.metadata = {
            "source_file": "tax_guide.md",
            "domain": "tax",
        }
        mock_instance = mock_chroma.return_value
        mock_instance.similarity_search_with_score.return_value = [
            (mock_doc, 0.85),
        ]
        store = FinanceVectorStore(embeddings=mock_embeddings)
        results = store.search("ภาษี", top_k=3)
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert results[0].content == "ภาษีเงินได้"
        assert results[0].domain == "tax"

    @patch("finance_ai.rag.vector_store.Chroma")
    def test_search_with_domain_filter(
        self, mock_chroma: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        """Test that domain filter is passed to similarity search."""
        mock_instance = mock_chroma.return_value
        mock_instance.similarity_search_with_score.return_value = []
        store = FinanceVectorStore(embeddings=mock_embeddings)
        store.search("query", domain_filter="tax")
        call_kwargs = mock_instance.similarity_search_with_score.call_args.kwargs
        assert call_kwargs["filter"] == {"domain": "tax"}

    @patch("finance_ai.rag.vector_store.Chroma")
    def test_search_without_domain_filter(
        self, mock_chroma: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        """Test that no filter is applied when domain_filter is None."""
        mock_instance = mock_chroma.return_value
        mock_instance.similarity_search_with_score.return_value = []
        store = FinanceVectorStore(embeddings=mock_embeddings)
        store.search("query")
        call_kwargs = mock_instance.similarity_search_with_score.call_args.kwargs
        assert "filter" not in call_kwargs

    @patch("finance_ai.rag.vector_store.Chroma")
    def test_empty_results(self, mock_chroma: MagicMock, mock_embeddings: MagicMock) -> None:
        """Test that empty search results return empty list."""
        mock_instance = mock_chroma.return_value
        mock_instance.similarity_search_with_score.return_value = []
        store = FinanceVectorStore(embeddings=mock_embeddings)
        results = store.search("nonexistent query")
        assert results == []


class TestClearCollection:
    """Tests for clearing the collection."""

    @patch("finance_ai.rag.vector_store.Chroma")
    def test_clear_deletes_collection(
        self, mock_chroma: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        """Test that clear_collection delegates to the underlying store."""
        store = FinanceVectorStore(embeddings=mock_embeddings)
        store.clear_collection()
        mock_chroma.return_value.delete_collection.assert_called_once()


class TestGetDocumentCount:
    """Tests for getting document count."""

    @patch("finance_ai.rag.vector_store.Chroma")
    def test_returns_correct_count(
        self, mock_chroma: MagicMock, mock_embeddings: MagicMock
    ) -> None:
        """Test that get_document_count returns the collection count."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 42
        mock_chroma.return_value._collection = mock_collection
        store = FinanceVectorStore(embeddings=mock_embeddings)
        assert store.get_document_count() == 42
