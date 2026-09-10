"""Tests for RAG knowledge base manager."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from finance_ai.rag.document_models import DocumentChunk, DocumentMetadata
from finance_ai.rag.knowledge_base import KnowledgeBaseManager, create_knowledge_base_manager


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Create a mock FinanceVectorStore."""
    mock = MagicMock()
    mock.add_chunks.return_value = 3
    return mock


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    """Create sample document metadata."""
    return DocumentMetadata(source_file="tax_guide.md", domain="tax", title="คู่มือภาษี")


@pytest.fixture
def temp_knowledge_dir(tmp_path: Path) -> Path:
    """Create a temp directory with sample documents."""
    (tmp_path / "tax_guide.md").write_text("# คู่มือภาษี\n\nเนื้อหาภาษี\n", encoding="utf-8")
    (tmp_path / "investment_stocks.md").write_text("# หุ้นไทย\n\nข้อมูลหุ้น\n", encoding="utf-8")
    return tmp_path


class TestIndexDocument:
    """Tests for indexing a single document."""

    @patch("finance_ai.rag.knowledge_base.load_document")
    @patch("finance_ai.rag.knowledge_base.split_text_into_chunks")
    def test_loads_splits_and_adds(
        self,
        mock_split: MagicMock,
        mock_load: MagicMock,
        mock_vector_store: MagicMock,
        sample_metadata: DocumentMetadata,
    ) -> None:
        """Test that index_document loads, splits, and adds to store."""
        mock_load.return_value = ("content text", sample_metadata)
        mock_chunks = [MagicMock(spec=DocumentChunk)]
        mock_split.return_value = mock_chunks
        mock_vector_store.add_chunks.return_value = 1

        manager = KnowledgeBaseManager(vector_store=mock_vector_store)
        count = manager.index_document(Path("tax_guide.md"))

        mock_load.assert_called_once_with(Path("tax_guide.md"))
        mock_split.assert_called_once()
        mock_vector_store.add_chunks.assert_called_once_with(mock_chunks)
        assert count == 1

    @patch("finance_ai.rag.knowledge_base.load_document")
    @patch("finance_ai.rag.knowledge_base.split_text_into_chunks")
    def test_passes_chunk_settings(
        self,
        mock_split: MagicMock,
        mock_load: MagicMock,
        mock_vector_store: MagicMock,
        sample_metadata: DocumentMetadata,
    ) -> None:
        """Test that chunk_size and chunk_overlap are passed to splitter."""
        mock_load.return_value = ("content", sample_metadata)
        mock_split.return_value = []
        mock_vector_store.add_chunks.return_value = 0

        manager = KnowledgeBaseManager(
            vector_store=mock_vector_store,
            chunk_size=300,
            chunk_overlap=30,
        )
        manager.index_document(Path("test.md"))

        call_kwargs = mock_split.call_args
        assert call_kwargs.kwargs["chunk_size"] == 300
        assert call_kwargs.kwargs["chunk_overlap"] == 30

    @patch("finance_ai.rag.knowledge_base.load_document")
    @patch("finance_ai.rag.knowledge_base.split_text_into_chunks")
    def test_empty_document_returns_zero(
        self,
        mock_split: MagicMock,
        mock_load: MagicMock,
        mock_vector_store: MagicMock,
        sample_metadata: DocumentMetadata,
    ) -> None:
        """Test that an empty document produces zero chunks."""
        mock_load.return_value = ("", sample_metadata)
        mock_split.return_value = []
        mock_vector_store.add_chunks.return_value = 0

        manager = KnowledgeBaseManager(vector_store=mock_vector_store)
        count = manager.index_document(Path("empty.md"))
        assert count == 0


class TestIndexDirectory:
    """Tests for indexing an entire directory."""

    @patch("finance_ai.rag.knowledge_base.discover_documents")
    @patch.object(KnowledgeBaseManager, "index_document")
    def test_indexes_all_documents(
        self,
        mock_index: MagicMock,
        mock_discover: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test that all discovered documents are indexed."""
        mock_discover.return_value = [Path("a.md"), Path("b.md")]
        mock_index.return_value = 5

        manager = KnowledgeBaseManager(vector_store=mock_vector_store)
        total = manager.index_directory(Path("docs"))

        assert mock_index.call_count == 2
        assert total == 10

    @patch("finance_ai.rag.knowledge_base.discover_documents")
    def test_empty_directory_returns_zero(
        self,
        mock_discover: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test that an empty directory returns zero chunks."""
        mock_discover.return_value = []
        manager = KnowledgeBaseManager(vector_store=mock_vector_store)
        total = manager.index_directory(Path("empty"))
        assert total == 0


class TestRebuildIndex:
    """Tests for rebuilding the index."""

    @patch("finance_ai.rag.knowledge_base.discover_documents")
    @patch.object(KnowledgeBaseManager, "index_document")
    def test_clears_then_reindexes(
        self,
        mock_index: MagicMock,
        mock_discover: MagicMock,
        mock_vector_store: MagicMock,
    ) -> None:
        """Test that rebuild clears collection then re-indexes."""
        mock_discover.return_value = [Path("a.md")]
        mock_index.return_value = 3

        manager = KnowledgeBaseManager(vector_store=mock_vector_store)
        total = manager.rebuild_index(Path("docs"))

        mock_vector_store.clear_collection.assert_called_once()
        assert total == 3


class TestGetDocumentCount:
    """Tests for delegating document count to the vector store."""

    def test_delegates_to_vector_store(self, mock_vector_store: MagicMock) -> None:
        """get_document_count returns the vector store's document count."""
        mock_vector_store.get_document_count.return_value = 7

        manager = KnowledgeBaseManager(vector_store=mock_vector_store)
        count = manager.get_document_count()

        assert count == 7
        mock_vector_store.get_document_count.assert_called_once()


class TestCreateKnowledgeBaseManager:
    """Tests for the factory function."""

    @patch("finance_ai.rag.knowledge_base.FinanceVectorStore")
    @patch("finance_ai.rag.knowledge_base.create_embeddings")
    @patch("finance_ai.rag.knowledge_base.get_settings")
    def test_creates_manager_from_settings(
        self,
        mock_get_settings: MagicMock,
        mock_create_embeddings: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Test that factory creates a fully configured manager."""
        from finance_ai.core.config import Settings

        settings = Settings(google_api_key="test-key")
        mock_get_settings.return_value = settings

        manager = create_knowledge_base_manager()

        mock_create_embeddings.assert_called_once_with(settings)
        mock_store_cls.assert_called_once()
        assert isinstance(manager, KnowledgeBaseManager)

    @patch("finance_ai.rag.knowledge_base.FinanceVectorStore")
    @patch("finance_ai.rag.knowledge_base.create_embeddings")
    def test_uses_provided_settings(
        self,
        mock_create_embeddings: MagicMock,
        mock_store_cls: MagicMock,
    ) -> None:
        """Test that provided settings are used."""
        from finance_ai.core.config import Settings

        settings = Settings(
            google_api_key="custom-key",
            rag_chunk_size=300,
        )
        manager = create_knowledge_base_manager(settings)
        mock_create_embeddings.assert_called_once_with(settings)
        assert isinstance(manager, KnowledgeBaseManager)
