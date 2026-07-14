"""Tests for evaluation CLI vector store helper."""

from unittest.mock import MagicMock, patch

from finance_ai.evaluation.cli import _create_vector_store


class TestCreateVectorStore:
    """Tests for _create_vector_store helper."""

    @patch("finance_ai.core.config.get_settings")
    @patch("finance_ai.rag.embedding_factory.create_embeddings")
    @patch("finance_ai.rag.vector_store.FinanceVectorStore")
    @patch("finance_ai.rag.knowledge_base.KnowledgeBaseManager")
    def test_create_vector_store_builds_index(
        self,
        mock_manager_cls: MagicMock,
        mock_store_cls: MagicMock,
        mock_create_embeddings: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """_create_vector_store creates embeddings, store, and rebuilds index."""
        mock_settings = MagicMock()
        mock_settings.rag_knowledge_base_directory = "/tmp/kb"
        mock_settings.rag_chunk_size = 500
        mock_settings.rag_chunk_overlap = 50
        mock_get_settings.return_value = mock_settings
        mock_embeddings = MagicMock()
        mock_create_embeddings.return_value = mock_embeddings
        mock_store = MagicMock()
        mock_store.get_document_count.return_value = 0
        mock_store_cls.return_value = mock_store
        mock_manager = MagicMock()
        mock_manager.rebuild_index.return_value = 42
        mock_manager_cls.return_value = mock_manager

        store = _create_vector_store(reuse_index=False)

        mock_create_embeddings.assert_called_once_with(mock_settings)
        mock_store_cls.assert_called_once_with(embeddings=mock_embeddings)
        mock_manager.rebuild_index.assert_called_once()
        assert store is mock_store

    @patch("finance_ai.core.config.get_settings")
    @patch("finance_ai.rag.embedding_factory.create_embeddings")
    @patch("finance_ai.rag.vector_store.FinanceVectorStore")
    def test_create_vector_store_reuses_index(
        self,
        mock_store_cls: MagicMock,
        mock_create_embeddings: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """_create_vector_store reuses existing index when requested."""
        mock_settings = MagicMock()
        mock_get_settings.return_value = mock_settings
        mock_store = MagicMock()
        mock_store.get_document_count.return_value = 10
        mock_store_cls.return_value = mock_store

        store = _create_vector_store(reuse_index=True)

        assert store is mock_store
