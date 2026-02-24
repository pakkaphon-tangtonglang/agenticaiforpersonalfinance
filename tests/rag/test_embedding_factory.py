"""Tests for RAG embedding factory."""

from unittest.mock import MagicMock, patch

import pytest

from finance_ai.core.config import Settings
from finance_ai.rag.embedding_factory import (
    create_embeddings,
    create_google_embeddings,
    create_sentence_transformer_embeddings,
    import_sentence_transformer_embeddings,
)


@pytest.fixture
def google_settings() -> Settings:
    """Create settings for Google embedding provider."""
    return Settings(
        google_api_key="test-api-key",
        rag_embedding_provider="google",
        rag_embedding_model="models/embedding-001",
    )


@pytest.fixture
def sentence_transformer_settings() -> Settings:
    """Create settings for sentence-transformers provider."""
    return Settings(
        rag_embedding_provider="sentence_transformers",
        rag_sentence_transformer_model="paraphrase-multilingual-MiniLM-L12-v2",
    )


class TestCreateGoogleEmbeddings:
    """Tests for Google embeddings creation."""

    @patch("finance_ai.rag.embedding_factory.GoogleGenerativeAIEmbeddings")
    def test_creates_google_embeddings(
        self, mock_cls: MagicMock, google_settings: Settings
    ) -> None:
        """Test that Google embeddings are created with correct parameters."""
        create_google_embeddings(google_settings)
        mock_cls.assert_called_once_with(
            model=google_settings.rag_embedding_model,
            google_api_key=google_settings.google_api_key,
        )

    def test_missing_api_key_raises_error(self) -> None:
        """Test that missing Google API key raises ValueError."""
        settings = Settings(
            google_api_key=None,
            rag_embedding_provider="google",
        )
        with pytest.raises(ValueError, match="google_api_key is required"):
            create_google_embeddings(settings)


class TestImportSentenceTransformerEmbeddings:
    """Tests for lazy import of sentence-transformers."""

    @patch("finance_ai.rag.embedding_factory.importlib.import_module")
    def test_successful_import(self, mock_import: MagicMock) -> None:
        """Test successful import returns the class."""
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        result = import_sentence_transformer_embeddings()
        assert result is mock_module.HuggingFaceEmbeddings

    @patch(
        "finance_ai.rag.embedding_factory.importlib.import_module",
        side_effect=ImportError("no module"),
    )
    def test_import_error_with_helpful_message(self, mock_import: MagicMock) -> None:
        """Test that missing package raises ImportError with install instructions."""
        with pytest.raises(ImportError, match="langchain-community"):
            import_sentence_transformer_embeddings()


class TestCreateSentenceTransformerEmbeddings:
    """Tests for sentence-transformer embeddings creation."""

    @patch("finance_ai.rag.embedding_factory.import_sentence_transformer_embeddings")
    def test_creates_with_correct_model(
        self,
        mock_import: MagicMock,
        sentence_transformer_settings: Settings,
    ) -> None:
        """Test that sentence-transformer embeddings use the configured model."""
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls
        create_sentence_transformer_embeddings(sentence_transformer_settings)
        mock_cls.assert_called_once_with(
            model_name=sentence_transformer_settings.rag_sentence_transformer_model,
        )


class TestCreateEmbeddings:
    """Tests for the factory dispatch function."""

    @patch("finance_ai.rag.embedding_factory.create_google_embeddings")
    @patch("finance_ai.rag.embedding_factory.get_settings")
    def test_defaults_to_google(self, mock_get_settings: MagicMock, mock_create: MagicMock) -> None:
        """Test that default settings dispatch to Google provider."""
        settings = Settings(
            google_api_key="test-key",
            rag_embedding_provider="google",
        )
        mock_get_settings.return_value = settings
        create_embeddings()
        mock_create.assert_called_once_with(settings)

    @patch("finance_ai.rag.embedding_factory.create_google_embeddings")
    def test_google_dispatch(self, mock_create: MagicMock, google_settings: Settings) -> None:
        """Test explicit Google provider dispatch."""
        create_embeddings(google_settings)
        mock_create.assert_called_once_with(google_settings)

    @patch("finance_ai.rag.embedding_factory.create_sentence_transformer_embeddings")
    def test_sentence_transformers_dispatch(
        self,
        mock_create: MagicMock,
        sentence_transformer_settings: Settings,
    ) -> None:
        """Test sentence-transformers provider dispatch."""
        create_embeddings(sentence_transformer_settings)
        mock_create.assert_called_once_with(sentence_transformer_settings)

    def test_uses_provided_settings(self, google_settings: Settings) -> None:
        """Test that provided settings are used instead of defaults."""
        with patch("finance_ai.rag.embedding_factory.create_google_embeddings") as mock_create:
            create_embeddings(google_settings)
            mock_create.assert_called_once_with(google_settings)
