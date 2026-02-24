"""Tests for LangChain ChatModel factory."""

from unittest.mock import MagicMock, patch

import pytest

from finance_ai.agents.llm_factory import (
    create_chat_model,
    create_google_chat_model,
    create_ollama_chat_model,
)
from finance_ai.core.config import Settings


class TestCreateGoogleChatModel:
    """Tests for Google ChatModel creation."""

    @patch("finance_ai.agents.llm_factory.ChatGoogleGenerativeAI")
    def test_creates_with_correct_params(self, mock_cls: MagicMock) -> None:
        """Creates ChatGoogleGenerativeAI with settings values."""
        settings = Settings(
            google_api_key="test-key",
            google_model="gemini-pro",
            llm_temperature=0.5,
            llm_max_tokens=2000,
        )
        create_google_chat_model(settings)
        mock_cls.assert_called_once_with(
            model="gemini-pro",
            google_api_key="test-key",
            temperature=0.5,
            max_output_tokens=2000,
        )

    def test_missing_api_key_raises(self) -> None:
        """Raises ValueError when google_api_key is not set."""
        settings = Settings(google_api_key=None)
        with pytest.raises(ValueError, match="google_api_key is required"):
            create_google_chat_model(settings)


class TestCreateOllamaChatModel:
    """Tests for OLLAMA ChatModel creation."""

    @patch("finance_ai.agents.llm_factory.import_chat_ollama")
    def test_creates_with_correct_params(self, mock_import: MagicMock) -> None:
        """Creates ChatOllama with settings values."""
        mock_chat_ollama_cls = MagicMock()
        mock_import.return_value = mock_chat_ollama_cls
        settings = Settings(
            ollama_model="THALLE",
            ollama_base_url="http://localhost:11434",
            llm_temperature=0.7,
        )
        create_ollama_chat_model(settings)
        mock_chat_ollama_cls.assert_called_once_with(
            model="THALLE",
            base_url="http://localhost:11434",
            temperature=0.7,
        )

    @patch(
        "finance_ai.agents.llm_factory.import_chat_ollama",
        side_effect=ImportError("langchain-ollama not installed"),
    )
    def test_import_error_raises(self, mock_import: MagicMock) -> None:
        """Raises ImportError when langchain-ollama is not installed."""
        settings = Settings(llm_provider="ollama")
        with pytest.raises(ImportError, match="langchain-ollama"):
            create_ollama_chat_model(settings)


class TestCreateChatModel:
    """Tests for the main factory function."""

    @patch("finance_ai.agents.llm_factory.create_google_chat_model")
    def test_google_provider(self, mock_google: MagicMock) -> None:
        """Dispatches to Google when provider is google."""
        settings = Settings(llm_provider="google", google_api_key="test-key")
        create_chat_model(settings)
        mock_google.assert_called_once_with(settings)

    @patch("finance_ai.agents.llm_factory.create_ollama_chat_model")
    def test_ollama_provider(self, mock_ollama: MagicMock) -> None:
        """Dispatches to OLLAMA when provider is ollama."""
        settings = Settings(llm_provider="ollama")
        create_chat_model(settings)
        mock_ollama.assert_called_once_with(settings)

    @patch("finance_ai.agents.llm_factory.get_settings")
    @patch("finance_ai.agents.llm_factory.create_google_chat_model")
    def test_uses_default_settings(
        self,
        mock_google: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """Uses get_settings() when no settings provided."""
        mock_settings = Settings(google_api_key="test-key")
        mock_get_settings.return_value = mock_settings
        create_chat_model()
        mock_get_settings.assert_called_once()
