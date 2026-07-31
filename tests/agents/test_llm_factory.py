"""Tests for LangChain ChatModel factory."""

from unittest.mock import MagicMock, patch

import pytest

from finance_ai.agents.llm_factory import (
    create_chat_model,
    create_google_chat_model,
    create_ocr_chat_model,
    create_ocr_google_chat_model,
    create_ocr_ollama_chat_model,
    create_ollama_chat_model,
    create_openrouter_chat_model,
)
from finance_ai.core.config import Settings


class TestCreateGoogleChatModel:
    """Tests for Google ChatModel creation."""

    @patch("finance_ai.agents.llm_factory.ChatGoogleGenerativeAI")
    def test_creates_with_correct_params(self, mock_cls: MagicMock) -> None:
        """Creates ChatGoogleGenerativeAI with settings values."""
        settings = Settings(
            google_api_key="test-key",
            google_model="gemini-2.5-flash",
            llm_temperature=0.5,
            llm_max_tokens=2000,
        )
        create_google_chat_model(settings)
        mock_cls.assert_called_once_with(
            model="gemini-2.5-flash",
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
            _env_file=None,
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

    @patch("finance_ai.agents.llm_factory.import_chat_ollama")
    def test_cloud_api_key_adds_bearer_header(self, mock_import: MagicMock) -> None:
        """Passes bearer Authorization header via client_kwargs for Ollama Cloud."""
        mock_chat_ollama_cls = MagicMock()
        mock_import.return_value = mock_chat_ollama_cls
        settings = Settings(
            _env_file=None,
            ollama_model="deepseek-v4-flash",
            ollama_base_url="https://ollama.com",
            ollama_api_key="ollama-key-123",
            llm_temperature=0.7,
        )
        create_ollama_chat_model(settings)
        mock_chat_ollama_cls.assert_called_once_with(
            model="deepseek-v4-flash",
            base_url="https://ollama.com",
            temperature=0.7,
            client_kwargs={"headers": {"Authorization": "Bearer ollama-key-123"}},
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


class TestCreateOpenRouterChatModel:
    """Tests for OpenRouter ChatModel creation."""

    @patch("finance_ai.agents.llm_factory.import_chat_openai")
    def test_creates_with_correct_params(self, mock_import: MagicMock) -> None:
        """Creates ChatOpenAI with OpenRouter settings."""
        mock_chat_openai_cls = MagicMock()
        mock_import.return_value = mock_chat_openai_cls
        settings = Settings(
            openrouter_api_key="sk-or-test",
            openrouter_model="deepseek/deepseek-chat-v3-0324",
            llm_temperature=0.7,
            llm_max_tokens=4000,
        )
        create_openrouter_chat_model(settings)
        mock_chat_openai_cls.assert_called_once_with(
            model="deepseek/deepseek-chat-v3-0324",
            api_key="sk-or-test",
            base_url="https://openrouter.ai/api/v1",
            temperature=0.7,
            max_tokens=4000,
        )

    def test_missing_api_key_raises(self) -> None:
        """Raises ValueError when openrouter_api_key is not set."""
        settings = Settings(llm_provider="openrouter", openrouter_api_key=None)
        with pytest.raises(ValueError, match="openrouter_api_key is required"):
            create_openrouter_chat_model(settings)

    @patch(
        "finance_ai.agents.llm_factory.import_chat_openai",
        side_effect=ImportError("langchain-openai not installed"),
    )
    def test_import_error_raises(self, mock_import: MagicMock) -> None:
        """Raises ImportError when langchain-openai is not installed."""
        settings = Settings(llm_provider="openrouter", openrouter_api_key="sk-or-test")
        with pytest.raises(ImportError, match="langchain-openai"):
            create_openrouter_chat_model(settings)


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

    @patch("finance_ai.agents.llm_factory.create_openrouter_chat_model")
    def test_openrouter_provider(self, mock_openrouter: MagicMock) -> None:
        """Dispatches to OpenRouter when provider is openrouter."""
        settings = Settings(llm_provider="openrouter", openrouter_api_key="sk-or-test")
        create_chat_model(settings)
        mock_openrouter.assert_called_once_with(settings)

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


class TestCreateOcrGoogleChatModel:
    """Tests for the OCR Google ChatModel creation."""

    @patch("finance_ai.agents.llm_factory.ChatGoogleGenerativeAI")
    def test_creates_with_ocr_settings(self, mock_cls: MagicMock) -> None:
        """Creates ChatGoogleGenerativeAI from OCR_* settings."""
        settings = Settings(
            _env_file=None,
            ocr_provider="google",
            ocr_model="gemini-2.5-flash",
            ocr_api_key="ocr-google-key",
            ocr_temperature=0.0,
            ocr_max_tokens=2000,
            ocr_timeout=120.0,
        )
        create_ocr_google_chat_model(settings)
        mock_cls.assert_called_once_with(
            model="gemini-2.5-flash",
            google_api_key="ocr-google-key",
            temperature=0.0,
            max_output_tokens=2000,
            request_timeout=120.0,
        )

    def test_missing_api_key_raises(self) -> None:
        """Raises ValueError when ocr_api_key is not set."""
        settings = Settings(_env_file=None, ocr_provider="google", ocr_api_key=None)
        with pytest.raises(ValueError, match="ocr_api_key is required"):
            create_ocr_google_chat_model(settings)


class TestCreateOcrOllamaChatModel:
    """Tests for the OCR OLLAMA ChatModel creation."""

    @patch("finance_ai.agents.llm_factory.import_chat_ollama")
    def test_creates_with_ocr_settings(self, mock_import: MagicMock) -> None:
        """Creates ChatOllama from OCR_* settings."""
        mock_chat_ollama_cls = MagicMock()
        mock_import.return_value = mock_chat_ollama_cls
        settings = Settings(
            _env_file=None,
            ocr_provider="ollama",
            ocr_model="gemma4:31b",
            ocr_base_url="https://ollama.com",
            ocr_temperature=0.0,
            ocr_max_tokens=2000,
            ocr_timeout=120.0,
        )
        create_ocr_ollama_chat_model(settings)
        mock_chat_ollama_cls.assert_called_once_with(
            model="gemma4:31b",
            base_url="https://ollama.com",
            temperature=0.0,
            num_predict=2000,
            timeout=120.0,
        )

    @patch("finance_ai.agents.llm_factory.import_chat_ollama")
    def test_cloud_api_key_adds_bearer_header(self, mock_import: MagicMock) -> None:
        """Passes bearer header from OCR_API_KEY for Ollama Cloud OCR."""
        mock_chat_ollama_cls = MagicMock()
        mock_import.return_value = mock_chat_ollama_cls
        settings = Settings(
            _env_file=None,
            ocr_provider="ollama",
            ocr_model="gemma4:31b",
            ocr_base_url="https://ollama.com",
            ocr_api_key="ocr-ollama-key",
            ocr_temperature=0.0,
            ocr_max_tokens=2000,
            ocr_timeout=120.0,
        )
        create_ocr_ollama_chat_model(settings)
        mock_chat_ollama_cls.assert_called_once_with(
            model="gemma4:31b",
            base_url="https://ollama.com",
            temperature=0.0,
            num_predict=2000,
            timeout=120.0,
            client_kwargs={"headers": {"Authorization": "Bearer ocr-ollama-key"}},
        )


class TestCreateOcrChatModel:
    """Tests for the OCR factory dispatch."""

    @patch("finance_ai.agents.llm_factory.create_ocr_google_chat_model")
    def test_google_provider(self, mock_google: MagicMock) -> None:
        """Dispatches to Google OCR when ocr_provider is google."""
        settings = Settings(_env_file=None, ocr_provider="google", ocr_api_key="ocr-key")
        create_ocr_chat_model(settings)
        mock_google.assert_called_once_with(settings)

    @patch("finance_ai.agents.llm_factory.create_ocr_ollama_chat_model")
    def test_ollama_provider(self, mock_ollama: MagicMock) -> None:
        """Dispatches to Ollama OCR when ocr_provider is ollama."""
        settings = Settings(_env_file=None, ocr_provider="ollama")
        create_ocr_chat_model(settings)
        mock_ollama.assert_called_once_with(settings)

    @patch("finance_ai.agents.llm_factory.get_settings")
    @patch("finance_ai.agents.llm_factory.create_ocr_ollama_chat_model")
    def test_uses_default_settings(
        self,
        mock_ollama: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """Uses get_settings() when no settings provided."""
        mock_settings = Settings(_env_file=None, ocr_provider="ollama")
        mock_get_settings.return_value = mock_settings
        create_ocr_chat_model()
        mock_get_settings.assert_called_once()
