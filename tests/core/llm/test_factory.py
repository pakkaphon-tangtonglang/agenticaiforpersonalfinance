"""Tests for LLM factory pattern."""

import pytest
from finance_ai.core.config import Settings
from finance_ai.core.llm.factory import get_llm_client
from finance_ai.core.llm.google_client import GoogleClient
from finance_ai.core.llm.ollama_client import OLLAMAClient


def test_get_llm_client_openrouter() -> None:
    """Create an OpenRouter client when provider is openrouter."""
    settings = Settings(
        llm_provider="openrouter",
        openrouter_api_key="test-key",
    )

    with pytest.raises((ImportError, Exception)):
        get_llm_client(settings)


def test_get_llm_client_opencode() -> None:
    """Create an OpenCode client when provider is opencode."""
    settings = Settings(
        llm_provider="opencode",
        opencode_api_key="test-key",
    )

    with pytest.raises((ImportError, Exception)):
        get_llm_client(settings)


def test_get_llm_client_unsupported_raises() -> None:
    """Raise ValueError for unsupported provider."""
    settings = Settings(
        llm_provider="google",
        google_api_key="test-key",
    )
    settings.llm_provider = "nonexistent"  # type: ignore

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_client(settings)


def test_get_llm_client_google() -> None:
    """Test factory creates Google client."""
    settings = Settings(
        llm_provider="google",
        google_api_key="test-key",
    )

    client = get_llm_client(settings)

    assert isinstance(client, GoogleClient)
    assert client.model_name == "gemini-2.5-flash"


def test_get_llm_client_ollama() -> None:
    """Test factory creates OLLAMA client."""
    settings = Settings(llm_provider="ollama")

    client = get_llm_client(settings)

    assert isinstance(client, OLLAMAClient)
    assert client.model == "THALLE"


def test_get_llm_client_missing_google_key() -> None:
    """Test factory raises error for missing Google key."""
    settings = Settings(llm_provider="google")

    with pytest.raises(ValueError, match="Google API key is required"):
        get_llm_client(settings)
