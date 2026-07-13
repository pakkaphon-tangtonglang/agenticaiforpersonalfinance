"""Tests for LLM factory pattern."""

import pytest
from finance_ai.core.config import Settings
from finance_ai.core.llm.factory import get_llm_client
from finance_ai.core.llm.google_client import GoogleClient
from finance_ai.core.llm.ollama_client import OLLAMAClient


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


def test_get_llm_client_invalid_provider() -> None:
    """Test factory raises error for invalid provider."""
    settings = Settings(
        llm_provider="google",  # Valid for Pydantic
        google_api_key="test",
    )
    settings.llm_provider = "invalid"  # type: ignore

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm_client(settings)
