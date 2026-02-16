"""
Shared pytest fixtures for all tests.

Provides mock configurations, clients, and responses for testing.
"""

from decimal import Decimal
import pytest
from finance_ai.core.config import Settings
from finance_ai.core.llm.base import LLMResponse


@pytest.fixture
def mock_settings() -> Settings:
    """
    Create mock settings for testing.

    Returns:
        Settings: Test configuration with safe defaults
    """
    return Settings(
        app_env="development",
        log_level="DEBUG",
        llm_provider="google",
        google_api_key="test-key",
        google_model="gemini-pro",
        ollama_base_url="http://localhost:11434",
        ollama_model="THALLE",
    )


@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """
    Create mock LLM response.

    Returns:
        LLMResponse: Standard response format for testing
    """
    return LLMResponse(
        content="Test response content",
        model="test-model",
        usage={
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
        provider="test",
    )


@pytest.fixture
def mock_messages() -> list[dict[str, str]]:
    """
    Create mock message list.

    Returns:
        list: Standard message format for testing
    """
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def mock_money() -> Decimal:
    """
    Create mock money amount.

    Returns:
        Decimal: Test money amount
    """
    return Decimal("1000.50")
