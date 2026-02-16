"""Tests for LLM base classes and types."""

from finance_ai.core.llm.base import LLMResponse, LLMUsage


def test_llm_usage_structure() -> None:
    """Test LLMUsage has correct structure."""
    usage: LLMUsage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }

    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    assert usage["total_tokens"] == 150


def test_llm_response_structure() -> None:
    """Test LLMResponse has correct structure."""
    response: LLMResponse = {
        "content": "Test content",
        "model": "test-model",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
        "provider": "test",
    }

    assert response["content"] == "Test content"
    assert response["model"] == "test-model"
    assert response["usage"]["total_tokens"] == 30
    assert response["provider"] == "test"
