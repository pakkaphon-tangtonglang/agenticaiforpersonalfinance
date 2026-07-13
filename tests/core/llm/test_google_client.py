"""Tests for Google client implementation."""

from unittest.mock import Mock, patch
import pytest
from finance_ai.core.llm.google_client import GoogleClient


@patch("finance_ai.core.llm.google_client.genai.Client")
def test_google_client_initialization(mock_client_class: Mock) -> None:
    """Test GoogleClient initializes correctly."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    client = GoogleClient(api_key="test-key", model="gemini-2.5-flash")

    assert client.model_name == "gemini-2.5-flash"
    mock_client_class.assert_called_once_with(api_key="test-key")


def test_google_client_empty_key_raises_error() -> None:
    """Test GoogleClient raises error with empty key."""
    with pytest.raises(ValueError, match="Google API key is required"):
        GoogleClient(api_key="")


@patch("finance_ai.core.llm.google_client.genai.Client")
def test_create_message_success(mock_client_class: Mock) -> None:
    """Test create_message returns proper response."""
    mock_usage = Mock()
    mock_usage.prompt_token_count = 10
    mock_usage.candidates_token_count = 20

    mock_response = Mock()
    mock_response.text = "Test response"
    mock_response.usage_metadata = mock_usage

    mock_client = Mock()
    mock_client.models.generate_content.return_value = mock_response
    mock_client_class.return_value = mock_client

    client = GoogleClient(api_key="test-key")
    messages = [{"role": "user", "content": "Hello"}]

    response = client.create_message(messages)

    assert response["content"] == "Test response"
    assert response["model"] == "gemini-2.5-flash"
    assert response["usage"]["input_tokens"] == 10
    assert response["usage"]["output_tokens"] == 20
    assert response["usage"]["total_tokens"] == 30
    assert response["provider"] == "google"


@patch("finance_ai.core.llm.google_client.genai.Client")
@patch("finance_ai.core.llm.google_client.time.sleep")
def test_create_message_retry_on_error(mock_sleep: Mock, mock_client_class: Mock) -> None:
    """Test create_message retries on error."""
    mock_client = Mock()
    mock_client.models.generate_content.side_effect = [
        Exception("Rate limit"),
        Exception("Rate limit"),
        Exception("Rate limit"),
    ]
    mock_client_class.return_value = mock_client

    client = GoogleClient(api_key="test-key")
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(Exception, match="Rate limit"):
        client.create_message(messages)

    assert mock_client.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2


@patch("finance_ai.core.llm.google_client.genai.Client")
def test_convert_messages(mock_client_class: Mock) -> None:
    """Test message conversion to Gemini format."""
    mock_client = Mock()
    mock_client_class.return_value = mock_client

    client = GoogleClient(api_key="test-key")

    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    result = client._convert_messages(messages)  # pylint: disable=protected-access

    assert "Instructions: You are helpful." in result
    assert "User: Hello" in result
    assert "Assistant: Hi there!" in result
