"""Tests for OLLAMA client implementation."""

from unittest.mock import Mock, patch
import pytest
from finance_ai.core.llm.ollama_client import OLLAMAClient


def test_ollama_client_initialization() -> None:
    """Test OLLAMAClient initializes correctly."""
    client = OLLAMAClient(base_url="http://localhost:11434", model="llama2")

    assert client.base_url == "http://localhost:11434"
    assert client.model == "llama2"
    assert client.api_url == "http://localhost:11434/api/chat"


def test_ollama_client_strips_trailing_slash() -> None:
    """Test OLLAMAClient strips trailing slash from base URL."""
    client = OLLAMAClient(base_url="http://localhost:11434/")

    assert client.base_url == "http://localhost:11434"


@patch("finance_ai.core.llm.ollama_client.httpx.Client")
def test_create_message_success(mock_httpx_client: Mock) -> None:
    """Test create_message returns proper response."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "message": {"content": "Test response"},
        "model": "llama2",
        "prompt_eval_count": 10,
        "eval_count": 20,
    }

    mock_client_instance = Mock()
    mock_client_instance.post.return_value = mock_response
    mock_httpx_client.return_value.__enter__.return_value = mock_client_instance

    client = OLLAMAClient()
    messages = [{"role": "user", "content": "Hello"}]

    response = client.create_message(messages)

    assert response["content"] == "Test response"
    assert response["model"] == "llama2"
    assert response["usage"]["input_tokens"] == 10
    assert response["usage"]["output_tokens"] == 20
    assert response["usage"]["total_tokens"] == 30
    assert response["provider"] == "ollama"


@patch("finance_ai.core.llm.ollama_client.httpx.Client")
@patch("finance_ai.core.llm.ollama_client.time.sleep")
def test_create_message_retry_on_error(mock_sleep: Mock, mock_httpx_client: Mock) -> None:
    """Test create_message retries on error."""
    mock_client_instance = Mock()
    mock_client_instance.post.side_effect = [
        Exception("Connection error"),
        Exception("Connection error"),
        Exception("Connection error"),
    ]
    mock_httpx_client.return_value.__enter__.return_value = mock_client_instance

    client = OLLAMAClient()
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(Exception, match="Connection error"):
        client.create_message(messages)

    assert mock_client_instance.post.call_count == 3
    assert mock_sleep.call_count == 2


@patch("finance_ai.core.llm.ollama_client.httpx.Client")
def test_send_request_parameters(mock_httpx_client: Mock) -> None:
    """Test _send_request sends correct parameters."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "message": {"content": "Test"},
        "prompt_eval_count": 5,
        "eval_count": 10,
    }

    mock_client_instance = Mock()
    mock_client_instance.post.return_value = mock_response
    mock_httpx_client.return_value.__enter__.return_value = mock_client_instance

    client = OLLAMAClient(model="llama2")
    messages = [{"role": "user", "content": "Hello"}]

    client.create_message(messages, max_tokens=2000, temperature=0.5)

    call_args = mock_client_instance.post.call_args
    payload = call_args[1]["json"]

    assert payload["model"] == "llama2"
    assert payload["messages"] == messages
    assert payload["stream"] is False
    assert payload["options"]["num_predict"] == 2000
    assert payload["options"]["temperature"] == 0.5
