"""
OLLAMA local models API client implementation.

Provides a client for OLLAMA local models with HTTP API integration.
"""

import logging
import time
from typing import Any
import httpx  # pylint: disable=import-error
from finance_ai.core.llm.base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class OLLAMAClient(BaseLLMClient):
    """OLLAMA API client with retry and logging."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2") -> None:
        """
        Initialize OLLAMA client.

        Args:
            base_url: OLLAMA server URL (default: http://localhost:11434)
            model: Model name to use (default: llama2)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_url = f"{self.base_url}/api/chat"

    def create_message(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Create a message with retry logic.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate (default: 4000)
            temperature: Sampling temperature 0.0-1.0 (default: 0.7)

        Returns:
            LLMResponse: Standardized response with content, model, usage, provider

        Raises:
            RuntimeError: If all retries fail

        Example:
            >>> client = OLLAMAClient(base_url="http://localhost:11434")
            >>> response = client.create_message(
            ...     messages=[{"role": "user", "content": "Hello"}]
            ... )
            >>> print(response["content"])
        """
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self._send_request(messages, max_tokens, temperature)
                return self._normalize_response(response)

            except Exception as error:  # pylint: disable=broad-exception-caught
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    logger.warning(
                        "Retry %d/%d after %ds: %s",
                        attempt + 1,
                        max_retries,
                        wait_time,
                        error,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("All %d retries failed: %s", max_retries, error)
                    raise

        raise RuntimeError(f"Failed after {max_retries} retries")

    def _send_request(
        self, messages: list[dict[str, str]], max_tokens: int, temperature: float
    ) -> Any:
        """
        Send HTTP request to OLLAMA API.

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Any: API response data
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        with httpx.Client(timeout=120.0) as client:
            response = client.post(self.api_url, json=payload)
            response.raise_for_status()
            result: Any = response.json()
            return result

    def _normalize_response(self, response: Any) -> LLMResponse:
        """
        Normalize OLLAMA response to standard format.

        Args:
            response: OLLAMA API response dictionary

        Returns:
            LLMResponse: Standardized response format
        """
        message = response.get("message", {})
        content = message.get("content", "")

        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)

        logger.info(
            "OLLAMA request completed. Tokens: input=%d, output=%d",
            prompt_tokens,
            completion_tokens,
        )

        return LLMResponse(
            content=content,
            model=self.model,
            usage={
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            provider="ollama",
        )
