"""
Google Gemini API client implementation.

Provides a client for Google's Gemini models with retry logic and logging.
"""

import logging
import time
from typing import Any

try:
    from google import genai
except ImportError:
    genai = None  # type: ignore[assignment]  # pylint: disable=invalid-name

from finance_ai.core.llm.base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)


class GoogleClient(BaseLLMClient):
    """Google Gemini API client with retry and logging."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        """
        Initialize Google Gemini client.

        Args:
            api_key: Google API key
            model: Model name to use (default: gemini-2.5-flash)

        Raises:
            ValueError: If api_key is empty
        """
        if not api_key:
            raise ValueError("Google API key is required")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model

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
            >>> client = GoogleClient(api_key="...")
            >>> response = client.create_message(
            ...     messages=[{"role": "user", "content": "Hello"}]
            ... )
            >>> print(response["content"])
        """
        max_retries = 3
        prompt = self._convert_messages(messages)

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    ),
                )
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

    def _convert_messages(self, messages: list[dict[str, str]]) -> str:
        """
        Convert message format to Gemini prompt.

        Args:
            messages: List of message dictionaries

        Returns:
            str: Combined prompt text
        """
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"Instructions: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts)

    def _normalize_response(self, response: Any) -> LLMResponse:
        """
        Normalize Google response to standard format.

        Args:
            response: Google GenerateContentResponse object

        Returns:
            LLMResponse: Standardized response format
        """
        content = response.text if hasattr(response, "text") else ""

        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata"):
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count

        logger.info(
            "Google request completed. Tokens: input=%d, output=%d",
            input_tokens,
            output_tokens,
        )

        return LLMResponse(
            content=content,
            model=self.model_name,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            provider="google",
        )

    def __del__(self) -> None:
        """Clean up client resources."""
        if hasattr(self, "client"):
            try:
                self.client.close()
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # Ignore errors during cleanup
