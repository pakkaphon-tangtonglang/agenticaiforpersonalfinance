"""OpenRouter API client via OpenAI-compatible endpoint.

Wraps the OpenAI SDK with OpenRouter-specific base URL.
"""

import logging
from typing import Any

from finance_ai.core.llm.base import BaseLLMClient, LLMResponse

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(BaseLLMClient):
    """OpenRouter API client using OpenAI SDK."""

    def __init__(self, api_key: str, model: str = "deepseek/deepseek-chat-v3.1") -> None:
        """Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key.
            model: Model name.

        Raises:
            ValueError: If api_key is empty.
        """
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "langchain-openai is required for OpenRouter. "
                "Install with: uv add langchain-openai"
            ) from exc
        self._client = OpenAI(api_key=api_key, base_url=_OPENROUTER_BASE_URL)
        self._model = model

    def create_message(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Create a message via OpenRouter API.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content, model, usage, provider.

        Raises:
            RuntimeError: If the API call fails.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _normalize_openrouter_response(response, self._model)


def _normalize_openrouter_response(response: Any, model: str) -> LLMResponse:
    """Normalize OpenRouter response to standard format.

    Args:
        response: OpenAI-compatible response object.
        model: Model name.

    Returns:
        LLMResponse with standardized fields.
    """
    content = response.choices[0].message.content or ""
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    logger.info(
        "OpenRouter request: input=%d, output=%d tokens",
        input_tokens,
        output_tokens,
    )
    return LLMResponse(
        content=content,
        model=model,
        usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        provider="openrouter",
    )
