"""
Factory pattern for LLM provider selection.

Creates the appropriate LLM client (Google Gemini or OLLAMA).
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finance_ai.core.config import Settings
    from finance_ai.core.llm.base import BaseLLMClient


def get_llm_client(settings: "Settings") -> "BaseLLMClient":
    """
    Factory function to create appropriate LLM client based on settings.

    Args:
        settings: Application settings with provider configuration

    Returns:
        BaseLLMClient: Configured LLM client instance

    Raises:
        ValueError: If provider is not supported or configuration is invalid

    Example:
        >>> from finance_ai.core.config import get_settings
        >>> settings = get_settings()
        >>> client = get_llm_client(settings)
        >>> response = client.create_message([{"role": "user", "content": "Hi"}])
    """
    provider = settings.llm_provider

    if provider == "google":
        from finance_ai.core.llm.google_client import (
            GoogleClient,
        )  # pylint: disable=import-outside-toplevel

        if not settings.google_api_key:
            raise ValueError("Google API key is required")
        return GoogleClient(api_key=settings.google_api_key, model=settings.google_model)

    if provider == "ollama":
        from finance_ai.core.llm.ollama_client import (
            OLLAMAClient,
        )  # pylint: disable=import-outside-toplevel

        return OLLAMAClient(base_url=settings.ollama_base_url, model=settings.ollama_model)

    if provider == "openrouter":
        from finance_ai.core.llm.openrouter_client import (
            OpenRouterClient,
        )  # pylint: disable=import-outside-toplevel

        if not settings.openrouter_api_key:
            raise ValueError("OpenRouter API key is required")
        return OpenRouterClient(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
        )

    if provider == "opencode":
        from finance_ai.core.llm.opencode_client import (
            OpenCodeClient,
        )  # pylint: disable=import-outside-toplevel

        if not settings.opencode_api_key:
            raise ValueError("OpenCode API key is required")
        return OpenCodeClient(
            api_key=settings.opencode_api_key,
            model=settings.opencode_model,
            base_url=settings.opencode_base_url,
        )

    raise ValueError(
        f"Unsupported LLM provider: {provider}. "
        f"Supported providers: google, ollama, openrouter, opencode"
    )
