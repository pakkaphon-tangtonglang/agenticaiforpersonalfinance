"""
Abstract base class for LLM providers.

Defines the standard interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import TypedDict


class LLMUsage(TypedDict):
    """Token usage information from LLM response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


class LLMResponse(TypedDict):
    """Standardized LLM response format across all providers."""

    content: str
    model: str
    usage: LLMUsage
    provider: str


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM providers.

    All provider implementations must inherit from this class and implement
    the create_message method to ensure consistent interface across providers.
    """

    @abstractmethod
    def create_message(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Create a message using the LLM provider.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-1.0)

        Returns:
            LLMResponse: Standardized response with content, model, usage, provider

        Raises:
            Exception: If the API call fails

        Example:
            >>> client = SomeLLMClient(api_key="...")
            >>> response = client.create_message(
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     max_tokens=1000
            ... )
            >>> print(response["content"])
        """
