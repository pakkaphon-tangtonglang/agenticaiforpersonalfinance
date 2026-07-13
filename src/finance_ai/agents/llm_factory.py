"""Factory for creating LangChain ChatModel instances.

Supports Google Gemini and OLLAMA providers based on application settings.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from finance_ai.core.config import Settings, get_settings
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)


def import_chat_ollama() -> type:
    """Import ChatOllama class, raising ImportError if not installed.

    Returns:
        The ChatOllama class.

    Raises:
        ImportError: If langchain-ollama is not installed.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is required for OLLAMA provider. "
            "Install with: pip install langchain-ollama"
        ) from exc
    return ChatOllama  # type: ignore[no-any-return]


def create_google_chat_model(settings: Settings) -> BaseChatModel:
    """Create a ChatGoogleGenerativeAI instance from settings.

    Args:
        settings: Application settings with Google config.

    Returns:
        Configured ChatGoogleGenerativeAI instance.

    Raises:
        ValueError: If google_api_key is not set.

    Example:
        >>> model = create_google_chat_model(settings)
    """
    if not settings.google_api_key:
        raise ValueError("google_api_key is required when llm_provider is 'google'.")
    logger.info("Creating Google ChatModel with model=%s", settings.google_model)
    return ChatGoogleGenerativeAI(  # type: ignore[no-any-return]
        model=settings.google_model,
        google_api_key=settings.google_api_key,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_tokens,
    )


def create_ollama_chat_model(settings: Settings) -> BaseChatModel:
    """Create a ChatOllama instance from settings.

    Args:
        settings: Application settings with OLLAMA config.

    Returns:
        Configured ChatOllama instance.

    Raises:
        ImportError: If langchain-ollama is not installed.

    Example:
        >>> model = create_ollama_chat_model(settings)
    """
    chat_ollama_cls = import_chat_ollama()
    logger.info("Creating OLLAMA ChatModel with model=%s", settings.ollama_model)
    return chat_ollama_cls(  # type: ignore[no-any-return]
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
    )


def import_chat_openai() -> type:
    """Import ChatOpenAI class, raising ImportError if not installed.

    Returns:
        The ChatOpenAI class.

    Raises:
        ImportError: If langchain-openai is not installed.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for OpenRouter provider. "
            "Install with: pip install langchain-openai"
        ) from exc
    return ChatOpenAI  # type: ignore[no-any-return]


def create_openrouter_chat_model(settings: Settings) -> BaseChatModel:
    """Create a ChatOpenAI instance configured for OpenRouter.

    Args:
        settings: Application settings with OpenRouter config.

    Returns:
        Configured ChatOpenAI instance pointing to OpenRouter API.

    Raises:
        ValueError: If openrouter_api_key is not set.
        ImportError: If langchain-openai is not installed.

    Example:
        >>> model = create_openrouter_chat_model(settings)
    """
    if not settings.openrouter_api_key:
        raise ValueError("openrouter_api_key is required when llm_provider is 'openrouter'.")
    chat_openai_cls = import_chat_openai()
    logger.info(
        "Creating OpenRouter ChatModel with model=%s",
        settings.openrouter_model,
    )
    return chat_openai_cls(  # type: ignore[no-any-return]
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def create_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Create a LangChain ChatModel based on the configured provider.

    Args:
        settings: Optional settings override. Uses get_settings() if None.

    Returns:
        A LangChain BaseChatModel instance (Google or OLLAMA).

    Raises:
        ValueError: If the provider is unsupported or config is missing.

    Example:
        >>> model = create_chat_model()
    """
    if settings is None:
        settings = get_settings()
    if settings.llm_provider == "google":
        return create_google_chat_model(settings)
    if settings.llm_provider == "ollama":
        return create_ollama_chat_model(settings)
    if settings.llm_provider == "openrouter":
        return create_openrouter_chat_model(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
