"""Factory for creating LangChain ChatModel instances.

Supports Google Gemini and OLLAMA providers based on application settings.
"""

from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
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
    # Call through Any + cast: langchain-ollama typing differs between
    # mypy environments (typed vs untyped import).
    chat_model_class: Any = ChatOllama
    return cast(type, chat_model_class)


def supports_thinking_level(model_name: str) -> bool:
    """Tell whether a Gemini model accepts ``thinking_level``.

    Gemini 1.x/2.x only know ``thinking_budget``; Gemini 3+ use levels.

    Args:
        model_name: Gemini model id, e.g. "gemini-3.5-flash".

    Returns:
        True for Gemini 3 and later.

    Example:
        >>> supports_thinking_level("gemini-2.5-flash")
        False
    """
    return not model_name.startswith(("gemini-1", "gemini-2"))


def _google_thinking_kwargs(settings: Settings) -> dict[str, Any]:
    """Build the thinking_level kwarg for the agent model, if it applies.

    Args:
        settings: Application settings.

    Returns:
        {"thinking_level": level} or an empty dict.
    """
    level = settings.llm_thinking_level
    if level is None or not supports_thinking_level(settings.google_model):
        return {}
    return {"thinking_level": level}


def router_chat_model(
    chat_model: BaseChatModel, settings: Settings | None = None
) -> Runnable[Any, Any]:
    """Return the shared model bound to the router's (lower) thinking level.

    Intent classification outputs one label, so Gemini 3+ thinking only
    adds latency there. Non-Gemini models and test doubles pass through.

    Args:
        chat_model: The shared chat model.
        settings: Optional settings override. Uses get_settings() if None.

    Returns:
        A runnable to invoke for routing.

    Example:
        >>> router_model = router_chat_model(create_chat_model())
    """
    settings = settings or get_settings()
    level = settings.router_thinking_level
    if not isinstance(chat_model, ChatGoogleGenerativeAI) or level is None:
        return chat_model
    google_model: Any = chat_model
    if not supports_thinking_level(str(google_model.model)):
        return cast(Runnable[Any, Any], google_model)
    return cast(Runnable[Any, Any], google_model.bind(thinking_level=level))


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
    chat_model_factory: Any = ChatGoogleGenerativeAI
    return cast(
        BaseChatModel,
        chat_model_factory(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
            **_google_thinking_kwargs(settings),
        ),
    )


def _ollama_client_kwargs(settings: Settings) -> dict[str, Any]:
    """Build ChatOllama constructor kwargs from settings.

    Adds a bearer Authorization header via client_kwargs when an Ollama
    Cloud API key is configured, so the same provider works for local
    Ollama (no key) and Ollama Cloud (https://ollama.com).

    Args:
        settings: Application settings with OLLAMA config.

    Returns:
        kwargs dict for ChatOllama.
    """
    kwargs: dict[str, Any] = {
        "model": settings.ollama_model,
        "base_url": settings.ollama_base_url,
        "temperature": settings.llm_temperature,
    }
    if settings.ollama_api_key:
        kwargs["client_kwargs"] = {
            "headers": {"Authorization": f"Bearer {settings.ollama_api_key}"}
        }
    return kwargs


def create_ollama_chat_model(settings: Settings) -> BaseChatModel:
    """Create a ChatOllama instance from settings.

    Supports local Ollama (no key) and Ollama Cloud (bearer API key).

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
    return chat_ollama_cls(**_ollama_client_kwargs(settings))  # type: ignore[no-any-return]


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
    return ChatOpenAI  # type: ignore[no-any-return, unused-ignore]


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


def create_chat_model(
    settings: Settings | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """Create a LangChain ChatModel based on the configured provider.

    Args:
        settings: Optional settings override. Uses get_settings() if None.
        temperature: Optional per-agent temperature override. When set,
            it replaces llm_temperature for this model only.

    Returns:
        A LangChain BaseChatModel instance (Google or OLLAMA).

    Raises:
        ValueError: If the provider is unsupported or config is missing.

    Example:
        >>> model = create_chat_model(temperature=0.3)
    """
    if settings is None:
        settings = get_settings()
    if temperature is not None:
        settings = settings.model_copy(update={"llm_temperature": temperature})
    if settings.llm_provider == "google":
        return create_google_chat_model(settings)
    if settings.llm_provider == "ollama":
        return create_ollama_chat_model(settings)
    if settings.llm_provider == "openrouter":
        return create_openrouter_chat_model(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")


def create_ocr_google_chat_model(settings: Settings) -> BaseChatModel:
    """Create a ChatGoogleGenerativeAI vision model from OCR settings.

    Args:
        settings: Application settings with OCR config.

    Returns:
        Configured ChatGoogleGenerativeAI instance for OCR.

    Raises:
        ValueError: If ocr_api_key is not set.

    Example:
        >>> model = create_ocr_google_chat_model(settings)
    """
    if not settings.ocr_api_key:
        raise ValueError("ocr_api_key is required when ocr_provider is 'google'.")
    logger.info("Creating Google OCR ChatModel with model=%s", settings.ocr_model)
    chat_model_factory: Any = ChatGoogleGenerativeAI
    return cast(
        BaseChatModel,
        chat_model_factory(
            model=settings.ocr_model,
            google_api_key=settings.ocr_api_key,
            temperature=settings.ocr_temperature,
            max_output_tokens=settings.ocr_max_tokens,
            request_timeout=settings.ocr_timeout,
        ),
    )


def _ocr_ollama_client_kwargs(settings: Settings) -> dict[str, Any]:
    """Build ChatOllama constructor kwargs from OCR settings.

    Args:
        settings: Application settings with OCR config.

    Returns:
        kwargs dict for ChatOllama (with bearer header when key is set).
    """
    kwargs: dict[str, Any] = {
        "model": settings.ocr_model,
        "base_url": settings.ocr_base_url,
        "temperature": settings.ocr_temperature,
        "num_predict": settings.ocr_max_tokens,
        "timeout": settings.ocr_timeout,
    }
    if settings.ocr_api_key:
        kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {settings.ocr_api_key}"}}
    return kwargs


def create_ocr_ollama_chat_model(settings: Settings) -> BaseChatModel:
    """Create a ChatOllama vision model from OCR settings.

    Supports local Ollama (no key) and Ollama Cloud (bearer API key).

    Args:
        settings: Application settings with OCR config.

    Returns:
        Configured ChatOllama instance for OCR.

    Raises:
        ImportError: If langchain-ollama is not installed.

    Example:
        >>> model = create_ocr_ollama_chat_model(settings)
    """
    chat_ollama_cls = import_chat_ollama()
    logger.info("Creating OLLAMA OCR ChatModel with model=%s", settings.ocr_model)
    return chat_ollama_cls(**_ocr_ollama_client_kwargs(settings))  # type: ignore[no-any-return]


def create_ocr_openrouter_chat_model(settings: Settings) -> BaseChatModel:
    """Create a ChatOpenAI vision model for OCR via OpenRouter.

    Reuses OPENROUTER_API_KEY when OCR_API_KEY is not set so one key can
    cover both the chat agent and document scanning.

    Args:
        settings: Application settings with OCR config.

    Returns:
        Configured ChatOpenAI instance pointing at OpenRouter.

    Raises:
        ValueError: If neither ocr_api_key nor openrouter_api_key is set.
        ImportError: If langchain-openai is not installed.

    Example:
        >>> model = create_ocr_openrouter_chat_model(settings)
    """
    api_key = settings.ocr_api_key or settings.openrouter_api_key
    if not api_key:
        raise ValueError("OCR provider 'openrouter' requires ocr_api_key or openrouter_api_key.")
    chat_openai_cls = import_chat_openai()
    logger.info(
        "Creating OpenRouter OCR ChatModel with model=%s",
        settings.ocr_model,
    )
    return chat_openai_cls(  # type: ignore[no-any-return]
        model=settings.ocr_model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=settings.ocr_temperature,
        max_tokens=settings.ocr_max_tokens,
        request_timeout=settings.ocr_timeout,
    )


def create_ocr_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Create a LangChain vision ChatModel for OCR from OCR_* settings.

    Uses the dedicated OCR provider/model/key config, independent of the
    main LLM_PROVIDER so document scanning can use a vision model (e.g.
    qwen3.5:27b on Ollama Cloud) without affecting chat agents.

    Args:
        settings: Optional settings override. Uses get_settings() if None.

    Returns:
        A LangChain BaseChatModel instance for OCR (Google, Ollama or
        OpenRouter).

    Raises:
        ValueError: If the OCR provider is unsupported or config is missing.

    Example:
        >>> model = create_ocr_chat_model()
    """
    if settings is None:
        settings = get_settings()
    if settings.ocr_provider == "google":
        return create_ocr_google_chat_model(settings)
    if settings.ocr_provider == "ollama":
        return create_ocr_ollama_chat_model(settings)
    if settings.ocr_provider == "openrouter":
        return create_ocr_openrouter_chat_model(settings)
    raise ValueError(f"Unsupported OCR provider: {settings.ocr_provider}")
