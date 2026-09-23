"""Factory for creating embedding model instances.

Supports Google Generative AI Embeddings and sentence-transformers (optional).
Follows the same factory pattern as llm_factory.py.
"""

import importlib
from typing import Any, cast

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from finance_ai.core.config import Settings, get_settings
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)


def import_sentence_transformer_embeddings() -> type:
    """Import HuggingFaceEmbeddings class, raising ImportError if not installed.

    Returns:
        The HuggingFaceEmbeddings class.

    Raises:
        ImportError: If langchain-community is not installed.
    """
    try:
        module = importlib.import_module("langchain_community.embeddings")
    except ImportError as exc:
        raise ImportError(
            "langchain-community is required for sentence-transformers embeddings. "
            "Install with: pip install langchain-community sentence-transformers"
        ) from exc
    return module.HuggingFaceEmbeddings  # type: ignore[no-any-return]


def create_google_embeddings(settings: Settings) -> Embeddings:
    """Create GoogleGenerativeAIEmbeddings from settings.

    Args:
        settings: Application settings with Google API config.

    Returns:
        Configured GoogleGenerativeAIEmbeddings instance.

    Raises:
        ValueError: If google_api_key is not set.

    Example:
        >>> embeddings = create_google_embeddings(settings)
    """
    if not settings.google_api_key:
        raise ValueError("google_api_key is required when rag_embedding_provider is 'google'.")
    logger.info("Creating Google Embeddings with model=%s", settings.rag_embedding_model)
    # Call through Any + cast: langchain-google-genai versions disagree on
    # the constructor's typing (str vs SecretStr vs dynamic pydantic fields).
    embeddings_factory: Any = GoogleGenerativeAIEmbeddings
    return cast(
        Embeddings,
        embeddings_factory(
            model=settings.rag_embedding_model,
            google_api_key=settings.google_api_key,
        ),
    )


def create_sentence_transformer_embeddings(settings: Settings) -> Embeddings:
    """Create HuggingFaceEmbeddings from settings.

    Args:
        settings: Application settings with sentence-transformer config.

    Returns:
        Configured HuggingFaceEmbeddings instance.

    Raises:
        ImportError: If langchain-community is not installed.

    Example:
        >>> embeddings = create_sentence_transformer_embeddings(settings)
    """
    huggingface_cls = import_sentence_transformer_embeddings()
    logger.info(
        "Creating SentenceTransformer Embeddings with model=%s",
        settings.rag_sentence_transformer_model,
    )
    return huggingface_cls(  # type: ignore[no-any-return]
        model_name=settings.rag_sentence_transformer_model,
    )


def import_openai_embeddings() -> type:
    """Import OpenAIEmbeddings class, raising ImportError if not installed.

    Returns:
        The OpenAIEmbeddings class.

    Raises:
        ImportError: If langchain-openai is not installed.
    """
    try:
        module = importlib.import_module("langchain_openai")
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for OpenRouter embeddings. "
            "Install with: pip install langchain-openai"
        ) from exc
    return module.OpenAIEmbeddings  # type: ignore[no-any-return]


def create_openrouter_embeddings(settings: Settings) -> Embeddings:
    """Create OpenAI-compatible embeddings pointed at the OpenRouter endpoint.

    Args:
        settings: Application settings with OpenRouter config.

    Returns:
        Configured Embeddings instance.

    Raises:
        ValueError: If openrouter_api_key is not set.

    Example:
        >>> embeddings = create_openrouter_embeddings(settings)
    """
    if not settings.openrouter_api_key:
        raise ValueError(
            "openrouter_api_key is required when rag_embedding_provider is 'openrouter'."
        )
    openai_embeddings_cls = import_openai_embeddings()
    logger.info(
        "Creating OpenRouter Embeddings with model=%s",
        settings.rag_embedding_model,
    )
    return openai_embeddings_cls(  # type: ignore[no-any-return]
        model=settings.rag_embedding_model,
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        # OpenRouter embeddings accept raw strings only — token-array input
        # (the OpenAI default) is rejected, and tiktoken is wrong for
        # non-OpenAI models anyway.
        check_embedding_ctx_length=False,
    )


def create_embeddings(settings: Settings | None = None) -> Embeddings:
    """Create an embedding model based on the configured provider.

    Args:
        settings: Optional settings override. Uses get_settings() if None.

    Returns:
        A LangChain Embeddings instance (Google or sentence-transformers).

    Raises:
        ValueError: If the provider is unsupported or config is missing.

    Example:
        >>> embeddings = create_embeddings()
    """
    if settings is None:
        settings = get_settings()
    if settings.rag_embedding_provider == "google":
        return create_google_embeddings(settings)
    if settings.rag_embedding_provider == "sentence_transformers":
        return create_sentence_transformer_embeddings(settings)
    if settings.rag_embedding_provider == "openrouter":
        return create_openrouter_embeddings(settings)
    raise ValueError(f"Unsupported embedding provider: {settings.rag_embedding_provider}")
