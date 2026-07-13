"""RAG retrieval tool for LangGraph agents.

Provides a @tool-decorated function that agents can use to search
the Thai personal finance knowledge base.
"""

from typing import Any

from langchain_core.tools import tool

from finance_ai.core.logging import get_logger
from finance_ai.rag.retriever import FinanceRetriever

logger = get_logger(__name__)

_retriever: FinanceRetriever | None = None


def get_retriever() -> FinanceRetriever:
    """Get or create the shared FinanceRetriever instance.

    Lazily initializes the retriever from application settings on first call.

    Returns:
        Configured FinanceRetriever instance.

    Example:
        >>> retriever = get_retriever()
    """
    global _retriever  # noqa: PLW0603
    if _retriever is None:
        _retriever = _create_retriever_from_settings()
    return _retriever


def _create_retriever_from_settings() -> FinanceRetriever:
    """Create a FinanceRetriever from application settings.

    Returns:
        Configured FinanceRetriever instance.
    """
    from finance_ai.core.config import get_settings
    from finance_ai.rag.embedding_factory import create_embeddings
    from finance_ai.rag.vector_store import FinanceVectorStore

    settings = get_settings()
    embeddings = create_embeddings(settings)
    vector_store = FinanceVectorStore(
        embeddings=embeddings,
        persist_directory=settings.rag_chroma_persist_directory,
        collection_name=settings.rag_chroma_collection_name,
    )
    return FinanceRetriever(
        vector_store=vector_store,
        top_k=settings.rag_search_top_k,
    )


def reset_retriever() -> None:
    """Reset the shared retriever instance (for testing).

    Example:
        >>> reset_retriever()
    """
    global _retriever  # noqa: PLW0603
    _retriever = None


@tool
def search_finance_knowledge(
    query: str,
    domain: str = "",
) -> dict[str, Any]:
    """Search the Thai personal finance knowledge base.

    Use this tool when you need reference information about Thai tax laws,
    investment rules, expense guidelines, or general financial planning advice.

    Args:
        query: Natural language search query (Thai or English).
        domain: Optional domain filter ("tax", "expense", "investment", or "" for all).

    Returns:
        Dict with 'results' list and 'context' formatted string.
    """
    retriever = get_retriever()
    domain_filter = domain if domain else None
    results = retriever.retrieve(query, domain=domain_filter)
    context = retriever.retrieve_as_context(query, domain=domain_filter)
    return {
        "results": [r.model_dump() for r in results],
        "context": context,
    }
