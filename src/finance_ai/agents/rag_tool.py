"""RAG retrieval tool for LangGraph agents.

Provides a @tool-decorated function that agents can use to search
the Thai personal finance knowledge base.
"""

import threading
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.citation_sources import (
    build_citations,
    collect_turn_citations,
    format_citations_as_context,
)
from finance_ai.core.logging import get_logger
from finance_ai.rag.retriever import FinanceRetriever

logger = get_logger(__name__)

_retriever: FinanceRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> FinanceRetriever:
    """Get or create the shared FinanceRetriever instance.

    Thread-safe singleton using a lock to prevent race conditions.

    Returns:
        Configured FinanceRetriever instance.

    Example:
        >>> retriever = get_retriever()
    """
    global _retriever  # noqa: PLW0603
    with _retriever_lock:
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
    with _retriever_lock:
        _retriever = None


@tool
def search_finance_knowledge(
    query: str,
    domain: str = "",
    messages: Annotated[list[Any] | None, InjectedState("messages")] = None,
) -> dict[str, Any]:
    """Search the Thai personal finance knowledge base.

    Use this tool when you need reference information about Thai tax laws,
    investment rules, expense guidelines, or general financial planning advice.
    Cite what you use with the [number] shown before each passage.

    Args:
        query: Natural language search query (Thai or English).
        domain: Optional domain filter ("tax", "expense", "investment", or "" for all).
        messages: Conversation so far (injected from graph state) for citation numbering.

    Returns:
        Dict with numbered 'citations' and the 'context' passages.
    """
    results = get_retriever().retrieve(query, domain=domain or None)
    citations = build_citations(results, collect_turn_citations(messages or []))
    return {
        "citations": [citation.model_dump() for citation in citations],
        "context": format_citations_as_context(results, citations),
    }
