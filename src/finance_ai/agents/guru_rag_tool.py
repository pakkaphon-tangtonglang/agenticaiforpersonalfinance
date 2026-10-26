"""Guru knowledge search tool for the Guru agent.

Searches the separate "guru_knowledge" Chroma collection. The guru_id is
injected from graph state, so the model can only read the active guru's
sources and cannot quote another guru by mistake.
"""

import threading
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.core.config import get_settings
from finance_ai.core.logging import get_logger
from finance_ai.rag.retriever import FinanceRetriever, format_results_as_context
from finance_ai.tools.guru_gap_service import record_gap_if_needed

logger = get_logger(__name__)

NOT_FOUND_CONTEXT = "ไม่พบในแหล่งข้อมูลของกูรู ให้ตอบตามหลักการหลักในโปรไฟล์เท่านั้น"

_guru_retriever: FinanceRetriever | None = None
_guru_retriever_lock = threading.Lock()


def get_guru_retriever() -> FinanceRetriever:
    """Get or create the shared retriever for the guru collection.

    Returns:
        FinanceRetriever bound to the guru_knowledge collection.

    Example:
        >>> retriever = get_guru_retriever()
    """
    global _guru_retriever  # noqa: PLW0603  # pylint: disable=global-statement
    with _guru_retriever_lock:
        if _guru_retriever is None:
            _guru_retriever = _create_guru_retriever_from_settings()
    return _guru_retriever


def _create_guru_retriever_from_settings() -> FinanceRetriever:
    """Build a retriever for the guru collection from settings.

    Returns:
        Configured FinanceRetriever instance.
    """
    from finance_ai.rag.embedding_factory import create_embeddings
    from finance_ai.rag.vector_store import FinanceVectorStore

    settings = get_settings()
    vector_store = FinanceVectorStore(
        embeddings=create_embeddings(settings),
        persist_directory=settings.rag_chroma_persist_directory,
        collection_name=settings.rag_guru_collection_name,
    )
    return FinanceRetriever(vector_store=vector_store, top_k=settings.rag_search_top_k)


def reset_guru_retriever() -> None:
    """Reset the shared guru retriever (for testing).

    Example:
        >>> reset_guru_retriever()
    """
    global _guru_retriever  # noqa: PLW0603  # pylint: disable=global-statement
    with _guru_retriever_lock:
        _guru_retriever = None


@tool
def search_guru_knowledge(
    query: str,
    guru_id: Annotated[str, InjectedState("guru_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """Search the active guru's public writings, letters and interviews.

    Call this before answering so the answer follows the guru's documented
    views, and cite the [แหล่งที่มา] labels it returns.

    Args:
        query: What to look up (Thai or English).
        guru_id: Active guru (injected from graph state).
        db_session_factory: Session factory for gap logging (injected from graph state).

    Returns:
        Dict with 'results' list and 'context' string, or 'error'.
    """
    if not guru_id:
        return {"error": "No guru selected. guru_id is missing from the conversation state."}
    results = get_guru_retriever().retrieve_by_metadata(query, {"guru_id": guru_id})
    logger.info("Guru search for %s returned %d results", guru_id, len(results))
    record_gap_if_needed(
        db_session_factory, guru_id, query, results, get_settings().rag_guru_gap_distance_threshold
    )
    context = format_results_as_context(results) if results else NOT_FOUND_CONTEXT
    return {"results": [result.model_dump() for result in results], "context": context}
