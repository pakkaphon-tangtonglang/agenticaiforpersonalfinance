"""High-level retriever for the finance knowledge base.

Combines vector store search with result formatting for LLM consumption.
"""

from finance_ai.rag.document_models import RetrievalResult
from finance_ai.rag.vector_store import FinanceVectorStore


class FinanceRetriever:
    """High-level retriever for the finance knowledge base.

    Combines vector store search with result formatting.

    Attributes:
        top_k: Default number of results to return.

    Example:
        >>> retriever = FinanceRetriever(vector_store=store)
        >>> results = retriever.retrieve("ค่าลดหย่อนภาษี", domain="tax")
    """

    def __init__(
        self,
        vector_store: FinanceVectorStore,
        top_k: int = 3,
    ) -> None:
        """Initialize with vector store and default top_k.

        Args:
            vector_store: The underlying vector store to search.
            top_k: Default number of results to return.
        """
        self._vector_store = vector_store
        self._top_k = top_k

    def retrieve(
        self,
        query: str,
        domain: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query.

        Args:
            query: Natural language search query.
            domain: Optional domain filter (e.g., "tax", "investment").

        Returns:
            List of RetrievalResult objects sorted by relevance.

        Example:
            >>> results = retriever.retrieve("ภาษีเงินได้", domain="tax")
        """
        return self._vector_store.search(query=query, top_k=self._top_k, domain_filter=domain)

    def retrieve_by_metadata(
        self,
        query: str,
        metadata_filter: dict[str, str],
    ) -> list[RetrievalResult]:
        """Retrieve documents whose metadata matches every given pair.

        Args:
            query: Natural language search query.
            metadata_filter: Exact-match metadata, e.g. {"guru_id": "buffett"}.

        Returns:
            List of RetrievalResult objects sorted by relevance.

        Example:
            >>> results = retriever.retrieve_by_metadata("ตลาดตก", {"guru_id": "buffett"})
        """
        return self._vector_store.search(
            query=query, top_k=self._top_k, metadata_filter=metadata_filter
        )

    def retrieve_as_context(
        self,
        query: str,
        domain: str | None = None,
    ) -> str:
        """Retrieve and format results as a context string for LLM.

        Args:
            query: Natural language search query.
            domain: Optional domain filter.

        Returns:
            Formatted context string with source attribution, or empty string.

        Example:
            >>> context = retriever.retrieve_as_context("ค่าลดหย่อน")
        """
        results = self.retrieve(query, domain=domain)
        if not results:
            return ""
        return format_results_as_context(results)


def _format_source_label(result: RetrievalResult) -> str:
    """Build the source label, appending the public URL when known.

    Args:
        result: A retrieval result.

    Returns:
        'file' or 'file | url'.
    """
    if result.source_url:
        return f"{result.source_file} | {result.source_url}"
    return result.source_file


def format_results_as_context(results: list[RetrievalResult]) -> str:
    """Format retrieval results into a context string with source attribution.

    Args:
        results: List of RetrievalResult objects.

    Returns:
        Formatted string with source headers and content.

    Example:
        >>> format_results_as_context(results)
        '[แหล่งที่มา: tax_guide.md]\\nค่าลดหย่อน...'
    """
    sections = [
        f"[แหล่งที่มา: {_format_source_label(result)}]\n{result.content}" for result in results
    ]
    return "\n\n".join(sections)
