"""ChromaDB-backed vector store for finance knowledge base.

Wraps langchain_chroma.Chroma with typed methods for adding, searching,
and managing document chunks.
"""

from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from finance_ai.core.logging import get_logger
from finance_ai.rag.document_models import DocumentChunk, RetrievalResult

logger = get_logger(__name__)


class FinanceVectorStore:
    """ChromaDB-backed vector store for finance knowledge base.

    Attributes:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Path for ChromaDB persistence.

    Example:
        >>> store = FinanceVectorStore(embeddings=embeddings)
        >>> store.add_chunks(chunks)
        >>> results = store.search("ค่าลดหย่อนภาษี")
    """

    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "finance_knowledge",
    ) -> None:
        """Initialize with embeddings and ChromaDB config.

        Args:
            embeddings: LangChain Embeddings instance for vectorization.
            persist_directory: Path for ChromaDB persistence.
            collection_name: Name of the ChromaDB collection.
        """
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Add document chunks to the vector store.

        Args:
            chunks: List of DocumentChunk objects to add.

        Returns:
            Number of chunks added.

        Example:
            >>> count = store.add_chunks(chunks)
        """
        if not chunks:
            return 0
        texts, metadatas, ids = _prepare_chunk_data(chunks)
        self._store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info("Added %d chunks to vector store", len(chunks))
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        domain_filter: str | None = None,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[RetrievalResult]:
        """Search for relevant chunks by similarity.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.
            domain_filter: Optional domain to filter results (e.g., "tax").
            metadata_filter: Optional exact-match metadata (e.g., {"guru_id": "buffett"}).

        Returns:
            List of RetrievalResult objects sorted by relevance.

        Example:
            >>> results = store.search("ค่าลดหย่อน", domain_filter="tax")
        """
        kwargs: dict[str, Any] = {"query": query, "k": top_k}
        chroma_filter = _build_chroma_filter(domain_filter, metadata_filter)
        if chroma_filter:
            kwargs["filter"] = chroma_filter
        docs_with_scores = self._store.similarity_search_with_score(**kwargs)
        return _convert_search_results(docs_with_scores)

    def clear_collection(self) -> None:
        """Delete all documents and recreate the collection.

        Example:
            >>> store.clear_collection()
        """
        self._store.reset_collection()
        logger.info("Cleared vector store collection")

    def get_document_count(self) -> int:
        """Return the number of documents in the collection.

        Returns:
            Number of documents stored.

        Example:
            >>> count = store.get_document_count()
        """
        return int(self._store._collection.count())


def _build_chroma_filter(
    domain_filter: str | None,
    metadata_filter: dict[str, str] | None,
) -> dict[str, Any] | None:
    """Combine domain and metadata filters into one Chroma where-clause.

    Args:
        domain_filter: Optional domain value.
        metadata_filter: Optional exact-match metadata pairs.

    Returns:
        A single condition, an "$and" of several, or None when empty.

    Example:
        >>> _build_chroma_filter("guru", {"guru_id": "dalio"})
        {'$and': [{'domain': 'guru'}, {'guru_id': 'dalio'}]}
    """
    conditions: list[dict[str, Any]] = []
    if domain_filter:
        conditions.append({"domain": domain_filter})
    for key, value in (metadata_filter or {}).items():
        conditions.append({key: value})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _chunk_to_chroma_metadata(chunk: DocumentChunk) -> dict[str, str]:
    """Build Chroma metadata for one chunk, skipping unset optional fields.

    Chroma rejects None values, so guru_id and source_url are only
    included when present.

    Args:
        chunk: The chunk to describe.

    Returns:
        Flat string metadata dict.
    """
    metadata = {
        "source_file": chunk.metadata.source_file,
        "domain": chunk.metadata.domain,
        "title": chunk.metadata.title,
    }
    if chunk.metadata.guru_id:
        metadata["guru_id"] = chunk.metadata.guru_id
    if chunk.metadata.source_url:
        metadata["source_url"] = chunk.metadata.source_url
    return metadata


def _prepare_chunk_data(
    chunks: list[DocumentChunk],
) -> tuple[list[str], list[dict[str, str]], list[str]]:
    """Extract texts, metadatas, and ids from chunks for ChromaDB.

    Args:
        chunks: List of DocumentChunk objects.

    Returns:
        Tuple of (texts, metadatas, ids).
    """
    texts = [chunk.content for chunk in chunks]
    metadatas = [_chunk_to_chroma_metadata(chunk) for chunk in chunks]
    ids = [chunk.chunk_id for chunk in chunks]
    return texts, metadatas, ids


def _convert_search_results(
    docs_with_scores: list[tuple[Any, float]],
) -> list[RetrievalResult]:
    """Convert LangChain search results to RetrievalResult models.

    Args:
        docs_with_scores: List of (Document, score) tuples.

    Returns:
        List of RetrievalResult objects.
    """
    return [
        RetrievalResult(
            content=doc.page_content,
            source_file=doc.metadata.get("source_file", ""),
            domain=doc.metadata.get("domain", "general"),
            relevance_score=score,
            guru_id=doc.metadata.get("guru_id"),
            source_url=doc.metadata.get("source_url"),
        )
        for doc, score in docs_with_scores
    ]
