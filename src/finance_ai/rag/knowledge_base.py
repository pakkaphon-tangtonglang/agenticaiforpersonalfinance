"""Knowledge base manager for RAG document indexing.

Orchestrates document loading, splitting, and indexing into the vector store.
"""

from pathlib import Path

from finance_ai.core.config import Settings, get_settings
from finance_ai.core.logging import get_logger
from finance_ai.rag.document_loader import discover_documents, load_document
from finance_ai.rag.embedding_factory import create_embeddings
from finance_ai.rag.text_splitter import split_text_into_chunks
from finance_ai.rag.vector_store import FinanceVectorStore

logger = get_logger(__name__)


class KnowledgeBaseManager:
    """Manages loading and indexing of knowledge base documents.

    Orchestrates the pipeline: load → split → embed → store.

    Example:
        >>> manager = KnowledgeBaseManager(vector_store=store)
        >>> count = manager.index_directory(Path("docs/knowledge_base"))
    """

    def __init__(
        self,
        vector_store: FinanceVectorStore,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        """Initialize with vector store and chunking parameters.

        Args:
            vector_store: The vector store to index documents into.
            chunk_size: Maximum chunk size in characters.
            chunk_overlap: Overlap between chunks in characters.
        """
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def index_document(self, file_path: Path) -> int:
        """Load, split, and index a single document.

        Args:
            file_path: Path to the document file.

        Returns:
            Number of chunks indexed.

        Example:
            >>> count = manager.index_document(Path("tax_guide.md"))
        """
        content, metadata = load_document(file_path)
        chunks = split_text_into_chunks(
            content,
            metadata,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        count = self._vector_store.add_chunks(chunks)
        logger.info("Indexed %s: %d chunks", file_path.name, count)
        return count

    def index_directory(self, directory: Path) -> int:
        """Index all documents in a directory.

        Args:
            directory: Path to the directory containing documents.

        Returns:
            Total number of chunks indexed across all documents.

        Example:
            >>> total = manager.index_directory(Path("docs/knowledge_base"))
        """
        paths = discover_documents(directory)
        total = sum(self.index_document(path) for path in paths)
        logger.info("Indexed %d documents, %d total chunks", len(paths), total)
        return total

    def get_document_count(self) -> int:
        """Return the number of documents in the vector store.

        Returns:
            Number of documents stored.

        Example:
            >>> count = manager.get_document_count()
        """
        return self._vector_store.get_document_count()

    def rebuild_index(self, directory: Path) -> int:
        """Clear the collection and re-index all documents.

        Args:
            directory: Path to the directory containing documents.

        Returns:
            Total number of chunks indexed after rebuild.

        Example:
            >>> total = manager.rebuild_index(Path("docs/knowledge_base"))
        """
        self._vector_store.clear_collection()
        logger.info("Cleared vector store, rebuilding index")
        return self.index_directory(directory)


def create_knowledge_base_manager(
    settings: Settings | None = None,
) -> KnowledgeBaseManager:
    """Create a fully configured KnowledgeBaseManager from settings.

    Args:
        settings: Optional settings override. Uses get_settings() if None.

    Returns:
        Configured KnowledgeBaseManager instance.

    Example:
        >>> manager = create_knowledge_base_manager()
    """
    if settings is None:
        settings = get_settings()
    embeddings = create_embeddings(settings)
    vector_store = FinanceVectorStore(
        embeddings=embeddings,
        persist_directory=settings.rag_chroma_persist_directory,
        collection_name=settings.rag_chroma_collection_name,
    )
    return KnowledgeBaseManager(
        vector_store=vector_store,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
