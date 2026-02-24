"""RAG (Retrieval-Augmented Generation) implementations.

Provides document loading, embedding, vector storage, and retrieval
for the Thai personal finance knowledge base.
"""

from finance_ai.rag.document_models import (
    DocumentChunk,
    DocumentMetadata,
    RetrievalResult,
)
from finance_ai.rag.embedding_factory import create_embeddings
from finance_ai.rag.knowledge_base import (
    KnowledgeBaseManager,
    create_knowledge_base_manager,
)
from finance_ai.rag.retriever import FinanceRetriever
from finance_ai.rag.vector_store import FinanceVectorStore

__all__ = [
    "DocumentChunk",
    "DocumentMetadata",
    "RetrievalResult",
    "create_embeddings",
    "FinanceVectorStore",
    "FinanceRetriever",
    "KnowledgeBaseManager",
    "create_knowledge_base_manager",
]
