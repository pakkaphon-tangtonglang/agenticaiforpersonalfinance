"""Pydantic models for RAG document processing pipeline."""

from typing import Literal

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for a knowledge base document.

    Attributes:
        source_file: Original filename (e.g., "tax_deductions_guide.md").
        domain: Knowledge domain for filtering retrieval results.
        title: Human-readable document title.
        language: Document language code.

    Example:
        >>> metadata = DocumentMetadata(
        ...     source_file="tax_guide.md",
        ...     domain="tax",
        ...     title="คู่มือภาษี",
        ... )
    """

    source_file: str
    domain: Literal["tax", "expense", "investment", "general"]
    title: str
    language: str = Field(default="th")


class DocumentChunk(BaseModel):
    """A single chunk of text with metadata, ready for embedding.

    Attributes:
        chunk_id: Deterministic ID derived from content and source.
        content: The text content of this chunk.
        metadata: Document metadata inherited from the source file.
        chunk_index: Position of this chunk within the source document.

    Example:
        >>> chunk = DocumentChunk(
        ...     chunk_id="abc123",
        ...     content="ภาษีเงินได้บุคคลธรรมดา",
        ...     metadata=metadata,
        ...     chunk_index=0,
        ... )
    """

    chunk_id: str
    content: str = Field(min_length=1)
    metadata: DocumentMetadata
    chunk_index: int = Field(ge=0)


class RetrievalResult(BaseModel):
    """A single result from a retrieval query.

    Attributes:
        content: The retrieved text content.
        source_file: Original source filename.
        domain: Knowledge domain of the result.
        relevance_score: Similarity or distance score (higher is more relevant).

    Example:
        >>> result = RetrievalResult(
        ...     content="ค่าลดหย่อนส่วนตัว 60,000 บาท",
        ...     source_file="tax_deductions_guide.md",
        ...     domain="tax",
        ...     relevance_score=0.85,
        ... )
    """

    content: str
    source_file: str
    domain: str
    relevance_score: float = Field(ge=0.0)
