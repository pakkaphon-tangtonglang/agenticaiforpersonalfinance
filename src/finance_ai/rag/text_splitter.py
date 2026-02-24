"""Text splitting for RAG document processing.

Splits documents into overlapping chunks suitable for embedding and retrieval.
"""

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from finance_ai.rag.document_models import DocumentChunk, DocumentMetadata


def _generate_chunk_id(content: str, source_file: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID from content, source, and index.

    Args:
        content: The text content of the chunk.
        source_file: The source filename.
        chunk_index: The index of the chunk within the document.

    Returns:
        A hex digest string uniquely identifying this chunk.

    Example:
        >>> _generate_chunk_id("hello", "test.md", 0)
        'a8f5f167...'
    """
    raw = f"{source_file}::{chunk_index}::{content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def split_text_into_chunks(
    text: str,
    metadata: DocumentMetadata,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """Split text into overlapping chunks with metadata.

    Uses paragraph and line breaks as separators, suitable for Thai text.

    Args:
        text: The raw text to split.
        metadata: Metadata to attach to each chunk.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of DocumentChunk objects. Empty list if text is blank.

    Example:
        >>> chunks = split_text_into_chunks("long text...", metadata)
        >>> len(chunks) > 0
        True
    """
    stripped = text.strip()
    if not stripped:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    texts = splitter.split_text(stripped)
    return _build_chunks(texts, metadata)


def _build_chunks(texts: list[str], metadata: DocumentMetadata) -> list[DocumentChunk]:
    """Build DocumentChunk objects from split text pieces.

    Args:
        texts: List of text strings from the splitter.
        metadata: Metadata to attach to each chunk.

    Returns:
        List of DocumentChunk objects with sequential indices.
    """
    return [
        DocumentChunk(
            chunk_id=_generate_chunk_id(text, metadata.source_file, index),
            content=text,
            metadata=metadata,
            chunk_index=index,
        )
        for index, text in enumerate(texts)
    ]
