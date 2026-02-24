"""Tests for RAG text splitter."""

import pytest

from finance_ai.rag.document_models import DocumentMetadata
from finance_ai.rag.text_splitter import split_text_into_chunks


@pytest.fixture
def tax_metadata() -> DocumentMetadata:
    """Create sample tax metadata for tests."""
    return DocumentMetadata(
        source_file="tax_guide.md",
        domain="tax",
        title="คู่มือภาษี",
    )


class TestSplitTextIntoChunks:
    """Tests for split_text_into_chunks function."""

    def test_splits_long_text_into_multiple_chunks(self, tax_metadata: DocumentMetadata) -> None:
        """Test that long text is split into multiple chunks."""
        text = "ย่อหน้าที่หนึ่ง " * 100 + "\n\n" + "ย่อหน้าที่สอง " * 100
        chunks = split_text_into_chunks(text, tax_metadata, chunk_size=200)
        assert len(chunks) > 1

    def test_short_text_produces_single_chunk(self, tax_metadata: DocumentMetadata) -> None:
        """Test that text shorter than chunk_size produces a single chunk."""
        text = "ข้อความสั้นๆ"
        chunks = split_text_into_chunks(text, tax_metadata, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_chunks_have_correct_metadata(self, tax_metadata: DocumentMetadata) -> None:
        """Test that each chunk inherits the source metadata."""
        text = "เนื้อหาทดสอบ " * 200
        chunks = split_text_into_chunks(text, tax_metadata, chunk_size=200)
        for chunk in chunks:
            assert chunk.metadata == tax_metadata

    def test_chunks_have_sequential_indices(self, tax_metadata: DocumentMetadata) -> None:
        """Test that chunk indices are sequential starting from 0."""
        text = "เนื้อหาทดสอบ " * 200
        chunks = split_text_into_chunks(text, tax_metadata, chunk_size=200)
        for index, chunk in enumerate(chunks):
            assert chunk.chunk_index == index

    def test_chunks_have_unique_ids(self, tax_metadata: DocumentMetadata) -> None:
        """Test that each chunk has a unique chunk_id."""
        text = "เนื้อหาทดสอบ " * 200
        chunks = split_text_into_chunks(text, tax_metadata, chunk_size=200)
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_ids_are_deterministic(self, tax_metadata: DocumentMetadata) -> None:
        """Test that same input produces same chunk_ids."""
        text = "เนื้อหาทดสอบ " * 200
        chunks_first = split_text_into_chunks(text, tax_metadata, chunk_size=200)
        chunks_second = split_text_into_chunks(text, tax_metadata, chunk_size=200)
        ids_first = [c.chunk_id for c in chunks_first]
        ids_second = [c.chunk_id for c in chunks_second]
        assert ids_first == ids_second

    def test_empty_text_returns_empty_list(self, tax_metadata: DocumentMetadata) -> None:
        """Test that empty text returns an empty list."""
        chunks = split_text_into_chunks("", tax_metadata, chunk_size=500)
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self, tax_metadata: DocumentMetadata) -> None:
        """Test that whitespace-only text returns an empty list."""
        chunks = split_text_into_chunks("   \n\n  ", tax_metadata, chunk_size=500)
        assert chunks == []

    def test_respects_chunk_overlap(self, tax_metadata: DocumentMetadata) -> None:
        """Test that chunks overlap correctly."""
        text = "A" * 300 + "\n\n" + "B" * 300
        chunks = split_text_into_chunks(text, tax_metadata, chunk_size=200, chunk_overlap=50)
        assert len(chunks) >= 2
