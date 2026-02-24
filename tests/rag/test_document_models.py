"""Tests for RAG document models."""

from pydantic import ValidationError
import pytest

from finance_ai.rag.document_models import (
    DocumentChunk,
    DocumentMetadata,
    RetrievalResult,
)


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_valid_tax_metadata(self) -> None:
        """Test creating valid tax document metadata."""
        metadata = DocumentMetadata(
            source_file="tax_deductions_guide.md",
            domain="tax",
            title="คู่มือค่าลดหย่อนภาษี",
        )
        assert metadata.source_file == "tax_deductions_guide.md"
        assert metadata.domain == "tax"
        assert metadata.title == "คู่มือค่าลดหย่อนภาษี"
        assert metadata.language == "th"

    def test_valid_investment_metadata(self) -> None:
        """Test creating valid investment document metadata."""
        metadata = DocumentMetadata(
            source_file="investment_thai_stocks.md",
            domain="investment",
            title="การลงทุนหุ้นไทย",
            language="en",
        )
        assert metadata.domain == "investment"
        assert metadata.language == "en"

    def test_valid_expense_metadata(self) -> None:
        """Test creating valid expense document metadata."""
        metadata = DocumentMetadata(
            source_file="expense_budgeting_guide.md",
            domain="expense",
            title="แนวทางจัดทำงบประมาณ",
        )
        assert metadata.domain == "expense"

    def test_valid_general_metadata(self) -> None:
        """Test creating valid general domain metadata."""
        metadata = DocumentMetadata(
            source_file="general_info.md",
            domain="general",
            title="ข้อมูลทั่วไป",
        )
        assert metadata.domain == "general"

    def test_invalid_domain_raises_error(self) -> None:
        """Test that an invalid domain raises ValidationError."""
        with pytest.raises(ValidationError):
            DocumentMetadata(
                source_file="test.md",
                domain="invalid_domain",  # type: ignore[arg-type]
                title="Test",
            )

    def test_default_language_is_thai(self) -> None:
        """Test that default language is Thai."""
        metadata = DocumentMetadata(
            source_file="test.md",
            domain="tax",
            title="Test",
        )
        assert metadata.language == "th"

    def test_serialization_round_trip(self) -> None:
        """Test model serialization and deserialization."""
        metadata = DocumentMetadata(
            source_file="tax_guide.md",
            domain="tax",
            title="คู่มือภาษี",
        )
        data = metadata.model_dump()
        restored = DocumentMetadata(**data)
        assert restored == metadata


class TestDocumentChunk:
    """Tests for DocumentChunk model."""

    def test_valid_chunk(self) -> None:
        """Test creating a valid document chunk."""
        metadata = DocumentMetadata(
            source_file="tax_guide.md",
            domain="tax",
            title="คู่มือภาษี",
        )
        chunk = DocumentChunk(
            chunk_id="abc123",
            content="ภาษีเงินได้บุคคลธรรมดา",
            metadata=metadata,
            chunk_index=0,
        )
        assert chunk.chunk_id == "abc123"
        assert chunk.content == "ภาษีเงินได้บุคคลธรรมดา"
        assert chunk.metadata.domain == "tax"
        assert chunk.chunk_index == 0

    def test_chunk_requires_content(self) -> None:
        """Test that chunk requires non-empty content."""
        metadata = DocumentMetadata(source_file="test.md", domain="tax", title="Test")
        with pytest.raises(ValidationError):
            DocumentChunk(
                chunk_id="abc",
                content="",
                metadata=metadata,
                chunk_index=0,
            )

    def test_chunk_index_must_be_non_negative(self) -> None:
        """Test that chunk_index must be non-negative."""
        metadata = DocumentMetadata(source_file="test.md", domain="tax", title="Test")
        with pytest.raises(ValidationError):
            DocumentChunk(
                chunk_id="abc",
                content="some content",
                metadata=metadata,
                chunk_index=-1,
            )

    def test_serialization_round_trip(self) -> None:
        """Test model serialization and deserialization."""
        metadata = DocumentMetadata(
            source_file="tax_guide.md",
            domain="tax",
            title="คู่มือภาษี",
        )
        chunk = DocumentChunk(
            chunk_id="abc123",
            content="ภาษีเงินได้",
            metadata=metadata,
            chunk_index=2,
        )
        data = chunk.model_dump()
        restored = DocumentChunk(**data)
        assert restored == chunk


class TestRetrievalResult:
    """Tests for RetrievalResult model."""

    def test_valid_result(self) -> None:
        """Test creating a valid retrieval result."""
        result = RetrievalResult(
            content="ค่าลดหย่อนส่วนตัว 60,000 บาท",
            source_file="tax_deductions_guide.md",
            domain="tax",
            relevance_score=0.85,
        )
        assert result.content == "ค่าลดหย่อนส่วนตัว 60,000 บาท"
        assert result.source_file == "tax_deductions_guide.md"
        assert result.domain == "tax"
        assert result.relevance_score == 0.85

    def test_relevance_score_must_be_non_negative(self) -> None:
        """Test that relevance score must be non-negative."""
        with pytest.raises(ValidationError):
            RetrievalResult(
                content="test",
                source_file="test.md",
                domain="tax",
                relevance_score=-0.1,
            )

    def test_serialization_round_trip(self) -> None:
        """Test model serialization and deserialization."""
        result = RetrievalResult(
            content="test content",
            source_file="test.md",
            domain="investment",
            relevance_score=0.95,
        )
        data = result.model_dump()
        restored = RetrievalResult(**data)
        assert restored == result
