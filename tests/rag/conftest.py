"""Shared test fixtures for RAG tests."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from finance_ai.rag.document_models import DocumentChunk, DocumentMetadata, RetrievalResult


@pytest.fixture
def sample_tax_metadata() -> DocumentMetadata:
    """Create sample tax document metadata."""
    return DocumentMetadata(
        source_file="tax_guide.md",
        domain="tax",
        title="คู่มือภาษี",
    )


@pytest.fixture
def sample_investment_metadata() -> DocumentMetadata:
    """Create sample investment document metadata."""
    return DocumentMetadata(
        source_file="investment_stocks.md",
        domain="investment",
        title="การลงทุนหุ้นไทย",
    )


@pytest.fixture
def sample_tax_chunks(sample_tax_metadata: DocumentMetadata) -> list[DocumentChunk]:
    """Create sample tax document chunks."""
    return [
        DocumentChunk(
            chunk_id="tax_chunk_0",
            content="ภาษีเงินได้บุคคลธรรมดา",
            metadata=sample_tax_metadata,
            chunk_index=0,
        ),
        DocumentChunk(
            chunk_id="tax_chunk_1",
            content="ค่าลดหย่อนส่วนตัว 60,000 บาท",
            metadata=sample_tax_metadata,
            chunk_index=1,
        ),
    ]


@pytest.fixture
def sample_retrieval_results() -> list[RetrievalResult]:
    """Create sample retrieval results."""
    return [
        RetrievalResult(
            content="ค่าลดหย่อนส่วนตัว 60,000 บาท",
            source_file="tax_deductions_guide.md",
            domain="tax",
            relevance_score=0.92,
        ),
    ]


@pytest.fixture
def mock_embeddings() -> MagicMock:
    """Create a mock embeddings instance."""
    return MagicMock()


@pytest.fixture
def temp_knowledge_directory(tmp_path: Path) -> Path:
    """Create a temporary directory with sample knowledge base files."""
    (tmp_path / "tax_guide.md").write_text(
        "# คู่มือภาษี\n\nภาษีเงินได้บุคคลธรรมดา\n",
        encoding="utf-8",
    )
    (tmp_path / "investment_stocks.md").write_text(
        "# หุ้นไทย\n\nข้อมูลการลงทุนในหุ้น SET\n",
        encoding="utf-8",
    )
    return tmp_path
