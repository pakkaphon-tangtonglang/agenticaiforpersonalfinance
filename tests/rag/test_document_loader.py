"""Tests for RAG document loader."""

from pathlib import Path

import pytest

from finance_ai.rag.document_loader import (
    discover_documents,
    infer_domain_from_filename,
    load_document,
    load_markdown_file,
    load_text_file,
)


@pytest.fixture
def sample_markdown_file(tmp_path: Path) -> Path:
    """Create a sample markdown file for testing."""
    content = "# คู่มือภาษีเงินได้\n\nเนื้อหาเกี่ยวกับภาษี\n"
    file_path = tmp_path / "tax_guide.md"
    file_path.write_text(content, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Create a sample text file for testing."""
    content = "ข้อมูลการลงทุนในหุ้นไทย\n"
    file_path = tmp_path / "investment_stocks.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


class TestInferDomainFromFilename:
    """Tests for domain inference from filename."""

    def test_tax_prefix(self) -> None:
        """Test that filenames starting with tax_ are classified as tax."""
        assert infer_domain_from_filename("tax_deductions_guide.md") == "tax"

    def test_investment_prefix(self) -> None:
        """Test that filenames starting with investment_ are classified."""
        assert infer_domain_from_filename("investment_stocks.md") == "investment"

    def test_expense_prefix(self) -> None:
        """Test that filenames starting with expense_ are classified."""
        assert infer_domain_from_filename("expense_budgeting.md") == "expense"

    def test_unknown_prefix_defaults_to_general(self) -> None:
        """Test that unknown prefixes default to general."""
        assert infer_domain_from_filename("random_file.md") == "general"

    def test_no_prefix_defaults_to_general(self) -> None:
        """Test that files without prefix default to general."""
        assert infer_domain_from_filename("guide.md") == "general"


class TestLoadMarkdownFile:
    """Tests for loading markdown files."""

    def test_loads_content(self, sample_markdown_file: Path) -> None:
        """Test that markdown file content is loaded correctly."""
        content, metadata = load_markdown_file(sample_markdown_file)
        assert "เนื้อหาเกี่ยวกับภาษี" in content

    def test_extracts_title_from_heading(self, sample_markdown_file: Path) -> None:
        """Test that title is extracted from the first # heading."""
        _, metadata = load_markdown_file(sample_markdown_file)
        assert metadata.title == "คู่มือภาษีเงินได้"

    def test_infers_domain_from_filename(self, sample_markdown_file: Path) -> None:
        """Test that domain is inferred from the filename prefix."""
        _, metadata = load_markdown_file(sample_markdown_file)
        assert metadata.domain == "tax"

    def test_sets_source_file(self, sample_markdown_file: Path) -> None:
        """Test that source_file is set to the filename."""
        _, metadata = load_markdown_file(sample_markdown_file)
        assert metadata.source_file == "tax_guide.md"

    def test_file_without_heading_uses_filename_as_title(self, tmp_path: Path) -> None:
        """Test fallback when no heading is found."""
        file_path = tmp_path / "expense_misc.md"
        file_path.write_text("ไม่มีหัวข้อ\n", encoding="utf-8")
        _, metadata = load_markdown_file(file_path)
        assert metadata.title == "expense_misc"

    def test_empty_file_returns_empty_content(self, tmp_path: Path) -> None:
        """Test loading an empty markdown file."""
        file_path = tmp_path / "general_empty.md"
        file_path.write_text("", encoding="utf-8")
        content, metadata = load_markdown_file(file_path)
        assert content == ""
        assert metadata.domain == "general"


class TestLoadTextFile:
    """Tests for loading text files."""

    def test_loads_content(self, sample_text_file: Path) -> None:
        """Test that text file content is loaded correctly."""
        content, metadata = load_text_file(sample_text_file)
        assert "ข้อมูลการลงทุนในหุ้นไทย" in content

    def test_infers_domain(self, sample_text_file: Path) -> None:
        """Test that domain is inferred from the filename."""
        _, metadata = load_text_file(sample_text_file)
        assert metadata.domain == "investment"

    def test_uses_filename_as_title(self, sample_text_file: Path) -> None:
        """Test that filename (without extension) is used as title."""
        _, metadata = load_text_file(sample_text_file)
        assert metadata.title == "investment_stocks"


class TestLoadDocument:
    """Tests for the dispatch function."""

    def test_loads_markdown(self, sample_markdown_file: Path) -> None:
        """Test dispatch to markdown loader."""
        content, metadata = load_document(sample_markdown_file)
        assert metadata.source_file == "tax_guide.md"

    def test_loads_text(self, sample_text_file: Path) -> None:
        """Test dispatch to text loader."""
        content, metadata = load_document(sample_text_file)
        assert metadata.source_file == "investment_stocks.txt"

    def test_unsupported_extension_raises_error(self, tmp_path: Path) -> None:
        """Test that unsupported file extensions raise ValueError."""
        file_path = tmp_path / "data.csv"
        file_path.write_text("a,b,c", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_document(file_path)


class TestDiscoverDocuments:
    """Tests for document discovery."""

    def test_finds_markdown_files(self, tmp_path: Path) -> None:
        """Test that markdown files are discovered."""
        (tmp_path / "doc1.md").write_text("content", encoding="utf-8")
        (tmp_path / "doc2.md").write_text("content", encoding="utf-8")
        paths = discover_documents(tmp_path)
        assert len(paths) == 2

    def test_finds_text_files(self, tmp_path: Path) -> None:
        """Test that text files are discovered."""
        (tmp_path / "doc.txt").write_text("content", encoding="utf-8")
        paths = discover_documents(tmp_path)
        assert len(paths) == 1

    def test_ignores_unsupported_files(self, tmp_path: Path) -> None:
        """Test that non-md/txt files are ignored."""
        (tmp_path / "doc.md").write_text("content", encoding="utf-8")
        (tmp_path / "data.csv").write_text("a,b", encoding="utf-8")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        paths = discover_documents(tmp_path)
        assert len(paths) == 1

    def test_empty_directory_returns_empty(self, tmp_path: Path) -> None:
        """Test that an empty directory returns an empty list."""
        paths = discover_documents(tmp_path)
        assert paths == []

    def test_returns_sorted_paths(self, tmp_path: Path) -> None:
        """Test that results are sorted by filename."""
        (tmp_path / "b_doc.md").write_text("content", encoding="utf-8")
        (tmp_path / "a_doc.md").write_text("content", encoding="utf-8")
        paths = discover_documents(tmp_path)
        assert paths[0].name == "a_doc.md"
        assert paths[1].name == "b_doc.md"
