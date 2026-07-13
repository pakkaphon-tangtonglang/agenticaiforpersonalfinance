"""Document loading for RAG knowledge base.

Loads Markdown and text files, extracting metadata from filenames and headings.
"""

import re
from pathlib import Path
from typing import Any, Literal

from finance_ai.rag.document_models import DocumentMetadata

DOMAIN_PREFIXES: dict[str, Literal["tax", "expense", "investment"]] = {
    "tax_": "tax",
    "investment_": "investment",
    "expense_": "expense",
}

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def infer_domain_from_filename(
    filename: str,
) -> Literal["tax", "expense", "investment", "general"]:
    """Infer the knowledge domain from a filename prefix.

    Args:
        filename: The filename to analyze (e.g., "tax_guide.md").

    Returns:
        The inferred domain string.

    Example:
        >>> infer_domain_from_filename("tax_deductions_guide.md")
        'tax'
    """
    for prefix, domain in DOMAIN_PREFIXES.items():
        if filename.startswith(prefix):
            return domain
    return "general"


def _extract_markdown_title(content: str) -> str | None:
    """Extract the first top-level heading from markdown content.

    Args:
        content: Raw markdown text.

    Returns:
        The heading text, or None if no heading found.
    """
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def load_markdown_file(file_path: Path) -> tuple[str, DocumentMetadata]:
    """Load a markdown file and extract metadata.

    Extracts title from the first # heading. Infers domain from filename prefix.

    Args:
        file_path: Path to the .md file.

    Returns:
        Tuple of (raw_text, metadata).

    Example:
        >>> content, meta = load_markdown_file(Path("tax_guide.md"))
    """
    content = file_path.read_text(encoding="utf-8")
    filename = file_path.name
    title = _extract_markdown_title(content) or file_path.stem
    metadata = DocumentMetadata(
        source_file=filename,
        domain=infer_domain_from_filename(filename),
        title=title,
    )
    return content, metadata


def load_text_file(file_path: Path) -> tuple[str, DocumentMetadata]:
    """Load a text file and infer metadata from filename.

    Args:
        file_path: Path to the .txt file.

    Returns:
        Tuple of (raw_text, metadata).

    Example:
        >>> content, meta = load_text_file(Path("investment_stocks.txt"))
    """
    content = file_path.read_text(encoding="utf-8")
    filename = file_path.name
    metadata = DocumentMetadata(
        source_file=filename,
        domain=infer_domain_from_filename(filename),
        title=file_path.stem,
    )
    return content, metadata


def _import_pdf_reader() -> Any:
    """Lazy-import PdfReader from pypdf.

    Returns:
        The PdfReader class.

    Raises:
        ImportError: If pypdf is not installed.
    """
    try:
        from pypdf import PdfReader  # noqa: PLC0415

        return PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF loading. Install it with: pip install pypdf"
        ) from exc


def load_pdf_file(file_path: Path) -> tuple[str, DocumentMetadata]:
    """Load a PDF file and extract text from all pages.

    Args:
        file_path: Path to the .pdf file.

    Returns:
        Tuple of (raw_text, metadata).

    Example:
        >>> content, meta = load_pdf_file(Path("investment_guide.pdf"))
    """
    pdf_reader_class = _import_pdf_reader()
    reader = pdf_reader_class(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    content = "\n\n".join(pages).strip()
    filename = file_path.name
    metadata = DocumentMetadata(
        source_file=filename,
        domain=infer_domain_from_filename(filename),
        title=file_path.stem,
    )
    return content, metadata


def load_document(file_path: Path) -> tuple[str, DocumentMetadata]:
    """Load a document, dispatching to the appropriate loader by extension.

    Args:
        file_path: Path to the document file.

    Returns:
        Tuple of (raw_text, metadata).

    Raises:
        ValueError: If the file extension is not supported.

    Example:
        >>> content, meta = load_document(Path("tax_guide.md"))
    """
    extension = file_path.suffix.lower()
    if extension == ".md":
        return load_markdown_file(file_path)
    if extension == ".txt":
        return load_text_file(file_path)
    if extension == ".pdf":
        return load_pdf_file(file_path)
    raise ValueError(
        f"Unsupported file extension: '{extension}'. "
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def discover_documents(directory: Path) -> list[Path]:
    """Find all supported documents in a directory.

    Args:
        directory: Directory to search for documents.

    Returns:
        Sorted list of paths to supported document files.

    Example:
        >>> paths = discover_documents(Path("docs/knowledge_base"))
    """
    paths: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(directory.glob(f"*{ext}"))
    return sorted(paths, key=lambda p: p.name)
