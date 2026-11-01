"""Document loading for RAG knowledge base.

Loads Markdown and text files, extracting metadata from filenames and headings.
"""

import re
from pathlib import Path
from typing import Any, Literal

import yaml

from finance_ai.rag.document_models import DocumentMetadata, SourceKind

DOMAIN_PREFIXES: dict[str, Literal["tax", "expense", "investment"]] = {
    "tax_": "tax",
    "investment_": "investment",
    "expense_": "expense",
}

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

FRONT_MATTER_PATTERN = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def infer_domain_from_filename(
    filename: str,
) -> Literal["tax", "expense", "investment", "general"]:
    """Infer the knowledge domain from a filename prefix (case-insensitive).

    Args:
        filename: The filename to analyze (e.g., "Tax_Guide.md").

    Returns:
        The inferred domain string.

    Example:
        >>> infer_domain_from_filename("tax_deductions_guide.md")
        'tax'
    """
    lowered = filename.lower()
    for prefix, domain in DOMAIN_PREFIXES.items():
        if lowered.startswith(prefix):
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


def parse_front_matter(content: str) -> tuple[dict[str, Any], str]:
    """Split optional YAML front-matter from markdown content.

    Args:
        content: Raw markdown text, possibly starting with a '---' block.

    Returns:
        Tuple of (front_matter dict, body text). The dict is empty and the
        body unchanged when there is no front-matter.

    Raises:
        ValueError: If the front-matter is not a YAML mapping.

    Example:
        >>> parse_front_matter("---\\nguru_id: dalio\\n---\\n# Title\\n")
        ({'guru_id': 'dalio'}, '# Title\\n')
    """
    match = FRONT_MATTER_PATTERN.match(content)
    if match is None:
        return {}, content
    front_matter = yaml.safe_load(match.group(1)) or {}
    if not isinstance(front_matter, dict):
        raise ValueError("Markdown front-matter must be a YAML mapping.")
    return front_matter, content[match.end() :]


def _source_fields(front_matter: dict[str, Any]) -> dict[str, Any]:
    """Pick the public-source attribution fields out of front-matter.

    Args:
        front_matter: Parsed front-matter (may be empty).

    Returns:
        Keyword arguments for DocumentMetadata's source_* fields.

    Example:
        >>> _source_fields({"auto_fetched": True})["source_kind"]
        'original'
    """
    kind: SourceKind = "original" if front_matter.get("auto_fetched") else "compiled"
    return {
        "source_url": front_matter.get("source_url"),
        "source_title": front_matter.get("source_title"),
        "source_publisher": front_matter.get("source_publisher"),
        "source_kind": kind,
    }


def _build_guru_metadata(
    filename: str, front_matter: dict[str, Any], body: str
) -> DocumentMetadata:
    """Build metadata for a guru knowledge document from its front-matter.

    Args:
        filename: Source filename.
        front_matter: Parsed front-matter containing at least guru_id.
        body: Markdown body used as a title fallback.

    Returns:
        DocumentMetadata in the "guru" domain.
    """
    title = front_matter.get("source_title") or _extract_markdown_title(body) or filename
    return DocumentMetadata(
        source_file=filename,
        domain="guru",
        title=str(title),
        guru_id=str(front_matter["guru_id"]),
        language=str(front_matter.get("language", "th")),
        **_source_fields(front_matter),
    )


def load_markdown_file(file_path: Path) -> tuple[str, DocumentMetadata]:
    """Load a markdown file and extract metadata.

    Extracts title from the first # heading. Infers domain from filename prefix.
    Files whose front-matter has a guru_id are tagged with the "guru" domain.

    Args:
        file_path: Path to the .md file.

    Returns:
        Tuple of (raw_text, metadata).

    Example:
        >>> content, meta = load_markdown_file(Path("tax_guide.md"))
    """
    front_matter, content = parse_front_matter(file_path.read_text(encoding="utf-8"))
    filename = file_path.name
    if front_matter.get("guru_id"):
        return content, _build_guru_metadata(filename, front_matter, content)
    title = _extract_markdown_title(content) or file_path.stem
    metadata = DocumentMetadata(
        source_file=filename,
        domain=infer_domain_from_filename(filename),
        title=title,
        **_source_fields(front_matter),
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


SOURCE_REGISTRY_FILENAME = "sources.yaml"
_REGISTRY_FIELDS = ("title", "source_title", "source_url", "source_publisher", "source_kind")


def _read_source_registry(registry_path: Path) -> dict[str, dict[str, Any]]:
    """Read the per-folder source registry, keyed by filename.

    Args:
        registry_path: Path to a sources.yaml file (may not exist).

    Returns:
        Mapping of filename to source fields; empty when the file is absent.

    Raises:
        ValueError: If the registry's 'sources' entry is not a mapping.
    """
    if not registry_path.is_file():
        return {}
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    sources = data.get("sources") or {}
    if not isinstance(sources, dict):
        raise ValueError(f"'sources' in {registry_path} must map filenames to source fields.")
    return sources


def apply_source_registry(metadata: DocumentMetadata, file_path: Path) -> DocumentMetadata:
    """Overlay readable title/source fields from the folder's sources.yaml.

    PDFs cannot carry front-matter, so a sources.yaml next to the documents
    names each file's display title and public source.

    Args:
        metadata: Metadata produced by the per-extension loader.
        file_path: Path of the loaded document.

    Returns:
        Updated metadata, or the original when the file is not listed.

    Raises:
        ValueError: If the registry is malformed.

    Example:
        >>> meta = apply_source_registry(meta, Path("docs/knowledge_base/a.pdf"))
    """
    registry = _read_source_registry(file_path.parent / SOURCE_REGISTRY_FILENAME)
    entry = registry.get(file_path.name)
    if not entry:
        return metadata
    overrides = {field: entry[field] for field in _REGISTRY_FIELDS if entry.get(field)}
    return metadata.model_copy(update=overrides)


def load_document(file_path: Path) -> tuple[str, DocumentMetadata]:
    """Load a document and attach its source attribution.

    Args:
        file_path: Path to the document file.

    Returns:
        Tuple of (raw_text, metadata).

    Raises:
        ValueError: If the file extension is not supported.

    Example:
        >>> content, meta = load_document(Path("tax_guide.md"))
    """
    content, metadata = _load_by_extension(file_path)
    return content, apply_source_registry(metadata, file_path)


def _load_by_extension(file_path: Path) -> tuple[str, DocumentMetadata]:
    """Dispatch to the loader for the file's extension.

    Args:
        file_path: Path to the document file.

    Returns:
        Tuple of (raw_text, metadata).

    Raises:
        ValueError: If the file extension is not supported.
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


def discover_documents(directory: Path, recursive: bool = False) -> list[Path]:
    """Find all supported documents in a directory.

    Args:
        directory: Directory to search for documents.
        recursive: Also search subfolders (used by the per-guru layout).

    Returns:
        Sorted list of paths to supported document files.

    Example:
        >>> paths = discover_documents(Path("docs/knowledge_base"))
    """
    paths: list[Path] = []
    pattern_prefix = "**/*" if recursive else "*"
    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(directory.glob(f"{pattern_prefix}{ext}"))
    return sorted(paths, key=lambda p: p.name)
