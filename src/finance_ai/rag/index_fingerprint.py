"""Fingerprints that tell bootstrap when a vector index is out of date.

The fingerprint hashes every indexed document (path + bytes), the folder's
``sources.yaml`` attribution registry and ``INDEX_SCHEMA_VERSION``. Bootstrap
stores it next to the Chroma index and rebuilds the collection when the
stored value no longer matches, so edited, renamed or re-attributed
documents are re-indexed automatically instead of serving stale chunks.
"""

import hashlib
from pathlib import Path

from finance_ai.rag.document_loader import SOURCE_REGISTRY_FILENAME, discover_documents

# Bump when chunk metadata changes shape (e.g. new title/source fields), so
# indexes built by older code are rebuilt even if the documents did not change.
INDEX_SCHEMA_VERSION = "2"


def _indexed_files(directory: Path, recursive: bool) -> list[Path]:
    """List the documents plus any source registries that feed the index."""
    registries = directory.glob(
        f"**/{SOURCE_REGISTRY_FILENAME}" if recursive else SOURCE_REGISTRY_FILENAME
    )
    files = set(discover_documents(directory, recursive=recursive)) | set(registries)
    return sorted(files, key=lambda path: path.relative_to(directory).as_posix())


def compute_fingerprint(directory: Path, recursive: bool = False) -> str:
    """Hash everything that determines the contents of a collection.

    Args:
        directory: Knowledge folder that is indexed.
        recursive: Include subfolders (the per-guru layout).

    Returns:
        Hex SHA-256 digest.

    Example:
        >>> compute_fingerprint(Path("docs/knowledge_base"))
        '3f1c…'
    """
    digest = hashlib.sha256(f"schema:{INDEX_SCHEMA_VERSION}\n".encode("utf-8"))
    for path in _indexed_files(directory, recursive):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def fingerprint_path(persist_directory: str, collection_name: str) -> Path:
    """Return where a collection's fingerprint is stored.

    Args:
        persist_directory: Chroma persist directory.
        collection_name: Chroma collection name.

    Returns:
        Path of the fingerprint file inside the persist directory.

    Example:
        >>> fingerprint_path("./chroma_db", "finance_knowledge").name
        'finance_knowledge.fingerprint'
    """
    return Path(persist_directory) / f"{collection_name}.fingerprint"


def read_fingerprint(path: Path) -> str | None:
    """Read a stored fingerprint, or None when none was saved yet.

    Args:
        path: Fingerprint file path.

    Returns:
        The stored digest, or None.

    Example:
        >>> read_fingerprint(Path("chroma_db/finance_knowledge.fingerprint"))
    """
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def write_fingerprint(path: Path, fingerprint: str) -> None:
    """Store a fingerprint, creating the folder if needed.

    Args:
        path: Fingerprint file path.
        fingerprint: Digest to store.

    Example:
        >>> write_fingerprint(Path("chroma_db/finance_knowledge.fingerprint"), "3f1c…")
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fingerprint, encoding="utf-8")
