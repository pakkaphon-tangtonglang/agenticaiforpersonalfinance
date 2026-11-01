"""Runtime bootstrap for hosted deployments.

Prepares the environment before the API starts:
1. Applies database migrations (``alembic upgrade head``).
2. Indexes the RAG knowledge base when it is empty or its files changed.
3. Does the same for the guru knowledge collection.

Idempotent by design: safe to run on every boot, whether the disk is
fresh (Render free tier wipes it) or already populated.

Example:
    >>> run_bootstrap()  # migrations + RAG index
"""

import threading
from pathlib import Path

import httpx
from alembic import command
from alembic.config import Config

from finance_ai.core.config import get_settings
from finance_ai.core.logging import get_logger
from finance_ai.rag.index_fingerprint import (
    compute_fingerprint,
    fingerprint_path,
    read_fingerprint,
    write_fingerprint,
)
from finance_ai.rag.knowledge_base import KnowledgeBaseManager, create_knowledge_base_manager
from finance_ai.tools.guru_source_fetcher import USER_AGENT, load_source_entries
from finance_ai.tools.guru_source_sync import sync_guru_sources

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ALEMBIC_CONFIG = _PROJECT_ROOT / "alembic.ini"
_GURU_SOURCES_FILE = _PROJECT_ROOT / "docs" / "gurus" / "sources.yaml"


def run_migrations(alembic_config_path: Path = _DEFAULT_ALEMBIC_CONFIG) -> None:
    """Apply all pending database migrations up to head.

    Args:
        alembic_config_path: Path to the alembic configuration file.

    Raises:
        alembic.util.CommandError: When migrations fail to apply.

    Example:
        >>> run_migrations()
    """
    config = Config(str(alembic_config_path))
    command.upgrade(config, "head")
    logger.info("Database migrations applied (head)")


def _sync_collection(
    manager: KnowledgeBaseManager, directory: Path, recursive: bool, fingerprint_file: Path
) -> int:
    """Index an empty collection, rebuild a stale one, skip a current one.

    Args:
        manager: Manager bound to the target collection.
        directory: Knowledge folder that feeds the collection.
        recursive: Read subfolders (the per-guru layout).
        fingerprint_file: Where the collection's content fingerprint is stored.

    Returns:
        Number of chunks (re)indexed; 0 when the collection was already current.
    """
    current = compute_fingerprint(directory, recursive=recursive)
    populated = manager.get_document_count() > 0
    if populated and read_fingerprint(fingerprint_file) == current:
        logger.info("Collection %s is current, skipping", fingerprint_file.stem)
        return 0
    if populated:
        logger.info("Knowledge files changed, rebuilding %s", fingerprint_file.stem)
        count = manager.rebuild_index(directory, recursive=recursive)
    else:
        count = manager.index_directory(directory, recursive=recursive)
    write_fingerprint(fingerprint_file, current)
    return count


def index_knowledge_base_if_stale() -> int:
    """Index the knowledge base when it is empty or its files changed.

    Returns:
        Number of chunks indexed (0 when the index is already current).

    Example:
        >>> count = index_knowledge_base_if_stale()
    """
    settings = get_settings()
    manager = create_knowledge_base_manager(settings)
    directory = Path(settings.rag_knowledge_base_directory)
    fingerprint_file = fingerprint_path(
        settings.rag_chroma_persist_directory, settings.rag_chroma_collection_name
    )
    count = _sync_collection(manager, directory, False, fingerprint_file)
    logger.info("Knowledge base index: %d chunks (re)indexed", count)
    return count


def index_guru_knowledge_if_stale() -> int:
    """Index the guru collection (one subfolder per guru) when empty or changed.

    Returns:
        Number of chunks indexed (0 when the collection is already current).

    Example:
        >>> count = index_guru_knowledge_if_stale()
    """
    settings = get_settings()
    collection = settings.rag_guru_collection_name
    manager = create_knowledge_base_manager(settings, collection_name=collection)
    directory = Path(settings.rag_guru_knowledge_directory)
    fingerprint_file = fingerprint_path(settings.rag_chroma_persist_directory, collection)
    count = _sync_collection(manager, directory, True, fingerprint_file)
    logger.info("Guru knowledge index: %d chunks (re)indexed", count)
    return count


def rebuild_guru_knowledge() -> int:
    """Clear and re-index the guru collection so edited docs are picked up.

    Returns:
        Number of chunks indexed.

    Example:
        >>> count = rebuild_guru_knowledge()  # make guru-index
    """
    settings = get_settings()
    manager = create_knowledge_base_manager(
        settings, collection_name=settings.rag_guru_collection_name
    )
    directory = Path(settings.rag_guru_knowledge_directory)
    count = manager.rebuild_index(directory, recursive=True)
    logger.info("Rebuilt guru knowledge: %d chunks", count)
    return count


def run_bootstrap() -> None:
    """Run the full bootstrap sequence: migrations, then RAG indexing.

    Migrations are required (the API cannot run without a schema), but a
    RAG indexing failure only logs a warning — a hosted deployment must
    still boot and serve traffic, even without a knowledge base index
    (e.g. missing embedding API key on Render).

    Example:
        >>> run_bootstrap()
    """
    logger.info("Bootstrap starting")
    run_migrations()
    _index_knowledge_base_safely()
    logger.info("Bootstrap complete")


def _index_knowledge_base_safely() -> None:
    """Run RAG indexing, converting any failure into a logged warning.

    Example:
        >>> _index_knowledge_base_safely()  # never raises
    """
    try:
        index_knowledge_base_if_stale()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Broad by design: API startup must never be blocked by RAG.
        logger.error("RAG indexing failed (agents run without knowledge base): %s", exc)
    try:
        index_guru_knowledge_if_stale()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.error("Guru indexing failed (gurus answer from profiles only): %s", exc)


def refresh_fetched_guru_knowledge() -> int:
    """Fetch allowlisted guru sources, write them to fetched/ and index them.

    Runs at deploy time (see start_guru_refresh_in_background) so the public
    repo keeps only links. Chunk ids are deterministic, so re-indexing the
    same file upserts instead of duplicating.

    Returns:
        Number of chunks indexed.

    Example:
        >>> refresh_fetched_guru_knowledge()
        412
    """
    settings = get_settings()
    entries = load_source_entries(_GURU_SOURCES_FILE)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        results = sync_guru_sources(entries, client, Path(settings.rag_guru_knowledge_directory))
    manager = create_knowledge_base_manager(
        settings, collection_name=settings.rag_guru_collection_name
    )
    paths = [Path(result.path) for result in results if result.ok and result.path]
    count = sum(manager.index_document(path) for path in paths)
    logger.info("Guru refresh: %d/%d sources, %d chunks", len(paths), len(results), count)
    return count


def _refresh_fetched_guru_knowledge_safely() -> None:
    """Run the guru refresh, converting any failure into a logged error."""
    try:
        refresh_fetched_guru_knowledge()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Broad by design: a background refresh must never affect the API.
        logger.error("Guru source refresh failed (curated docs still indexed): %s", exc)


def start_guru_refresh_in_background() -> threading.Thread:
    """Start the guru source refresh in a daemon thread.

    The API serves immediately; gurus use the curated docs until the fetched
    sources are indexed (usually within a minute).

    Returns:
        The started thread.

    Example:
        >>> start_guru_refresh_in_background()
    """
    thread = threading.Thread(
        target=_refresh_fetched_guru_knowledge_safely, name="guru-refresh", daemon=True
    )
    thread.start()
    return thread
