"""Runtime bootstrap for hosted deployments.

Prepares the environment before the API starts:
1. Applies database migrations (``alembic upgrade head``).
2. Indexes the RAG knowledge base when the vector store is empty.

Idempotent by design: safe to run on every boot, whether the disk is
fresh (Render free tier wipes it) or already populated.

Example:
    >>> run_bootstrap()  # migrations + RAG index
"""

from pathlib import Path

from alembic import command
from alembic.config import Config

from finance_ai.core.config import get_settings
from finance_ai.core.logging import get_logger
from finance_ai.rag.knowledge_base import create_knowledge_base_manager

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ALEMBIC_CONFIG = _PROJECT_ROOT / "alembic.ini"


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


def index_knowledge_base_if_empty() -> int:
    """Index the knowledge base when the vector store is empty.

    Returns:
        Number of chunks indexed (0 when the store is already populated).

    Example:
        >>> count = index_knowledge_base_if_empty()
    """
    settings = get_settings()
    manager = create_knowledge_base_manager(settings)
    if manager.get_document_count() > 0:
        logger.info("Vector store already indexed, skipping")
        return 0
    directory = Path(settings.rag_knowledge_base_directory)
    count = manager.index_directory(directory)
    logger.info("Indexed knowledge base: %d chunks", count)
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
        index_knowledge_base_if_empty()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Broad by design: API startup must never be blocked by RAG.
        logger.error("RAG indexing failed (agents run without knowledge base): %s", exc)
