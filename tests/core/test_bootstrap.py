"""Tests for the runtime bootstrap module.

Bootstrap prepares a hosted deployment: database migrations + RAG indexing.
It MUST be idempotent so it can run on every boot (Render release command)
against a fresh ephemeral disk or a pre-populated one.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from finance_ai.core.bootstrap import (
    _DEFAULT_ALEMBIC_CONFIG,
    index_knowledge_base_if_empty,
    run_bootstrap,
    run_migrations,
)


class TestRunMigrations:
    """Tests for applying database migrations."""

    def test_default_config_points_to_real_alembic_ini(self) -> None:
        """The default alembic.ini path exists at the project root."""
        assert _DEFAULT_ALEMBIC_CONFIG.name == "alembic.ini"
        assert _DEFAULT_ALEMBIC_CONFIG.exists()

    @patch("finance_ai.core.bootstrap.command.upgrade")
    @patch("finance_ai.core.bootstrap.Config")
    def test_upgrades_to_head(self, mock_config: MagicMock, mock_upgrade: MagicMock) -> None:
        """Migrations are applied up to head via alembic."""
        run_migrations(Path("alembic.ini"))

        mock_config.assert_called_once_with(str(Path("alembic.ini")))
        mock_upgrade.assert_called_once_with(mock_config.return_value, "head")


class TestIndexKnowledgeBaseIfEmpty:
    """Tests for idempotent knowledge base indexing."""

    @patch("finance_ai.core.bootstrap.create_knowledge_base_manager")
    def test_indexes_when_store_empty(self, mock_create_manager: MagicMock) -> None:
        """An empty vector store gets indexed, returning the chunk count."""
        mock_manager = MagicMock()
        mock_manager.get_document_count.return_value = 0
        mock_manager.index_directory.return_value = 42
        mock_create_manager.return_value = mock_manager

        with patch("finance_ai.core.bootstrap.get_settings") as mock_settings:
            mock_settings.return_value.rag_knowledge_base_directory = "docs/knowledge_base"
            count = index_knowledge_base_if_empty()

        assert count == 42
        mock_manager.index_directory.assert_called_once_with(Path("docs/knowledge_base"))

    @patch("finance_ai.core.bootstrap.create_knowledge_base_manager")
    def test_skips_when_store_populated(self, mock_create_manager: MagicMock) -> None:
        """A populated vector store is left untouched (no re-embedding)."""
        mock_manager = MagicMock()
        mock_manager.get_document_count.return_value = 100
        mock_create_manager.return_value = mock_manager

        with patch("finance_ai.core.bootstrap.get_settings") as mock_settings:
            mock_settings.return_value.rag_knowledge_base_directory = "docs/knowledge_base"
            count = index_knowledge_base_if_empty()

        assert count == 0
        mock_manager.index_directory.assert_not_called()


class TestRunBootstrap:
    """Tests for the full bootstrap sequence."""

    @patch("finance_ai.core.bootstrap.index_knowledge_base_if_empty")
    @patch("finance_ai.core.bootstrap.run_migrations")
    def test_runs_migrations_then_indexing(
        self, mock_migrations: MagicMock, mock_indexing: MagicMock
    ) -> None:
        """Bootstrap applies migrations first, then indexes the knowledge base."""
        run_bootstrap()

        mock_migrations.assert_called_once()
        mock_indexing.assert_called_once()

    @patch("finance_ai.core.bootstrap.index_knowledge_base_if_empty")
    @patch("finance_ai.core.bootstrap.run_migrations")
    def test_survives_rag_indexing_failure(
        self, mock_migrations: MagicMock, mock_indexing: MagicMock
    ) -> None:
        """A RAG indexing error never blocks startup (hosted API must still boot).

        Example: on Render, embeddings default to Google but the API key
        may be absent — the service must start anyway and serve traffic
        without RAG.
        """
        mock_indexing.side_effect = RuntimeError("No Google API key configured")

        run_bootstrap()

        mock_migrations.assert_called_once()
