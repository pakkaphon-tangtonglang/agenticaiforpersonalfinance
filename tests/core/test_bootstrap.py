"""Tests for the runtime bootstrap module.

Bootstrap prepares a hosted deployment: database migrations + RAG indexing.
It MUST be idempotent so it can run on every boot (Render release command)
against a fresh ephemeral disk or a pre-populated one.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from finance_ai.core.bootstrap import (
    _DEFAULT_ALEMBIC_CONFIG,
    index_guru_knowledge_if_stale,
    index_knowledge_base_if_stale,
    rebuild_guru_knowledge,
    refresh_fetched_guru_knowledge,
    run_bootstrap,
    run_migrations,
    start_guru_refresh_in_background,
)
from finance_ai.tools.guru_source_sync import SyncResult


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


def _manager(document_count: int, indexed: int = 42) -> MagicMock:
    """A knowledge-base manager mock with a given current chunk count."""
    manager = MagicMock()
    manager.get_document_count.return_value = document_count
    manager.index_directory.return_value = indexed
    manager.rebuild_index.return_value = indexed
    return manager


def _kb_settings(mock_settings: MagicMock, tmp_path: Path) -> None:
    """Point the settings mock at a real temp knowledge folder and index dir."""
    folder = tmp_path / "kb"
    folder.mkdir(exist_ok=True)
    (folder / "Tax_Guide.md").write_text("# ภาษี\n", encoding="utf-8")
    mock_settings.return_value.rag_knowledge_base_directory = str(folder)
    mock_settings.return_value.rag_chroma_persist_directory = str(tmp_path / "chroma")
    mock_settings.return_value.rag_chroma_collection_name = "finance_knowledge"


class TestIndexKnowledgeBaseIfStale:
    """Bootstrap indexes when empty, skips when current, rebuilds when changed."""

    @patch("finance_ai.core.bootstrap.get_settings")
    @patch("finance_ai.core.bootstrap.create_knowledge_base_manager")
    def test_indexes_when_store_empty(
        self, mock_create_manager: MagicMock, mock_settings: MagicMock, tmp_path: Path
    ) -> None:
        """An empty vector store gets indexed, and the fingerprint is saved."""
        _kb_settings(mock_settings, tmp_path)
        mock_create_manager.return_value = _manager(document_count=0)

        count = index_knowledge_base_if_stale()

        assert count == 42
        mock_create_manager.return_value.index_directory.assert_called_once()
        assert (tmp_path / "chroma" / "finance_knowledge.fingerprint").is_file()

    @patch("finance_ai.core.bootstrap.get_settings")
    @patch("finance_ai.core.bootstrap.create_knowledge_base_manager")
    def test_skips_when_store_current(
        self, mock_create_manager: MagicMock, mock_settings: MagicMock, tmp_path: Path
    ) -> None:
        """A populated store whose fingerprint matches is left untouched."""
        _kb_settings(mock_settings, tmp_path)
        mock_create_manager.return_value = _manager(document_count=0)
        index_knowledge_base_if_stale()
        mock_create_manager.return_value = _manager(document_count=100)

        count = index_knowledge_base_if_stale()

        assert count == 0
        mock_create_manager.return_value.index_directory.assert_not_called()
        mock_create_manager.return_value.rebuild_index.assert_not_called()

    @patch("finance_ai.core.bootstrap.get_settings")
    @patch("finance_ai.core.bootstrap.create_knowledge_base_manager")
    def test_rebuilds_when_documents_changed(
        self, mock_create_manager: MagicMock, mock_settings: MagicMock, tmp_path: Path
    ) -> None:
        """Editing a document after indexing triggers a full rebuild."""
        _kb_settings(mock_settings, tmp_path)
        mock_create_manager.return_value = _manager(document_count=0)
        index_knowledge_base_if_stale()
        (tmp_path / "kb" / "Tax_Guide.md").write_text("# ภาษี\nแก้ไข\n", encoding="utf-8")
        mock_create_manager.return_value = _manager(document_count=100, indexed=7)

        count = index_knowledge_base_if_stale()

        assert count == 7
        mock_create_manager.return_value.rebuild_index.assert_called_once()

    @patch("finance_ai.core.bootstrap.get_settings")
    @patch("finance_ai.core.bootstrap.create_knowledge_base_manager")
    def test_rebuilds_populated_store_without_fingerprint(
        self, mock_create_manager: MagicMock, mock_settings: MagicMock, tmp_path: Path
    ) -> None:
        """An index built before fingerprints existed is treated as stale."""
        _kb_settings(mock_settings, tmp_path)
        mock_create_manager.return_value = _manager(document_count=100)

        index_knowledge_base_if_stale()

        mock_create_manager.return_value.rebuild_index.assert_called_once()


class TestRunBootstrap:
    """Tests for the full bootstrap sequence."""

    @patch("finance_ai.core.bootstrap.index_guru_knowledge_if_stale")
    @patch("finance_ai.core.bootstrap.index_knowledge_base_if_stale")
    @patch("finance_ai.core.bootstrap.run_migrations")
    def test_runs_migrations_then_indexing(
        self,
        mock_migrations: MagicMock,
        mock_indexing: MagicMock,
        mock_guru_indexing: MagicMock,
    ) -> None:
        """Bootstrap applies migrations first, then indexes the knowledge base."""
        run_bootstrap()

        mock_migrations.assert_called_once()
        mock_indexing.assert_called_once()
        mock_guru_indexing.assert_called_once()

    @patch("finance_ai.core.bootstrap.index_guru_knowledge_if_stale")
    @patch("finance_ai.core.bootstrap.index_knowledge_base_if_stale")
    @patch("finance_ai.core.bootstrap.run_migrations")
    def test_survives_rag_indexing_failure(
        self,
        mock_migrations: MagicMock,
        mock_indexing: MagicMock,
        mock_guru_indexing: MagicMock,
    ) -> None:
        """A RAG indexing error never blocks startup (hosted API must still boot).

        Example: on Render, embeddings default to Google but the API key
        may be absent — the service must start anyway and serve traffic
        without RAG.
        """
        mock_indexing.side_effect = RuntimeError("No Google API key configured")

        run_bootstrap()

        mock_migrations.assert_called_once()
        mock_guru_indexing.assert_called_once()


class TestIndexGuruKnowledgeIfStale:
    """Tests for indexing the separate guru collection."""

    @patch("finance_ai.core.bootstrap.get_settings")
    @patch("finance_ai.core.bootstrap.create_knowledge_base_manager")
    def test_indexes_guru_folders_into_guru_collection(
        self, mock_create_manager: MagicMock, mock_settings: MagicMock, tmp_path: Path
    ) -> None:
        """Guru docs go to their own collection, read recursively."""
        (tmp_path / "gurus" / "buffett").mkdir(parents=True)
        (tmp_path / "gurus" / "buffett" / "a.md").write_text("# A\n", encoding="utf-8")
        mock_settings.return_value.rag_guru_collection_name = "guru_knowledge"
        mock_settings.return_value.rag_guru_knowledge_directory = str(tmp_path / "gurus")
        mock_settings.return_value.rag_chroma_persist_directory = str(tmp_path / "chroma")
        mock_create_manager.return_value = _manager(document_count=0, indexed=7)

        count = index_guru_knowledge_if_stale()

        assert count == 7
        assert mock_create_manager.call_args.kwargs["collection_name"] == "guru_knowledge"
        mock_create_manager.return_value.index_directory.assert_called_once_with(
            tmp_path / "gurus", recursive=True
        )
        assert (tmp_path / "chroma" / "guru_knowledge.fingerprint").is_file()


class TestRebuildGuruKnowledge:
    """Tests for the explicit rebuild used by make guru-index."""

    @patch("finance_ai.core.bootstrap.create_knowledge_base_manager")
    def test_rebuild_clears_and_reindexes(self, mock_create_manager: MagicMock) -> None:
        """Rebuild always re-embeds so new or edited guru docs are picked up."""
        mock_create_manager.return_value.rebuild_index.return_value = 12

        with patch("finance_ai.core.bootstrap.get_settings") as mock_settings:
            mock_settings.return_value.rag_guru_knowledge_directory = "docs/guru_knowledge"
            count = rebuild_guru_knowledge()

        assert count == 12
        mock_create_manager.return_value.rebuild_index.assert_called_once_with(
            Path("docs/guru_knowledge"), recursive=True
        )


class TestRefreshFetchedGuruKnowledge:
    """Tests for the deploy-time fetch -> clean -> index refresh."""

    @patch("finance_ai.core.bootstrap.create_knowledge_base_manager")
    @patch("finance_ai.core.bootstrap.sync_guru_sources")
    @patch("finance_ai.core.bootstrap.load_source_entries")
    def test_indexes_only_successfully_synced_files(
        self, mock_load: MagicMock, mock_sync: MagicMock, mock_create_manager: MagicMock
    ) -> None:
        """Skipped or failed sources are not indexed."""
        mock_load.return_value = ["entry"]
        mock_sync.return_value = [
            SyncResult(
                url="a", ok=True, detail="saved", path="docs/guru_knowledge/niwes/fetched/a.md"
            ),
            SyncResult(url="b", ok=False, detail="404"),
        ]
        mock_create_manager.return_value.index_document.return_value = 5

        count = refresh_fetched_guru_knowledge()

        assert count == 5
        mock_create_manager.return_value.index_document.assert_called_once_with(
            Path("docs/guru_knowledge/niwes/fetched/a.md")
        )

    @patch("finance_ai.core.bootstrap.refresh_fetched_guru_knowledge")
    def test_background_refresh_never_raises(self, mock_refresh: MagicMock) -> None:
        """A failed refresh is logged; the running API is unaffected."""
        mock_refresh.side_effect = RuntimeError("network down")

        thread = start_guru_refresh_in_background()
        thread.join(timeout=5)

        assert not thread.is_alive()
        mock_refresh.assert_called_once()
