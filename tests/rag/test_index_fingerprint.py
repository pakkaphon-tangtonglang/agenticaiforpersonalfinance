"""Tests for knowledge-base fingerprints (detecting a stale vector index)."""

from pathlib import Path

import pytest

from finance_ai.rag import index_fingerprint
from finance_ai.rag.index_fingerprint import (
    compute_fingerprint,
    fingerprint_path,
    read_fingerprint,
    write_fingerprint,
)


@pytest.fixture
def knowledge_dir(tmp_path: Path) -> Path:
    """A small knowledge folder with one doc and a source registry."""
    folder = tmp_path / "kb"
    folder.mkdir()
    (folder / "Tax_Guide.md").write_text("# ภาษี\nเนื้อหา", encoding="utf-8")
    (folder / "sources.yaml").write_text("sources: {}\n", encoding="utf-8")
    return folder


class TestComputeFingerprint:
    """The fingerprint changes exactly when indexed content changes."""

    def test_same_content_same_fingerprint(self, knowledge_dir: Path) -> None:
        """Re-computing without changes is stable."""
        assert compute_fingerprint(knowledge_dir) == compute_fingerprint(knowledge_dir)

    def test_edited_document_changes_fingerprint(self, knowledge_dir: Path) -> None:
        """Editing a document's text marks the index stale."""
        before = compute_fingerprint(knowledge_dir)
        (knowledge_dir / "Tax_Guide.md").write_text("# ภาษี\nแก้ไขแล้ว", encoding="utf-8")

        assert compute_fingerprint(knowledge_dir) != before

    def test_renamed_document_changes_fingerprint(self, knowledge_dir: Path) -> None:
        """A rename changes source_file metadata, so it counts as a change."""
        before = compute_fingerprint(knowledge_dir)
        (knowledge_dir / "Tax_Guide.md").rename(knowledge_dir / "Tax_Deductions.md")

        assert compute_fingerprint(knowledge_dir) != before

    def test_source_registry_edit_changes_fingerprint(self, knowledge_dir: Path) -> None:
        """Attribution lives in sources.yaml, so editing it marks the index stale."""
        before = compute_fingerprint(knowledge_dir)
        (knowledge_dir / "sources.yaml").write_text(
            "sources:\n  Tax_Guide.md:\n    source_publisher: กรมสรรพากร\n", encoding="utf-8"
        )

        assert compute_fingerprint(knowledge_dir) != before

    def test_schema_version_changes_fingerprint(
        self, knowledge_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bumping the metadata schema version forces a rebuild."""
        before = compute_fingerprint(knowledge_dir)
        monkeypatch.setattr(index_fingerprint, "INDEX_SCHEMA_VERSION", "999")

        assert compute_fingerprint(knowledge_dir) != before

    def test_recursive_includes_subfolders(self, knowledge_dir: Path) -> None:
        """Guru knowledge lives in subfolders; recursive mode must see them."""
        flat = compute_fingerprint(knowledge_dir, recursive=True)
        (knowledge_dir / "buffett").mkdir()
        (knowledge_dir / "buffett" / "letter.md").write_text("# L\n", encoding="utf-8")

        assert compute_fingerprint(knowledge_dir, recursive=True) != flat
        assert compute_fingerprint(knowledge_dir) == compute_fingerprint(knowledge_dir)


class TestFingerprintStorage:
    """The fingerprint is stored as a small file next to the vector index."""

    def test_path_is_per_collection(self, tmp_path: Path) -> None:
        """Each collection gets its own fingerprint file."""
        assert fingerprint_path(str(tmp_path), "finance") != fingerprint_path(str(tmp_path), "guru")
        assert fingerprint_path(str(tmp_path), "guru").parent == tmp_path

    def test_round_trip(self, tmp_path: Path) -> None:
        """A written fingerprint reads back unchanged, creating folders as needed."""
        path = tmp_path / "chroma" / "finance.fingerprint"

        write_fingerprint(path, "abc123")

        assert read_fingerprint(path) == "abc123"

    def test_missing_file_reads_none(self, tmp_path: Path) -> None:
        """No stored fingerprint means 'unknown', which callers treat as stale."""
        assert read_fingerprint(tmp_path / "absent.fingerprint") is None
