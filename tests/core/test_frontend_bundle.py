"""Tests for the frontend deploy bundle builder.

Bundles the static frontend for iHost upload: copies the static files and
generates a config.js pointing at the deployed backend. The API base must
be HTTPS because the iHost page is served over HTTPS (mixed-content rule).
"""

from pathlib import Path

import pytest

from finance_ai.core.frontend_bundle import bundle_frontend


@pytest.fixture(name="static_dir", scope="module")
def static_dir_fixture() -> Path:
    """Point at the real static assets shipped with the app."""
    return Path(__file__).resolve().parents[2] / "src" / "finance_ai" / "static"


class TestBundleFrontend:
    """Tests for building the iHost upload folder."""

    def test_bundles_all_static_files(self, static_dir: Path, tmp_path: Path) -> None:
        """index.html, styles.css, and app.js are copied into the bundle."""
        output = tmp_path / "bundle"

        copied = bundle_frontend(
            api_base_url="https://app.onrender.com",
            source_dir=static_dir,
            output_dir=output,
        )

        copied_names = {path.name for path in copied}
        assert copied_names == {"index.html", "styles.css", "app.js", "config.js"}
        for name in copied_names:
            assert (output / name).exists()

    def test_generated_config_js_points_at_backend(self, static_dir: Path, tmp_path: Path) -> None:
        """The generated config.js sets FINANCE_API_BASE to the given URL."""
        output = tmp_path / "bundle"

        bundle_frontend(
            api_base_url="https://app.onrender.com",
            source_dir=static_dir,
            output_dir=output,
        )

        config_text = (output / "config.js").read_text(encoding="utf-8")
        assert 'window.FINANCE_API_BASE = "https://app.onrender.com"' in config_text

    def test_http_api_base_rejected(self, static_dir: Path, tmp_path: Path) -> None:
        """Non-HTTPS API bases are rejected (mixed-content rule)."""
        with pytest.raises(ValueError, match="HTTPS"):
            bundle_frontend(
                api_base_url="http://app.onrender.com",
                source_dir=static_dir,
                output_dir=tmp_path / "bundle",
            )

    def test_empty_api_base_rejected(self, static_dir: Path, tmp_path: Path) -> None:
        """An empty API base is rejected with a clear message."""
        with pytest.raises(ValueError, match="api_base_url"):
            bundle_frontend(
                api_base_url="",
                source_dir=static_dir,
                output_dir=tmp_path / "bundle",
            )

    def test_missing_source_file_rejected(self, tmp_path: Path) -> None:
        """A source directory missing required files fails loudly."""
        empty_source = tmp_path / "empty_static"
        empty_source.mkdir()

        with pytest.raises(FileNotFoundError):
            bundle_frontend(
                api_base_url="https://app.onrender.com",
                source_dir=empty_source,
                output_dir=tmp_path / "bundle",
            )
