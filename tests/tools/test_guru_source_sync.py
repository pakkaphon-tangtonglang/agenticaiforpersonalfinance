"""Tests for syncing allowlisted guru sources into the knowledge folder."""

from pathlib import Path

import httpx
import pytest

from finance_ai.tools.guru_source_fetcher import DownloadedSource, SourceEntry
from finance_ai.tools.guru_source_sync import (
    FETCHED_DIRNAME,
    SourceSkippedError,
    build_fetched_markdown,
    sync_guru_sources,
)

ARTICLE = "\n".join(["Header menu", "1,234"] + ["เนื้อหาบทความที่ยาวพอสมควรสำหรับทดสอบระบบ"] * 40)
ARTICLE_HTML = (
    "<html><body>" + "".join(f"<p>{line}</p>" for line in ARTICLE.split("\n")) + "</body></html>"
)
ROBOTS_OK = httpx.Response(200, text="User-agent: *\nAllow: /\n")


def _client(routes: dict[str, httpx.Response]) -> httpx.Client:
    """httpx client answering from a URL -> Response map (404 otherwise)."""
    return httpx.Client(
        transport=httpx.MockTransport(
            lambda request: routes.get(str(request.url), httpx.Response(404))
        )
    )


class TestBuildFetchedMarkdown:
    """Tests for build_fetched_markdown."""

    def test_builds_front_matter_and_clean_body(self) -> None:
        """Front-matter marks the file as auto-fetched with its source URL."""
        entry = SourceEntry(guru_id="niwes", url="https://www.finnomena.com/dr-niwes/x")

        markdown = build_fetched_markdown(
            entry, DownloadedSource(title="T", text=ARTICLE, is_pdf=False)
        )

        assert "guru_id: niwes" in markdown
        assert "source_url: https://www.finnomena.com/dr-niwes/x" in markdown
        assert "auto_fetched: true" in markdown
        assert "Header menu" not in markdown

    def test_missing_marker_is_skipped(self) -> None:
        """A changed layout (marker not found) is skipped, not indexed whole."""
        entry = SourceEntry(
            guru_id="buffett", url="https://example.com/a.pdf", start_marker="^Gone"
        )

        with pytest.raises(SourceSkippedError, match="marker"):
            build_fetched_markdown(entry, DownloadedSource(title="", text=ARTICLE, is_pdf=True))

    def test_too_short_body_is_skipped(self) -> None:
        """Near-empty extractions (e.g. a redirect page) are skipped."""
        entry = SourceEntry(guru_id="dalio", url="https://example.com/a")

        with pytest.raises(SourceSkippedError, match="too short"):
            build_fetched_markdown(
                entry, DownloadedSource(title="", text="Redirecting...", is_pdf=False)
            )


class TestSyncGuruSources:
    """Tests for sync_guru_sources."""

    def test_writes_into_fetched_folder_and_reports_failures(self, tmp_path: Path) -> None:
        """Good sources are written; failures are reported without stopping."""
        good = SourceEntry(guru_id="niwes", url="https://www.finnomena.com/dr-niwes/good")
        missing = SourceEntry(guru_id="niwes", url="https://www.finnomena.com/dr-niwes/missing")
        client = _client(
            {
                "https://www.finnomena.com/robots.txt": ROBOTS_OK,
                good.url: httpx.Response(
                    200, text=ARTICLE_HTML, headers={"content-type": "text/html"}
                ),
            }
        )
        sleeps: list[float] = []

        results = sync_guru_sources([good, missing], client, tmp_path, sleep=sleeps.append)

        assert [result.ok for result in results] == [True, False]
        written = Path(results[0].path or "")
        assert written.parent == tmp_path / "niwes" / FETCHED_DIRNAME
        assert written.name == "www_finnomena_com_dr_niwes_good.md"
        assert sleeps == [1.0]

    def test_rerun_overwrites_same_file(self, tmp_path: Path) -> None:
        """Deterministic file names mean reruns replace, not duplicate."""
        entry = SourceEntry(guru_id="niwes", url="https://www.finnomena.com/dr-niwes/good")
        client = _client(
            {
                "https://www.finnomena.com/robots.txt": ROBOTS_OK,
                entry.url: httpx.Response(
                    200, text=ARTICLE_HTML, headers={"content-type": "text/html"}
                ),
            }
        )

        sync_guru_sources([entry], client, tmp_path, sleep=lambda _: None)
        sync_guru_sources([entry], client, tmp_path, sleep=lambda _: None)

        assert len(list((tmp_path / "niwes" / FETCHED_DIRNAME).glob("*.md"))) == 1
