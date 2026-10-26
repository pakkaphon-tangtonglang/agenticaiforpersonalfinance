"""Tests for the allowlist-only guru source fetcher (no real network)."""

from pathlib import Path

import httpx
import pytest

from finance_ai.tools.guru_source_fetcher import (
    SourceBlockedError,
    SourceEntry,
    fetch_all_sources,
    fetch_source,
    html_to_text,
    load_source_entries,
)

ARTICLE_HTML = """
<html><head><title>Value Investing 101</title><style>.x{}</style></head>
<body><nav>menu</nav><h1>Value Investing</h1><p>Buy good businesses.</p>
<script>track()</script><footer>copyright</footer></body></html>
"""


def _client(routes: dict[str, httpx.Response]) -> httpx.Client:
    """An httpx client whose responses come from a URL -> Response map."""

    def handler(request: httpx.Request) -> httpx.Response:
        return routes.get(str(request.url), httpx.Response(404))

    return httpx.Client(transport=httpx.MockTransport(handler))


ALLOW_ALL = httpx.Response(200, text="User-agent: *\nAllow: /\n")
ENTRY = SourceEntry(guru_id="niwes", url="https://example.com/article", kind="article")


class TestHtmlToText:
    """Tests for html_to_text."""

    def test_extracts_title_and_body_without_boilerplate(self) -> None:
        """Scripts, styles, nav and footer are dropped."""
        title, text = html_to_text(ARTICLE_HTML)

        assert title == "Value Investing 101"
        assert "Buy good businesses." in text
        assert "track()" not in text
        assert "menu" not in text
        assert "copyright" not in text


class TestLoadSourceEntries:
    """Tests for load_source_entries."""

    def test_loads_yaml_list(self, tmp_path: Path) -> None:
        """Entries are validated from the sources file."""
        sources_file = tmp_path / "sources.yaml"
        sources_file.write_text(
            "sources:\n  - guru_id: dalio\n    url: https://example.com/a\n    kind: article\n",
            encoding="utf-8",
        )

        entries = load_source_entries(sources_file)

        assert entries == [
            SourceEntry(guru_id="dalio", url="https://example.com/a", kind="article")
        ]

    def test_rejects_non_http_urls(self, tmp_path: Path) -> None:
        """Only http(s) URLs may be fetched."""
        sources_file = tmp_path / "sources.yaml"
        sources_file.write_text(
            "sources:\n  - guru_id: dalio\n    url: file:///etc/passwd\n    kind: article\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="http"):
            load_source_entries(sources_file)

    def test_shipped_sources_file_is_valid(self) -> None:
        """docs/gurus/sources.yaml parses."""
        assert load_source_entries(Path("docs/gurus/sources.yaml"))


class TestFetchSource:
    """Tests for fetch_source."""

    def test_writes_markdown_with_front_matter(self, tmp_path: Path) -> None:
        """A fetched page is saved under the guru folder with its source URL."""
        client = _client(
            {
                "https://example.com/robots.txt": ALLOW_ALL,
                "https://example.com/article": httpx.Response(
                    200, text=ARTICLE_HTML, headers={"content-type": "text/html"}
                ),
            }
        )

        saved_path = fetch_source(ENTRY, client, tmp_path)

        content = saved_path.read_text(encoding="utf-8")
        assert saved_path.parent.name == "niwes"
        assert "guru_id: niwes" in content
        assert "source_url: https://example.com/article" in content
        assert "Buy good businesses." in content

    def test_robots_disallow_blocks_fetch(self, tmp_path: Path) -> None:
        """A path disallowed by robots.txt is never downloaded."""
        client = _client(
            {
                "https://example.com/robots.txt": httpx.Response(
                    200, text="User-agent: *\nDisallow: /\n"
                )
            }
        )

        with pytest.raises(SourceBlockedError, match="robots.txt"):
            fetch_source(ENTRY, client, tmp_path)

    def test_http_error_raises(self, tmp_path: Path) -> None:
        """A failed download raises instead of saving an empty file."""
        client = _client({"https://example.com/robots.txt": ALLOW_ALL})

        with pytest.raises(httpx.HTTPStatusError):
            fetch_source(ENTRY, client, tmp_path)


class TestFetchAllSources:
    """Tests for fetch_all_sources."""

    def test_reports_each_source_and_waits_between_requests(self, tmp_path: Path) -> None:
        """Failures are reported per source and requests are spaced out."""
        client = _client(
            {
                "https://example.com/robots.txt": ALLOW_ALL,
                "https://example.com/article": httpx.Response(200, text=ARTICLE_HTML),
            }
        )
        missing = SourceEntry(guru_id="niwes", url="https://example.com/missing", kind="article")
        sleeps: list[float] = []

        report = fetch_all_sources([ENTRY, missing], client, tmp_path, sleep=sleeps.append)

        assert [result.ok for result in report] == [True, False]
        assert sleeps == [1.0]
