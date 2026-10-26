"""Fetch guru source pages from an explicit allowlist (docs/gurus/sources.yaml).

Polite by design: only listed http(s) URLs, robots.txt is respected, a
delay separates requests, and output lands in data/guru_raw/ for human
review before anything is promoted to docs/guru_knowledge/.
No LLM calls are made here.
"""

import io
import re
import time
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx
import yaml
from pydantic import BaseModel, Field, field_validator

from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

USER_AGENT = "PersonalFinanceAI-GuruFetcher/1.0 (educational research)"
REQUEST_DELAY_SECONDS = 1.0
_SKIPPED_TAGS = {"script", "style", "nav", "footer", "header", "noscript", "svg"}


class SourceBlockedError(RuntimeError):
    """Raised when robots.txt disallows fetching a source."""


class SourceEntry(BaseModel):
    """One allowlisted public source."""

    guru_id: str = Field(min_length=1)
    url: str
    kind: str = "article"
    title: str | None = None
    note: str | None = None
    start_marker: str | None = None
    end_marker: str | None = None

    @field_validator("url")
    @classmethod
    def _require_http(cls, url: str) -> str:
        """Only http(s) URLs may be fetched."""
        if urlsplit(url).scheme not in {"http", "https"}:
            raise ValueError(f"Source URL must use http or https. Received: {url}")
        return url


class FetchResult(BaseModel):
    """Outcome of fetching one source."""

    url: str
    ok: bool
    detail: str


class _TextExtractor(HTMLParser):
    """Collect <title> and visible text, skipping boilerplate tags."""

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self.parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track entry into skipped tags and the title."""
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        self._in_title = tag == "title"

    def handle_endtag(self, tag: str) -> None:
        """Track exit from skipped tags and the title."""
        if tag in _SKIPPED_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        """Keep visible text outside skipped tags."""
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title = text
        elif not self._skip_depth:
            self.parts.append(text)


def html_to_text(html: str) -> tuple[str, str]:
    """Extract the page title and readable body text from HTML.

    Args:
        html: Raw HTML.

    Returns:
        Tuple of (title, text with one block per line).

    Example:
        >>> html_to_text("<title>T</title><p>Hi</p>")
        ('T', 'Hi')
    """
    extractor = _TextExtractor()
    extractor.feed(html)
    return extractor.title, "\n".join(extractor.parts)


def load_source_entries(sources_path: Path) -> list[SourceEntry]:
    """Load and validate the allowlist file.

    Args:
        sources_path: YAML file with a top-level 'sources' list.

    Returns:
        Validated source entries.

    Raises:
        ValueError: If a URL is not http(s) or the file is malformed.

    Example:
        >>> load_source_entries(Path("docs/gurus/sources.yaml"))[0].guru_id
        'buffett'
    """
    raw = yaml.safe_load(sources_path.read_text(encoding="utf-8")) or {}
    return [SourceEntry.model_validate(item) for item in raw.get("sources", [])]


def is_allowed_by_robots(url: str, client: httpx.Client) -> bool:
    """Check robots.txt for the URL's site (missing robots.txt allows).

    Args:
        url: Page to fetch.
        client: HTTP client.

    Returns:
        True when our user agent may fetch the URL.
    """
    parts = urlsplit(url)
    response = client.get(f"{parts.scheme}://{parts.netloc}/robots.txt")
    if response.status_code >= 400:
        return True
    parser = RobotFileParser()
    parser.parse(response.text.splitlines())
    return parser.can_fetch(USER_AGENT, url)


class DownloadedSource(BaseModel):
    """Text extracted from one fetched source."""

    title: str
    text: str
    is_pdf: bool


def _is_pdf_response(response: httpx.Response) -> bool:
    """True when the response is a PDF document."""
    return "pdf" in response.headers.get("content-type", "") or response.url.path.endswith(".pdf")


def _response_to_text(response: httpx.Response) -> tuple[str, str]:
    """Turn an HTML or PDF response into (title, text)."""
    if _is_pdf_response(response):
        from pypdf import PdfReader  # pylint: disable=import-outside-toplevel

        pages = PdfReader(io.BytesIO(response.content)).pages
        return "", "\n\n".join(page.extract_text() or "" for page in pages)
    return html_to_text(response.text)


def download_source(entry: SourceEntry, client: httpx.Client) -> DownloadedSource:
    """Check robots.txt, download one source and extract its text.

    Args:
        entry: The allowlisted source.
        client: HTTP client.

    Returns:
        DownloadedSource with title, text and whether it was a PDF.

    Raises:
        SourceBlockedError: If robots.txt disallows the URL.
        httpx.HTTPStatusError: If the download fails.

    Example:
        >>> download_source(entry, client).is_pdf
        True
    """
    if not is_allowed_by_robots(entry.url, client):
        raise SourceBlockedError(f"robots.txt disallows fetching {entry.url}")
    response = client.get(entry.url, follow_redirects=True)
    response.raise_for_status()
    title, text = _response_to_text(response)
    return DownloadedSource(title=title, text=text, is_pdf=_is_pdf_response(response))


def source_slug(url: str) -> str:
    """Filesystem-safe name derived from the URL path.

    Example:
        >>> source_slug("https://www.finnomena.com/dr-niwes/vi-to-ai")
        'www_finnomena_com_dr_niwes_vi_to_ai'
    """
    parts = urlsplit(url)
    raw = f"{parts.netloc}{parts.path}".rstrip("/") or parts.netloc
    return re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")[:80]


def _to_markdown(entry: SourceEntry, title: str, text: str) -> str:
    """Wrap fetched text with the front-matter the RAG loader expects."""
    source_title = (entry.title or title or entry.url).replace("\n", " ")
    front_matter = yaml.safe_dump(
        {
            "guru_id": entry.guru_id,
            "source_title": source_title,
            "source_url": entry.url,
            "kind": entry.kind,
            "reviewed": False,
        },
        allow_unicode=True,
        sort_keys=False,
    )
    return f"---\n{front_matter}---\n# {source_title}\n\n{text}\n"


def fetch_source(entry: SourceEntry, client: httpx.Client, output_dir: Path) -> Path:
    """Download one allowlisted source and save it as markdown for review.

    Args:
        entry: The allowlisted source.
        client: HTTP client.
        output_dir: Root folder; the file goes to output_dir/<guru_id>/.

    Returns:
        Path of the saved markdown file.

    Raises:
        SourceBlockedError: If robots.txt disallows the URL.
        httpx.HTTPStatusError: If the download fails.

    Example:
        >>> fetch_source(entry, httpx.Client(), Path("data/guru_raw"))
    """
    source = download_source(entry, client)
    target = output_dir / entry.guru_id / f"{source_slug(entry.url)}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_to_markdown(entry, source.title, source.text), encoding="utf-8")
    logger.info("Saved %s -> %s", entry.url, target)
    return target


def fetch_all_sources(
    entries: list[SourceEntry],
    client: httpx.Client,
    output_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> list[FetchResult]:
    """Fetch every entry, one at a time, reporting failures instead of stopping.

    Args:
        entries: Allowlisted sources.
        client: HTTP client.
        output_dir: Root output folder.
        sleep: Delay function (injectable for tests).

    Returns:
        One FetchResult per entry, in order.

    Example:
        >>> fetch_all_sources(entries, client, Path("data/guru_raw"))
    """
    results: list[FetchResult] = []
    for index, entry in enumerate(entries):
        if index:
            sleep(REQUEST_DELAY_SECONDS)
        try:
            saved_path = fetch_source(entry, client, output_dir)
            results.append(FetchResult(url=entry.url, ok=True, detail=str(saved_path)))
        except (SourceBlockedError, httpx.HTTPError, ValueError) as error:
            results.append(FetchResult(url=entry.url, ok=False, detail=str(error)))
    return results
