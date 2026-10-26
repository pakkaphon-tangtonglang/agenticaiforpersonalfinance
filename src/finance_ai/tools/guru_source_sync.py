"""Sync allowlisted guru sources into docs/guru_knowledge/<guru_id>/fetched/.

One step from URL to indexable markdown: download (robots.txt respected),
cut to the article body or the configured section, and write with
front-matter. Used at deploy time so the public repo only stores links,
never the articles themselves (the fetched/ folders are gitignored).
Sources whose layout changed or whose text is too short are skipped.
No LLM calls.
"""

import time
from collections.abc import Callable
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel

from finance_ai.core.logging import get_logger
from finance_ai.tools.guru_raw_cleaner import clean_text, markers_found, rule_for_entry
from finance_ai.tools.guru_source_fetcher import (
    REQUEST_DELAY_SECONDS,
    DownloadedSource,
    SourceBlockedError,
    SourceEntry,
    download_source,
    source_slug,
)

logger = get_logger(__name__)

FETCHED_DIRNAME = "fetched"
MIN_BODY_CHARS = 800


class SourceSkippedError(ValueError):
    """Raised when a downloaded source should not be indexed."""


class SyncResult(BaseModel):
    """Outcome of syncing one source."""

    url: str
    ok: bool
    detail: str
    path: str | None = None


def build_fetched_markdown(entry: SourceEntry, source: DownloadedSource) -> str:
    """Clean a downloaded source and wrap it with front-matter.

    Args:
        entry: The allowlisted source (with optional section markers).
        source: Downloaded title and text.

    Returns:
        Markdown ready for the guru knowledge folder.

    Raises:
        SourceSkippedError: If the start marker is missing or the body is too short.

    Example:
        >>> build_fetched_markdown(entry, source).startswith("---")
        True
    """
    rule = rule_for_entry(entry)
    if not markers_found(source.text, rule):
        raise SourceSkippedError(f"Start marker not found (layout changed?): {entry.url}")
    body = clean_text(source.text, rule, is_pdf=source.is_pdf)
    if len(body) < MIN_BODY_CHARS:
        raise SourceSkippedError(f"Body too short ({len(body)} chars): {entry.url}")
    title = (entry.title or source.title or entry.url).replace("\n", " ")
    header: dict[str, str | bool] = {
        "guru_id": entry.guru_id,
        "source_title": title,
        "source_url": entry.url,
        "kind": entry.kind,
        "reviewed": False,
        "auto_fetched": True,
    }
    front_matter = yaml.safe_dump(header, allow_unicode=True, sort_keys=False)
    return f"---\n{front_matter}---\n# {title}\n\n{body}\n"


def _sync_one(entry: SourceEntry, client: httpx.Client, knowledge_dir: Path) -> SyncResult:
    """Download, clean and write one source; failures become a result."""
    try:
        markdown = build_fetched_markdown(entry, download_source(entry, client))
    except (SourceBlockedError, SourceSkippedError, httpx.HTTPError, ValueError) as error:
        logger.warning("Guru source skipped: %s", error)
        return SyncResult(url=entry.url, ok=False, detail=str(error))
    target = knowledge_dir / entry.guru_id / FETCHED_DIRNAME / f"{source_slug(entry.url)}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    return SyncResult(url=entry.url, ok=True, detail="saved", path=str(target))


def sync_guru_sources(
    entries: list[SourceEntry],
    client: httpx.Client,
    knowledge_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
) -> list[SyncResult]:
    """Sync every allowlisted source, one request at a time.

    Args:
        entries: Allowlisted sources.
        client: HTTP client.
        knowledge_dir: docs/guru_knowledge root.
        sleep: Delay function between sources (injectable for tests).

    Returns:
        One SyncResult per entry, in order. Re-runs overwrite the same files.

    Example:
        >>> sync_guru_sources(entries, client, Path("docs/guru_knowledge"))
    """
    results: list[SyncResult] = []
    for index, entry in enumerate(entries):
        if index:
            sleep(REQUEST_DELAY_SECONDS)
        results.append(_sync_one(entry, client, knowledge_dir))
    return results
