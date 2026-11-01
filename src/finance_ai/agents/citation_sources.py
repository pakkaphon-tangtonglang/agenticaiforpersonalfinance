"""Numbered citation sources built from RAG search results.

Search tools number each retrieved document ([1], [2], ...) and return the
numbers with their results, so the model cites by number and the UI can show
a readable source card. Internal filenames never reach the user: documents
are keyed by a short hash and displayed by title and publisher.
"""

import hashlib
import re
import json
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from finance_ai.rag.document_models import RetrievalResult, SourceKind

SNIPPET_MAX_CHARS = 240
SNIPPET_MIN_CHARS = 40  # shorter than this after trimming → keep the doc's note instead
# Heading marks, emphasis, code ticks, quotes, table pipes and separator rows.
MARKDOWN_SYNTAX = re.compile(r"(?m)^\s*(?:#{1,6}|>|[-*+]\s)|\*+|__|`|\||:?-{3,}:?")
GENERIC_SOURCE_TITLE = "เอกสารอ้างอิง"
SEARCH_TOOL_NAMES = frozenset({"search_finance_knowledge", "search_guru_knowledge"})


class CitationSource(BaseModel):
    """One numbered source shown as a citation pill and popover card.

    Attributes:
        index: Citation number the model writes as [index].
        document_key: Opaque per-document key (hash, not the filename).
        title: Readable document title.
        url: Public URL of the source, if any.
        domain: Host of the URL without "www.", if any.
        publisher: Organisation or author of the source, if known.
        source_title: Title of the public source the document is based on.
        snippet: Short passage the answer drew on.
        kind: "original" for verbatim source text, else "compiled".

    Example:
        >>> CitationSource(index=1, document_key="ab12", title="คู่มือภาษี", snippet="...")
    """

    index: int
    document_key: str
    title: str
    url: str | None = None
    domain: str | None = None
    publisher: str | None = None
    source_title: str | None = None
    snippet: str = ""
    kind: SourceKind = "compiled"


def _document_key(source_file: str) -> str:
    """Return a short opaque key for a source file (an identifier, not security)."""
    return hashlib.sha1(source_file.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def _url_domain(url: str | None) -> str | None:
    """Return the URL host without a leading 'www.', or None."""
    if not url:
        return None
    host = urlparse(url).hostname or ""
    return host.removeprefix("www.") or None


def _shorten(text: str) -> str:
    """Drop markdown syntax, collapse whitespace and cut to SNIPPET_MAX_CHARS."""
    flat = " ".join(MARKDOWN_SYNTAX.sub(" ", text).split())
    if len(flat) <= SNIPPET_MAX_CHARS:
        return flat
    return flat[: SNIPPET_MAX_CHARS - 1].rstrip() + "…"


def _without_lines(content: str, title: str, drop_notes: bool) -> str:
    """Remove the heading that repeats the title, and optionally '>' note lines."""
    kept = [
        line
        for line in content.splitlines()
        if line.strip().lstrip("#").strip() != title
        and not (drop_notes and line.lstrip().startswith(">"))
    ]
    return "\n".join(kept)


def _body_text(content: str, title: str) -> str:
    """Pick the snippet text: skip the repeated title and '>' notes when possible.

    The card already shows the title, and a document's opening '>' note
    ("this is a summary of …") is not the passage the answer used. When a
    chunk holds little else, keep the note rather than show a bare heading.
    """
    for drop_notes in (True, False):
        body = _without_lines(content, title, drop_notes)
        if len(_shorten(body)) >= SNIPPET_MIN_CHARS:
            return body
    return content


def _citation_from_result(result: RetrievalResult, index: int) -> CitationSource:
    """Build a citation card for one retrieval result."""
    return CitationSource(
        index=index,
        document_key=_document_key(result.source_file),
        title=result.title or result.source_title or GENERIC_SOURCE_TITLE,
        url=result.source_url,
        domain=_url_domain(result.source_url),
        publisher=result.source_publisher,
        source_title=result.source_title,
        snippet=_shorten(_body_text(result.content, result.title)),
        kind=result.source_kind,
    )


def build_citations(
    results: list[RetrievalResult], existing: list[CitationSource]
) -> list[CitationSource]:
    """Number the documents in a search result, continuing earlier numbers.

    Args:
        results: Retrieval results from one search.
        existing: Citations already given out earlier in the same turn.

    Returns:
        One citation per distinct document in results, in result order.

    Example:
        >>> [c.index for c in build_citations(results, [])]
        [1, 2]
    """
    numbers = {citation.document_key: citation.index for citation in existing}
    next_number = max(numbers.values(), default=0) + 1
    citations: dict[str, CitationSource] = {}
    for result in results:
        key = _document_key(result.source_file)
        if key in citations:
            continue
        if key not in numbers:
            numbers[key], next_number = next_number, next_number + 1
        citations[key] = _citation_from_result(result, numbers[key])
    return list(citations.values())


def format_citations_as_context(
    results: list[RetrievalResult], citations: list[CitationSource]
) -> str:
    """Format results for the model, labelled with citation numbers and titles.

    Args:
        results: Retrieval results from one search.
        citations: Citations built from the same results.

    Returns:
        Context string such as "[1] คู่มือภาษี\nเนื้อหา...".

    Example:
        >>> format_citations_as_context(results, build_citations(results, []))
        '[1] คู่มือภาษี\n...'
    """
    by_key = {citation.document_key: citation for citation in citations}
    sections = []
    for result in results:
        citation = by_key[_document_key(result.source_file)]
        sections.append(f"[{citation.index}] {citation.title}\n{result.content}")
    return "\n\n".join(sections)


def _messages_of_latest_turn(messages: Iterable[Any]) -> list[Any]:
    """Return the messages after the last human message."""
    collected: list[Any] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            collected = []
        else:
            collected.append(message)
    return collected


def _citations_in_tool_message(message: Any) -> list[CitationSource]:
    """Parse the citations a search tool returned, or [] for anything else."""
    if not isinstance(message, ToolMessage) or message.name not in SEARCH_TOOL_NAMES:
        return []
    try:
        raw = json.loads(message.text).get("citations") or []
        return [CitationSource.model_validate(item) for item in raw]
    except (ValueError, AttributeError, TypeError, ValidationError):
        return []


def collect_turn_citations(messages: Iterable[Any]) -> list[CitationSource]:
    """Collect the distinct citations given out during the latest turn.

    Args:
        messages: Graph messages (LangChain messages or input tuples).

    Returns:
        Citations ordered by number, one per document.

    Example:
        >>> collect_turn_citations(final_state["messages"])
        [CitationSource(index=1, ...)]
    """
    by_number: dict[int, CitationSource] = {}
    for message in _messages_of_latest_turn(messages):
        for citation in _citations_in_tool_message(message):
            by_number.setdefault(citation.index, citation)
    return [by_number[number] for number in sorted(by_number)]
