"""Low-level client for the SEC Thailand Open API (Fund API V2, api.sec.or.th).

Every Fund API endpoint returns `{"items": [...], "next_cursor": ...}` pages
and answers 204 (no body) when nothing matches. This module fetches items,
follows cursors, and caches answers for a while; callers never see HTTP.

No key configured, or any network/API failure → an empty list (logged), so
the app keeps working without fund data.
"""

import time
from typing import Any

import httpx

from finance_ai.core.config import get_settings
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

SEC_API_BASE = "https://api.sec.or.th"
SEC_KEY_HEADER = "Ocp-Apim-Subscription-Key"
PAGE_SIZE = 100  # API maximum
REQUEST_TIMEOUT_SECONDS = 10.0
NO_CONTENT = 204  # the API's answer when nothing matches

NAV_PATH = "/v2/fund/daily-info/nav"
PROFILES_PATH = "/v2/fund/general-info/profiles"
AMCS_PATH = "/v2/fund/general-info/amcs"

_cache: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[float, list[dict[str, Any]]]] = {}


def is_configured() -> bool:
    """True when an SEC API key is set."""
    return bool(get_settings().sec_api_key)


def clear_cache() -> None:
    """Forget cached answers (used by tests)."""
    _cache.clear()


def fetch_items(
    path: str, params: dict[str, Any], max_pages: int = 1, ttl_seconds: float = 600.0
) -> list[dict[str, Any]]:
    """All items for a Fund API query, following cursors up to max_pages.

    Args:
        path: Endpoint path, e.g. NAV_PATH.
        params: Query parameters (page_size is added).
        max_pages: Most pages to read.
        ttl_seconds: How long a successful answer is reused.

    Returns:
        The items, or [] without a key, when nothing matches, or on errors.

    Example:
        >>> fetch_items(AMCS_PATH, {})  # doctest: +SKIP
        [{'unique_id': 'C0000000021', 'comp_name_th': '...'}]
    """
    key = (path, tuple(sorted((name, str(value)) for name, value in params.items())))
    cached = _cache.get(key)
    if cached and time.monotonic() - cached[0] < ttl_seconds:
        return list(cached[1])
    api_key = get_settings().sec_api_key
    if not api_key:
        return []
    items = _fetch_pages(path, {**params, "page_size": PAGE_SIZE}, max_pages, api_key)
    if items is None:
        return []
    _cache[key] = (time.monotonic(), items)
    return list(items)


def _fetch_pages(
    path: str, params: dict[str, Any], max_pages: int, api_key: str
) -> list[dict[str, Any]] | None:
    """Read pages until the last one or max_pages (None on failure)."""
    items: list[dict[str, Any]] = []
    query = dict(params)
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            for _ in range(max_pages):
                page = _get_page(client, path, query, api_key)
                items.extend(page.get("items") or [])
                if not page.get("next_cursor"):
                    break
                query["next_cursor"] = page["next_cursor"]
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("SEC Fund API %s failed for %s: %s", path, params, exc)
        return None
    return items


def _get_page(
    client: httpx.Client, path: str, params: dict[str, Any], api_key: str
) -> dict[str, Any]:
    """One page; 204 (nothing matches) is an empty page."""
    response = client.get(SEC_API_BASE + path, params=params, headers={SEC_KEY_HEADER: api_key})
    response.raise_for_status()
    if response.status_code == NO_CONTENT or not response.content:
        return {}
    page: dict[str, Any] = response.json()
    return page
