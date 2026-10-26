"""Fetch allowlisted guru sources into data/guru_raw/ for review.

Reads docs/gurus/sources.yaml, respects robots.txt, waits between
requests and makes no LLM calls.

Example:
    uv run python scripts/fetch_guru_sources.py
"""

from pathlib import Path

import httpx

from finance_ai.tools.guru_source_fetcher import (
    USER_AGENT,
    fetch_all_sources,
    load_source_entries,
)

SOURCES_FILE = Path("docs/gurus/sources.yaml")
OUTPUT_DIR = Path("data/guru_raw")

if __name__ == "__main__":
    entries = load_source_entries(SOURCES_FILE)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        for result in fetch_all_sources(entries, client, OUTPUT_DIR):
            print(f"{'OK  ' if result.ok else 'FAIL'} {result.url} -> {result.detail}")
