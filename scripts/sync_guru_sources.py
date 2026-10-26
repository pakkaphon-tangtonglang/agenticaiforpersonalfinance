"""Fetch allowlisted guru sources into docs/guru_knowledge/*/fetched/ and reindex.

Same pipeline the deployed app runs in the background on startup
(GURU_FETCH_ON_STARTUP=true). Uses only the embedding provider, no chat LLM.

Example:
    uv run python scripts/sync_guru_sources.py
"""

from pathlib import Path

import httpx

from finance_ai.core.bootstrap import rebuild_guru_knowledge
from finance_ai.core.config import get_settings
from finance_ai.tools.guru_source_fetcher import USER_AGENT, load_source_entries
from finance_ai.tools.guru_source_sync import sync_guru_sources

SOURCES_FILE = Path("docs/gurus/sources.yaml")

if __name__ == "__main__":
    knowledge_dir = Path(get_settings().rag_guru_knowledge_directory)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        results = sync_guru_sources(load_source_entries(SOURCES_FILE), client, knowledge_dir)
    for result in results:
        print(
            f"{'OK  ' if result.ok else 'SKIP'} {result.url[:90]} -> {result.path or result.detail}"
        )
    print(f"Indexed {rebuild_guru_knowledge()} guru chunks")
