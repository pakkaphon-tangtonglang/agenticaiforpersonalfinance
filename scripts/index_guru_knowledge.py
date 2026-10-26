"""Rebuild the guru knowledge collection from docs/guru_knowledge.

Run after adding or editing guru documents. Uses only the configured
embedding provider (no chat LLM calls).

Example:
    uv run python scripts/index_guru_knowledge.py
"""

from finance_ai.core.bootstrap import rebuild_guru_knowledge

if __name__ == "__main__":
    print(f"Indexed {rebuild_guru_knowledge()} guru chunks")
