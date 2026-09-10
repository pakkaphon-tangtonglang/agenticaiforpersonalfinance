"""Runtime bootstrap entry point for hosted deployments.

Applies database migrations and indexes the RAG knowledge base when
empty. Safe to run repeatedly (idempotent). Used as the Render release
command and via ``make bootstrap``.

Example:
    uv run python scripts/bootstrap_runtime.py
"""

from finance_ai.core.bootstrap import run_bootstrap

if __name__ == "__main__":
    run_bootstrap()
