"""Seed the database with demo data for the thesis defense demo.

Wipes and re-creates a fixed demo user with ~3 months of Thai-labeled
expenses, salary income, RMF/SSF deductions, goals, watchlist entries,
and investment holdings. Idempotent — safe to run before every demo.

Example:
    uv run python scripts/seed_demo.py
"""

from finance_ai.tools.demo_seed_service import seed_demo_database

if __name__ == "__main__":
    print(seed_demo_database())  # noqa: T201
