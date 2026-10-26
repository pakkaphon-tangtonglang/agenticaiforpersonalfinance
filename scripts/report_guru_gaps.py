"""Show the guru questions the knowledge base most often could not answer.

Use it to decide which sources to add to docs/gurus/sources.yaml next.
No LLM calls.

Example:
    uv run python scripts/report_guru_gaps.py
    uv run python scripts/report_guru_gaps.py --guru buffett --limit 10
"""

import argparse

from finance_ai.database.session import create_database_engine, create_session_factory
from finance_ai.tools.guru_gap_service import list_top_gaps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guru", default=None, help="Only this guru_id")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    session = create_session_factory(create_database_engine())()
    try:
        gaps = list_top_gaps(session, limit=args.limit, guru_id=args.guru)
    finally:
        session.close()
    if not gaps:
        print("No knowledge gaps logged yet.")
    for guru_id, query, count in gaps:
        print(f"{count:>4}  {guru_id:<11} {query}")
