"""Trim fetched guru pages in data/guru_raw/ into data/guru_clean/ for review.

Raw files are not modified. No LLM calls.

Example:
    uv run python scripts/clean_guru_raw.py
"""

from pathlib import Path

from finance_ai.tools.guru_raw_cleaner import clean_raw_file

RAW_DIR = Path("data/guru_raw")
CLEAN_DIR = Path("data/guru_clean")

if __name__ == "__main__":
    for raw_path in sorted(RAW_DIR.glob("*/*.md")):
        cleaned_path = clean_raw_file(raw_path, CLEAN_DIR)
        before, after = raw_path.stat().st_size, cleaned_path.stat().st_size
        print(f"{before:>7} -> {after:>7} bytes  {cleaned_path}")
