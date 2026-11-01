"""Tests for input_normalizer: fixing how people type values."""

from datetime import date

import pytest

from finance_ai.tools.input_normalizer import (
    clean_number_text,
    clean_symbol_text,
    normalize_digits,
    parse_flexible_date,
    twelve_hour_to_24,
)


def test_normalize_digits() -> None:
    """Thai digits become Arabic; other text stays."""
    assert normalize_digits("๑๒๓ บาท") == "123 บาท"


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("$AAPL", "AAPL"),
        ("SET:PTT", "PTT"),
        ("set ptt", "ptt"),
        (" BKK:KBANK ", "KBANK"),
        ("ปตท", "ปตท"),
    ],
)
def test_clean_symbol_text(typed: str, expected: str) -> None:
    """Ticker decorations are removed, names are kept."""
    assert clean_symbol_text(typed) == expected


@pytest.mark.parametrize(("typed", "expected"), [("35.-", "35"), ("๓๕.๕๐", "35.50"), (" 7 ", "7")])
def test_clean_number_text(typed: str, expected: str) -> None:
    """Thai price marks and digits are cleaned."""
    assert clean_number_text(typed) == expected


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("8:30 pm", "20:30"),
        ("8.30PM", "20:30"),
        ("12 am", "00:00"),
        ("12:15 pm", "12:15"),
        ("7 am", "07:00"),
        ("20:00", "20:00"),
        ("๘:๓๐", "8:30"),
    ],
)
def test_twelve_hour_to_24(typed: str, expected: str) -> None:
    """am/pm becomes 24-hour; anything else passes through (digits normalized)."""
    assert twelve_hour_to_24(typed) == expected


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("2026-10-01", date(2026, 10, 1)),
        ("2026-1-5", date(2026, 1, 5)),
        ("1/10/2026", date(2026, 10, 1)),
        ("01-10-2026", date(2026, 10, 1)),
        ("1/10/2569", date(2026, 10, 1)),
        ("2569-10-01", date(2026, 10, 1)),
        ("29/02/2567", date(2024, 2, 29)),
        ("๑/๑๐/๒๕๖๙", date(2026, 10, 1)),
    ],
)
def test_parse_flexible_date(typed: str, expected: date) -> None:
    """AD, Buddhist era (incl. leap day), day-first and Thai digits."""
    assert parse_flexible_date(typed) == expected


@pytest.mark.parametrize("typed", ["31/02/2026", "tomorrow", "", "2026/10/01", "10-2026"])
def test_parse_flexible_date_rejects(typed: str) -> None:
    """Impossible or unsupported dates raise with an example."""
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_flexible_date(typed)
