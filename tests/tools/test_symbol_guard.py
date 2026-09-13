"""Tests for the symbol resolution guard (symbol_guard module).

All tests mock search_asset_symbols so no network calls are made.
"""

# pylint: disable=redefined-outer-name

import pytest

from finance_ai.tools.market_data_models import AssetSymbolMatch
from finance_ai.tools.symbol_guard import (
    SymbolResolutionError,
    resolve_and_validate_symbol,
)


def make_match(symbol: str, name: str = "Mock Asset") -> AssetSymbolMatch:
    """Build a single search candidate.

    Args:
        symbol: Canonical Yahoo symbol of the candidate.
        name: Display name of the candidate.

    Returns:
        AssetSymbolMatch usable in mocked search results.
    """
    return AssetSymbolMatch(symbol=symbol, name=name, exchange="SET", quote_type="EQUITY")


def stub_search(
    monkeypatch: pytest.MonkeyPatch,
    matches: list[AssetSymbolMatch] | None = None,
    capture: dict[str, str] | None = None,
    raises: Exception | None = None,
) -> None:
    """Replace search_asset_symbols with a canned response.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        matches: Candidates to return (ignored when raises is set).
        capture: Optional dict receiving the raw query for assertions.
        raises: Optional exception for the stub to raise.
    """

    def fake_search(query: str) -> list[AssetSymbolMatch]:
        """Return the canned matches after optionally capturing the query."""
        if capture is not None:
            capture["query"] = query
        if raises is not None:
            raise raises
        return matches or []

    monkeypatch.setattr("finance_ai.tools.symbol_guard.search_asset_symbols", fake_search)


class TestExactSymbolResolution:
    """Inputs matching a candidate symbol resolve to the canonical symbol."""

    def test_exact_symbol_case_insensitive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """aapl resolves to AAPL via a case-insensitive exact match."""
        stub_search(monkeypatch, [make_match("AAPL", "Apple Inc.")])

        assert resolve_and_validate_symbol("aapl") == "AAPL"

    def test_bk_suffix_stripped_for_set_stocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ptt resolves to the canonical PTT.BK Yahoo symbol."""
        stub_search(
            monkeypatch,
            [make_match("PTT.BK", "PTT Public Company Limited"), make_match("PTTGC.BK")],
        )

        assert resolve_and_validate_symbol("ptt") == "PTT.BK"

    def test_full_symbol_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A full canonical symbol resolves to itself."""
        stub_search(monkeypatch, [make_match("PTT.BK", "PTT Public Company Limited")])

        assert resolve_and_validate_symbol("PTT.BK") == "PTT.BK"

    def test_whitespace_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Surrounding whitespace is stripped before resolution."""
        capture: dict[str, str] = {}
        stub_search(monkeypatch, [make_match("AAPL", "Apple Inc.")], capture)

        resolve_and_validate_symbol("  aapl  ")

        assert capture["query"] == "aapl"


class TestSingleStrongMatch:
    """A lone candidate is accepted even without an exact symbol match."""

    def test_single_candidate_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """appl with a single AAPL candidate resolves to AAPL."""
        stub_search(monkeypatch, [make_match("AAPL", "Apple Inc.")])

        assert resolve_and_validate_symbol("appl") == "AAPL"

    def test_multiple_non_exact_candidates_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ambiguous candidates never invent a symbol."""
        stub_search(monkeypatch, [make_match("AAP"), make_match("APAM")])

        with pytest.raises(SymbolResolutionError):
            resolve_and_validate_symbol("ap")


class TestUnresolvableInput:
    """Unresolvable inputs raise SymbolResolutionError with Thai guidance."""

    def test_empty_input_raises(self) -> None:
        """Whitespace-only input is rejected."""
        with pytest.raises(SymbolResolutionError):
            resolve_and_validate_symbol("   ")

    def test_no_matches_raises_with_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The error message echoes the raw input for actionability."""
        stub_search(monkeypatch, [])

        with pytest.raises(SymbolResolutionError) as exc_info:
            resolve_and_validate_symbol("APPL")

        message = str(exc_info.value)
        assert "APPL" in message
        assert "ไม่พบสัญลักษณ์" in message
        assert "กรุณาตรวจสอบตัวย่อหุ้น" in message

    def test_search_error_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failing search degrades to SymbolResolutionError, not a crash."""
        stub_search(monkeypatch, raises=RuntimeError("network down"))

        with pytest.raises(SymbolResolutionError):
            resolve_and_validate_symbol("AAPL")

    def test_error_is_value_error_subclass(self) -> None:
        """SymbolResolutionError is a ValueError so callers can catch broadly."""
        assert issubclass(SymbolResolutionError, ValueError)
