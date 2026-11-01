"""Tests for clarification results (ask the user instead of guessing)."""

from finance_ai.agents.clarification import (
    ASK_FOR_FULL_NAME,
    ASK_WHICH_ONE,
    clarification,
    symbol_problem,
)
from finance_ai.tools.market_data_models import AssetSymbolMatch
from finance_ai.tools.symbol_guard import SymbolResolutionError, SymbolSearchUnavailableError


def test_clarification_without_options() -> None:
    """No options key when there is nothing to choose from."""
    assert clarification("missing", "ถามอะไร") == {
        "error": "missing",
        "needs_clarification": True,
        "ask_user": "ถามอะไร",
    }


def test_clarification_with_options() -> None:
    """Options are passed through."""
    assert clarification("x", "q", ["A", "B"])["options"] == ["A", "B"]


def test_ambiguous_symbol_offers_candidates() -> None:
    """Candidates become 'SYMBOL — name' options."""
    match = AssetSymbolMatch(symbol="PTT.BK", name="PTT PCL", exchange="SET", quote_type="EQUITY")
    result = symbol_problem(SymbolResolutionError("ambiguous", candidates=[match]))
    assert result["ask_user"] == ASK_WHICH_ONE
    assert result["options"] == ["PTT.BK — PTT PCL"]


def test_unknown_symbol_asks_again() -> None:
    """No candidates: ask for the full name."""
    result = symbol_problem(SymbolResolutionError("not found"))
    assert result["ask_user"] == ASK_FOR_FULL_NAME
    assert "options" not in result


def test_search_outage_is_not_a_question() -> None:
    """A search outage says retry later, it does not ask the user to retype."""
    result = symbol_problem(SymbolSearchUnavailableError("down"))
    assert result == {"error": "down", "retry_later": True}
