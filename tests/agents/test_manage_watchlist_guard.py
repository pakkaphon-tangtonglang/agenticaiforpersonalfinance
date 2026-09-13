"""Tests for the manage_watchlist add-path symbol guard.

Verifies that invalid symbols return {"error": ...} instead of being
stored, and that free-text input is resolved to canonical Yahoo symbols.
"""

# pylint: disable=redefined-outer-name

from collections.abc import Callable

import pytest
from sqlalchemy.orm import Session

from finance_ai.agents.asset_monitoring_tools import manage_watchlist
from finance_ai.database.crud.watched_asset_crud import WatchedAssetCRUD
from finance_ai.database.models.user import User
from finance_ai.tools.market_data_models import AssetSymbolMatch


def make_match(symbol: str, name: str) -> AssetSymbolMatch:
    """Build a single search candidate.

    Args:
        symbol: Canonical Yahoo symbol of the candidate.
        name: Display name of the candidate.

    Returns:
        AssetSymbolMatch usable in mocked search results.
    """
    return AssetSymbolMatch(symbol=symbol, name=name, exchange="SET", quote_type="EQUITY")


def stored_symbols(
    db_session_factory: Callable[[], Session],
    user_id: str,
) -> list[str]:
    """List the symbols currently stored for a user.

    Args:
        db_session_factory: Session factory bound to the test engine.
        user_id: UUID of the user.

    Returns:
        List of stored symbol strings.
    """
    with db_session_factory() as session:
        assets = WatchedAssetCRUD().get_by_user(session, user_id)
        return [asset.symbol for asset in assets]


class TestManageWatchlistAddGuard:
    """The add path rejects unresolvable symbols with a Thai error."""

    def test_invalid_symbol_returns_error(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unresolvable symbol returns an error and stores nothing."""
        monkeypatch.setattr("finance_ai.tools.symbol_guard.search_asset_symbols", lambda query: [])

        result = manage_watchlist.invoke(
            {
                "action": "add",
                "symbol": "APPL",
                "name": "Apple",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )

        assert "error" in result
        assert "APPL" in result["error"]
        assert "ไม่พบสัญลักษณ์" in result["error"]
        assert stored_symbols(db_session_factory, sample_user.id) == []

    def test_free_text_resolves_to_canonical(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ptt is resolved to PTT.BK before being stored."""
        monkeypatch.setattr(
            "finance_ai.tools.symbol_guard.search_asset_symbols",
            lambda query: [make_match("PTT.BK", "PTT Public Company Limited")],
        )

        result = manage_watchlist.invoke(
            {
                "action": "add",
                "symbol": "ptt",
                "name": "PTT",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )

        assert result["status"] == "added"
        assert result["symbol"] == "PTT.BK"
        assert stored_symbols(db_session_factory, sample_user.id) == ["PTT.BK"]

    def test_duplicate_after_canonicalization(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Adding ptt then PTT.BK deduplicates to already_exists."""
        monkeypatch.setattr(
            "finance_ai.tools.symbol_guard.search_asset_symbols",
            lambda query: [make_match("PTT.BK", "PTT Public Company Limited")],
        )
        base_args = {
            "action": "add",
            "name": "PTT",
            "user_id": sample_user.id,
            "db_session_factory": db_session_factory,
        }

        first_args = dict(base_args, symbol="ptt")
        second_args = dict(base_args, symbol="PTT.BK")
        first = manage_watchlist.invoke(first_args)
        second = manage_watchlist.invoke(second_args)

        assert first["status"] == "added"
        assert second["status"] == "already_exists"
        assert stored_symbols(db_session_factory, sample_user.id) == ["PTT.BK"]

    def test_single_strong_match_resolves(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """appl with a lone AAPL candidate resolves and stores AAPL."""
        monkeypatch.setattr(
            "finance_ai.tools.symbol_guard.search_asset_symbols",
            lambda query: [make_match("AAPL", "Apple Inc.")],
        )

        result = manage_watchlist.invoke(
            {
                "action": "add",
                "symbol": "appl",
                "name": "Apple",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )

        assert result["status"] == "added"
        assert result["symbol"] == "AAPL"
        assert stored_symbols(db_session_factory, sample_user.id) == ["AAPL"]
