"""Tests for asset monitoring agent tool wrappers."""

from collections.abc import Callable

import pytest
from sqlalchemy.orm import Session

from finance_ai.agents.asset_monitoring_tools import manage_watchlist
from finance_ai.database.models.user import User
from finance_ai.tools.market_data_models import AssetSymbolMatch


@pytest.fixture(autouse=True)
def mock_symbol_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub symbol resolution so watchlist tests make no network calls."""

    def fake_search(query: str) -> list[AssetSymbolMatch]:
        """Echo the query back as a single exact candidate."""
        upper = query.strip().upper()
        return [AssetSymbolMatch(symbol=upper, name=upper, exchange="MOCK", quote_type="EQUITY")]

    monkeypatch.setattr("finance_ai.tools.symbol_guard.search_asset_symbols", fake_search)


class TestManageWatchlistAdd:
    """Tests for manage_watchlist action='add'."""

    def test_adds_symbol(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Should add a symbol to the watchlist."""
        result = manage_watchlist.invoke(
            {
                "action": "add",
                "symbol": "PTT.BK",
                "name": "PTT",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "add"
        assert result["symbol"] == "PTT.BK"
        assert result["status"] == "added"

    def test_duplicate_add_returns_already_exists(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Adding the same symbol twice should return already_exists."""
        manage_watchlist.invoke(
            {
                "action": "add",
                "symbol": "BTC-USD",
                "name": "Bitcoin",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        result = manage_watchlist.invoke(
            {
                "action": "add",
                "symbol": "BTC-USD",
                "name": "Bitcoin",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["status"] == "already_exists"

    def test_add_without_symbol_returns_error(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Should return error dict when symbol is missing."""
        result = manage_watchlist.invoke(
            {
                "action": "add",
                "symbol": "",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert "error" in result

    def test_add_stores_name(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Name should be stored and returned."""
        result = manage_watchlist.invoke(
            {
                "action": "add",
                "symbol": "GC=F",
                "name": "ทองคำ",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["name"] == "ทองคำ"


class TestManageWatchlistRemove:
    """Tests for manage_watchlist action='remove'."""

    def test_removes_existing_symbol(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Should remove an existing symbol and return status=removed."""
        manage_watchlist.invoke(
            {
                "action": "add",
                "symbol": "AAPL",
                "name": "Apple",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        result = manage_watchlist.invoke(
            {
                "action": "remove",
                "symbol": "AAPL",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "remove"
        assert result["status"] == "removed"

    def test_remove_nonexistent_returns_not_found(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Should return status=not_found for non-existent symbol."""
        result = manage_watchlist.invoke(
            {
                "action": "remove",
                "symbol": "FAKE.BK",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["status"] == "not_found"

    def test_remove_without_symbol_returns_error(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Should return error dict when symbol is missing."""
        result = manage_watchlist.invoke(
            {
                "action": "remove",
                "symbol": "",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert "error" in result


class TestManageWatchlistList:
    """Tests for manage_watchlist action='list'."""

    def test_empty_watchlist(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Should return empty watchlist for new user."""
        result = manage_watchlist.invoke(
            {
                "action": "list",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["action"] == "list"
        assert result["count"] == 0
        assert result["watchlist"] == []

    def test_lists_added_symbols(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Should return all added symbols."""
        for symbol, name in [("PTT.BK", "PTT"), ("AOT.BK", "AOT")]:
            manage_watchlist.invoke(
                {
                    "action": "add",
                    "symbol": symbol,
                    "name": name,
                    "user_id": sample_user.id,
                    "db_session_factory": db_session_factory,
                }
            )
        result = manage_watchlist.invoke(
            {
                "action": "list",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["count"] == 2
        symbols = [item["symbol"] for item in result["watchlist"]]
        assert "PTT.BK" in symbols
        assert "AOT.BK" in symbols

    def test_count_decreases_after_remove(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Count should decrease after removing a symbol."""
        manage_watchlist.invoke(
            {
                "action": "add",
                "symbol": "PTT.BK",
                "name": "PTT",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        manage_watchlist.invoke(
            {
                "action": "remove",
                "symbol": "PTT.BK",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        result = manage_watchlist.invoke(
            {
                "action": "list",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert result["count"] == 0


class TestManageWatchlistUnknownAction:
    """Tests for manage_watchlist with unknown action."""

    def test_unknown_action_returns_error(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
    ) -> None:
        """Should return error dict for unknown action."""
        result = manage_watchlist.invoke(
            {
                "action": "unknown_action",
                "user_id": sample_user.id,
                "db_session_factory": db_session_factory,
            }
        )
        assert "error" in result
