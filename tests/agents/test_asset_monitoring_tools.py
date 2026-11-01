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


def _call(user: User, factory: Callable[[], Session], **arguments: str) -> dict[str, object]:
    """Invoke manage_watchlist for a user."""
    payload: dict[str, object] = {"user_id": user.id, "db_session_factory": factory, **arguments}
    result: dict[str, object] = manage_watchlist.invoke(payload)
    return result


class TestWatchlistAlerts:
    """Daily news alert times through manage_watchlist."""

    def test_add_without_time_asks_for_one(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Adding with no time tells the agent to ask the user."""
        result = _call(sample_user, db_session_factory, action="add", symbol="PTT.BK")
        assert result["alert_time"] is None
        assert "HH:MM" in str(result["next_step"])

    def test_add_with_time_sets_alert(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """A time given with the add is saved and shown by list."""
        result = _call(
            sample_user, db_session_factory, action="add", symbol="PTT.BK", alert_time="8.30"
        )
        assert result["alert_time"] == "08:30"
        listed = _call(sample_user, db_session_factory, action="list")
        assert listed["watchlist"] == [{"symbol": "PTT.BK", "name": "", "alert_time": "08:30"}]

    def test_add_with_bad_time_reports_error(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """The asset is still added; the bad time is reported."""
        result = _call(
            sample_user, db_session_factory, action="add", symbol="PTT.BK", alert_time="25:00"
        )
        assert result["status"] == "added"
        assert "24-hour" in str(result["alert_error"])

    def test_set_alert_requires_watched_asset(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Only watched assets can get an alert."""
        result = _call(
            sample_user, db_session_factory, action="set_alert", symbol="AOT.BK", alert_time="09:00"
        )
        assert "not on the watchlist" in str(result["error"])

    def test_set_alert_requires_time(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """set_alert without a time is an error."""
        result = _call(sample_user, db_session_factory, action="set_alert", symbol="PTT.BK")
        assert "alert_time" in str(result["error"])

    def test_set_alert_invalid_time(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """An invalid time comes back as an error."""
        _call(sample_user, db_session_factory, action="add", symbol="PTT.BK")
        result = _call(
            sample_user, db_session_factory, action="set_alert", symbol="PTT.BK", alert_time="noon"
        )
        assert "24-hour" in str(result["error"])

    def test_set_then_clear_alert(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """set_alert saves the time; clear_alert turns it off."""
        _call(sample_user, db_session_factory, action="add", symbol="PTT.BK")
        result = _call(
            sample_user, db_session_factory, action="set_alert", symbol="ptt.bk", alert_time="20:00"
        )
        assert result == {"action": "set_alert", "symbol": "PTT.BK", "alert_time": "20:00"}
        cleared = _call(sample_user, db_session_factory, action="clear_alert", symbol="PTT.BK")
        assert cleared["status"] == "cleared"
        again = _call(sample_user, db_session_factory, action="clear_alert", symbol="PTT.BK")
        assert again["status"] == "no_alert"

    def test_clear_alert_requires_symbol(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """clear_alert without a symbol is an error."""
        assert "error" in _call(sample_user, db_session_factory, action="clear_alert")

    def test_remove_turns_off_alert(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Removing the asset also clears its alert."""
        _call(sample_user, db_session_factory, action="add", symbol="PTT.BK", alert_time="07:00")
        _call(sample_user, db_session_factory, action="remove", symbol="PTT.BK")
        _call(sample_user, db_session_factory, action="add", symbol="PTT.BK")
        listed = _call(sample_user, db_session_factory, action="list")
        assert listed["watchlist"][0]["alert_time"] is None  # type: ignore[index]


class TestWatchlistInputEdgeCases:
    """Symbols and times typed the way people type them."""

    def test_remove_without_bk_suffix(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """'ptt' removes PTT.BK."""
        _call(sample_user, db_session_factory, action="add", symbol="PTT.BK")
        result = _call(sample_user, db_session_factory, action="remove", symbol="ptt")
        assert result == {"action": "remove", "symbol": "PTT.BK", "status": "removed"}

    def test_set_alert_without_bk_and_with_pm(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """'ptt' + '8:30 pm' sets PTT.BK at 20:30."""
        _call(sample_user, db_session_factory, action="add", symbol="PTT.BK")
        result = _call(
            sample_user, db_session_factory, action="set_alert", symbol="ptt", alert_time="8:30 pm"
        )
        assert result == {"action": "set_alert", "symbol": "PTT.BK", "alert_time": "20:30"}

    def test_set_alert_thai_time_suffix(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """'20.00 น.' is accepted."""
        _call(sample_user, db_session_factory, action="add", symbol="PTT.BK")
        result = _call(
            sample_user, db_session_factory, action="set_alert", symbol="PTT", alert_time="20.00 น."
        )
        assert result["alert_time"] == "20:00"

    def test_unknown_asset_offers_watched_options(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """An asset not on the list asks which watched one was meant."""
        _call(sample_user, db_session_factory, action="add", symbol="PTT.BK")
        result = _call(
            sample_user, db_session_factory, action="set_alert", symbol="AOT", alert_time="08:00"
        )
        assert result["needs_clarification"] is True
        assert result["options"] == ["PTT.BK"]

    def test_missing_symbol_for_set_alert_asks(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """No symbol: ask which watched asset, listing them."""
        _call(sample_user, db_session_factory, action="add", symbol="PTT.BK")
        result = _call(sample_user, db_session_factory, action="set_alert", alert_time="08:00")
        assert result["options"] == ["PTT.BK"]

    def test_ambiguous_time_asks(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """A bare hour like '8' is ambiguous: ask for a 24-hour time."""
        _call(sample_user, db_session_factory, action="add", symbol="PTT.BK")
        result = _call(
            sample_user, db_session_factory, action="set_alert", symbol="PTT", alert_time="8"
        )
        assert result["needs_clarification"] is True
        assert "24" in str(result["ask_user"])

    def test_add_bad_time_passes_question_on(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Add succeeds; the unclear time comes back as a question."""
        result = _call(
            sample_user, db_session_factory, action="add", symbol="PTT.BK", alert_time="8"
        )
        assert result["status"] == "added"
        assert result["needs_clarification"] is True

    def test_add_empty_symbol_asks(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Blank symbol asks which asset to add."""
        result = _call(sample_user, db_session_factory, action="add", symbol="  ")
        assert result["needs_clarification"] is True

    def test_add_ambiguous_symbol_offers_candidates(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Several close matches become options."""
        monkeypatch.setattr(
            "finance_ai.tools.symbol_guard.search_asset_symbols",
            lambda query: [
                AssetSymbolMatch(
                    symbol="PTTEP.BK", name="PTTEP", exchange="SET", quote_type="EQUITY"
                ),
                AssetSymbolMatch(
                    symbol="PTTGC.BK", name="PTTGC", exchange="SET", quote_type="EQUITY"
                ),
            ],
        )
        result = _call(sample_user, db_session_factory, action="add", symbol="ปตท")
        assert result["options"] == ["PTTEP.BK — PTTEP", "PTTGC.BK — PTTGC"]

    def test_search_outage_says_retry(
        self,
        sample_user: User,
        db_session_factory: Callable[[], Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A search failure is reported as retry-later, not as a question."""

        def broken(query: str) -> list[AssetSymbolMatch]:
            raise ConnectionError("offline")

        monkeypatch.setattr("finance_ai.tools.symbol_guard.search_asset_symbols", broken)
        result = _call(sample_user, db_session_factory, action="add", symbol="PTT")
        assert result["retry_later"] is True

    def test_remove_from_empty_watchlist(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Nothing to choose from: plain not-found, no options."""
        result = _call(sample_user, db_session_factory, action="remove", symbol="PTT")
        assert result["status"] == "not_found"
        assert "options" not in result


class TestBareHourReply:
    """A reply of just "8" is asked about, whatever time the model chose."""

    @staticmethod
    def _with_reply(
        user: User, factory: Callable[[], Session], reply: str, **arguments: str
    ) -> dict[str, object]:
        """Invoke with a conversation whose last user message is `reply`."""
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [HumanMessage("เพิ่ม PTT"), AIMessage("แจ้งกี่โมงดี"), HumanMessage(reply)]
        payload: dict[str, object] = {
            "user_id": user.id,
            "db_session_factory": factory,
            "messages": messages,
            **arguments,
        }
        result: dict[str, object] = manage_watchlist.invoke(payload)
        return result

    def test_bare_hour_asks_morning_or_evening(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """The model guessed 08:00 from '8'; the tool asks instead of saving."""
        _call(sample_user, db_session_factory, action="add", symbol="PTT.BK")
        result = self._with_reply(
            sample_user,
            db_session_factory,
            "8",
            action="set_alert",
            symbol="PTT.BK",
            alert_time="08:00",
        )
        assert result["needs_clarification"] is True
        assert result["options"] == ["08:00 (8 โมงเช้า)", "20:00 (2 ทุ่ม)"]
        listed = _call(sample_user, db_session_factory, action="list")
        assert listed["watchlist"][0]["alert_time"] is None  # type: ignore[index]

    def test_clear_reply_is_saved(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """'2 ทุ่ม' is clear, so the model's 20:00 is saved."""
        _call(sample_user, db_session_factory, action="add", symbol="PTT.BK")
        result = self._with_reply(
            sample_user,
            db_session_factory,
            "2 ทุ่ม",
            action="set_alert",
            symbol="PTT.BK",
            alert_time="20:00",
        )
        assert result["alert_time"] == "20:00"

    def test_add_with_bare_hour_adds_but_asks(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """For 'add', the asset is added and only the time waits for an answer."""
        result = self._with_reply(
            sample_user,
            db_session_factory,
            "๑๒ ครับ",
            action="add",
            symbol="PTT.BK",
            alert_time="12:00",
        )
        assert result["status"] == "added"
        assert result["options"] == ["12:00 (เที่ยง)", "00:00 (เที่ยงคืน)"]
        assert "next_step" not in result

    def test_no_conversation_trusts_the_time(
        self, sample_user: User, db_session_factory: Callable[[], Session]
    ) -> None:
        """Without messages (e.g. direct calls), the given time is used."""
        _call(sample_user, db_session_factory, action="add", symbol="PTT.BK")
        result = _call(
            sample_user, db_session_factory, action="set_alert", symbol="PTT", alert_time="08:00"
        )
        assert result["alert_time"] == "08:00"
