"""Tests for the /assets endpoints.

Covers:
- GET /assets/notifications (content column bug + Thai date formatting)
- GET /assets/search (free-text symbol search)
- POST /assets/fetch (structured result, no notification side-effect)
- GET/POST/DELETE /assets/watchlist (canonicalization + validation guard)
"""

# pylint: disable=redefined-outer-name,unsubscriptable-object,too-many-lines

# pylint: disable=redefined-outer-name

from datetime import datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from finance_ai.database.base import Base
from finance_ai.database.models.asset_notification import AssetNotification
from finance_ai.database.models.user import User
from finance_ai.main import app
from finance_ai.tools.market_data_models import AssetSymbolMatch, NewsItem


@pytest.fixture
def client() -> TestClient:
    """Return a TestClient for the app."""
    return TestClient(app)


@pytest.fixture
def temp_db(monkeypatch: pytest.MonkeyPatch) -> sessionmaker[Session]:
    """Create an in-memory DB, patch main._session_factory, and add a user.

    Returns:
        sessionmaker bound to the in-memory engine (with schema + one user).
    """
    engine: Engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine)

    session = factory()
    user = User(
        email="t@example.com",
        hashed_password="x",
        full_name="T",
        tax_id="1234567890123",
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    session.close()

    monkeypatch.setattr("finance_ai.main._session_factory", factory)
    return factory


class TestAssetNotificationsField:
    """Tests for GET /assets/notifications (content column bug)."""

    def test_returns_content_not_message(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
    ) -> None:
        """The endpoint returns the notification's content text without crashing."""
        session = temp_db()
        user = session.query(User).first()
        assert user is not None
        notification = AssetNotification(
            user_id=user.id,
            symbol="GC=F",
            content="ราคาทอง: 2,350 USD",
            is_read=False,
        )
        session.add(notification)
        session.commit()
        user_id = user.id
        session.close()

        response = client.get("/assets/notifications", params={"user_id": user_id})

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["symbol"] == "GC=F"
        assert body[0]["message"] == "ราคาทอง: 2,350 USD"


class TestAssetSearchEndpoint:
    """Tests for GET /assets/search (free-text symbol search)."""

    def test_returns_mapped_results(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Search results are mapped to symbol/name/exchange/type dicts."""
        matches = [
            AssetSymbolMatch(
                symbol="PTT.BK",
                name="PTT Public Company Limited",
                exchange="SET",
                quote_type="EQUITY",
            ),
            AssetSymbolMatch(
                symbol="AAPL",
                name="Apple Inc.",
                exchange="NASDAQ",
                quote_type="EQUITY",
            ),
        ]

        def fake_search(query: str) -> list[AssetSymbolMatch]:
            """Return the canned matches after asserting the raw query."""
            assert query == "PTT"
            return matches

        monkeypatch.setattr(
            "finance_ai.tools.symbol_search_service.search_asset_symbols", fake_search
        )

        response = client.get("/assets/search", params={"query": "PTT"})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["results"] == [
            {
                "symbol": "PTT.BK",
                "name": "PTT Public Company Limited",
                "exchange": "SET",
                "type": "EQUITY",
                "category": "thai_stock",
            },
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "type": "EQUITY",
                "category": "foreign_stock",
            },
        ]

    def test_whitespace_query_returns_422(self, client: TestClient) -> None:
        """Empty/whitespace-only queries are rejected with 422 and a clear message."""
        response = client.get("/assets/search", params={"query": "   "})

        assert response.status_code == 422
        assert "คำค้นหา" in response.json()["detail"]


def _first_user_id(factory: sessionmaker[Session]) -> str:
    """Return the id of the seeded user in the temp database.

    Args:
        factory: Session factory bound to the in-memory test engine.

    Returns:
        UUID string of the seeded test user.
    """
    session = factory()
    user = session.query(User).first()
    assert user is not None
    user_id = user.id
    session.close()
    return user_id


def _stub_fetch_mocks(monkeypatch: pytest.MonkeyPatch, price: Decimal | None) -> None:
    """Stub price, currency, and news lookups for /assets/fetch tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        price: Price returned by the mocked fetch_current_price.
    """
    monkeypatch.setattr("finance_ai.tools.price_client.fetch_current_price", lambda symbol: price)
    monkeypatch.setattr("finance_ai.tools.price_client.fetch_currency", lambda symbol: "USD")
    monkeypatch.setattr(
        "finance_ai.tools.market_data_service.fetch_news_items",
        lambda symbol: [
            NewsItem(
                title="Apple unveils new iPhone",
                source="Reuters",
                link="https://example.com/aapl",
                published_at="2026-09-13T10:45:47+00:00",
            )
        ],
    )


class TestAssetFetchEndpoint:
    """Tests for POST /assets/fetch (structured result, no notifications)."""

    def test_returns_structured_price_and_news(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A successful all-fetch returns symbol/price/currency/news with no error."""
        user_id = _first_user_id(temp_db)
        _stub_fetch_mocks(monkeypatch, Decimal("254.30"))

        response = client.post(
            "/assets/fetch",
            json={"user_id": user_id, "symbol": "aapl", "fetch_type": "all"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        result = body["result"]
        assert result["symbol"] == "AAPL"
        assert result["price"] == "254.30"
        assert result["currency"] == "USD"
        assert result["news"] == [
            {
                "title": "Apple unveils new iPhone",
                "source": "Reuters",
                "source_url": None,
                "link": "https://example.com/aapl",
                "published_at": "2026-09-13T10:45:47+00:00",
            }
        ]
        assert result["error"] is None

    def test_no_notification_created(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The immediate fetch endpoint must not create an AssetNotification row."""
        user_id = _first_user_id(temp_db)
        _stub_fetch_mocks(monkeypatch, Decimal("254.30"))

        client.post(
            "/assets/fetch",
            json={"user_id": user_id, "symbol": "aapl", "fetch_type": "all"},
        )

        response = client.get("/assets/notifications", params={"user_id": user_id})
        assert response.status_code == 200
        assert response.json() == []

    def test_news_only_fetch_has_no_price(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """fetch_type=news returns news with null price/currency and no error."""
        user_id = _first_user_id(temp_db)
        _stub_fetch_mocks(monkeypatch, Decimal("254.30"))

        response = client.post(
            "/assets/fetch",
            json={"user_id": user_id, "symbol": "AAPL", "fetch_type": "news"},
        )

        assert response.status_code == 200
        result = response.json()["result"]
        assert result["price"] is None
        assert result["currency"] is None
        assert len(result["news"]) == 1
        assert result["error"] is None

    def test_unavailable_data_sets_thai_error(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When price and news are unavailable the Thai error message is set."""
        user_id = _first_user_id(temp_db)
        monkeypatch.setattr(
            "finance_ai.tools.price_client.fetch_current_price", lambda symbol: None
        )
        monkeypatch.setattr("finance_ai.tools.price_client.fetch_currency", lambda symbol: "USD")
        monkeypatch.setattr(
            "finance_ai.tools.market_data_service.fetch_news_items", lambda symbol: []
        )

        response = client.post(
            "/assets/fetch",
            json={"user_id": user_id, "symbol": "AAPL", "fetch_type": "all"},
        )

        assert response.status_code == 200
        result = response.json()["result"]
        assert result["price"] is None
        assert "ไม่สามารถดึงราคา" in result["error"]
        assert "ไม่พบข่าว" in result["error"]

    def test_invalid_fetch_type_returns_422(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
    ) -> None:
        """fetch_type outside price/news/all is rejected with 422."""
        user_id = _first_user_id(temp_db)

        response = client.post(
            "/assets/fetch",
            json={"user_id": user_id, "symbol": "AAPL", "fetch_type": "bogus"},
        )

        assert response.status_code == 422


class TestWatchlistEndpoints:
    """Tests for GET/POST/DELETE /assets/watchlist."""

    def test_get_returns_empty_list(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """GET returns an empty list for a user with no watched assets."""
        user_id = _first_user_id(temp_db)

        response = client.get("/assets/watchlist", params={"user_id": user_id})

        assert response.status_code == 200
        assert response.json() == []

    def test_post_canonicalizes_and_stores(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """POST ptt stores the canonical PTT.BK symbol and GET shows it."""
        user_id = _first_user_id(temp_db)
        monkeypatch.setattr(
            "finance_ai.tools.symbol_guard.search_asset_symbols",
            lambda query: [
                AssetSymbolMatch(
                    symbol="PTT.BK",
                    name="PTT Public Company Limited",
                    exchange="SET",
                    quote_type="EQUITY",
                )
            ],
        )

        response = client.post(
            "/assets/watchlist",
            json={"user_id": user_id, "symbol": "ptt", "name": "PTT"},
        )

        assert response.status_code == 200
        assert response.json() == {"status": "added", "symbol": "PTT.BK", "name": "PTT"}
        listed = client.get("/assets/watchlist", params={"user_id": user_id})
        assert len(listed.json()) == 1
        row = listed.json()[0]
        assert row["symbol"] == "PTT.BK"
        assert row["name"] == "PTT"
        assert row["id"]

    def test_post_duplicate_after_canonicalization(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Adding ptt then PTT.BK returns already_exists (idempotent 200)."""
        user_id = _first_user_id(temp_db)
        monkeypatch.setattr(
            "finance_ai.tools.symbol_guard.search_asset_symbols",
            lambda query: [
                AssetSymbolMatch(
                    symbol="PTT.BK",
                    name="PTT Public Company Limited",
                    exchange="SET",
                    quote_type="EQUITY",
                )
            ],
        )

        first = client.post(
            "/assets/watchlist", json={"user_id": user_id, "symbol": "ptt", "name": "PTT"}
        )
        second = client.post(
            "/assets/watchlist",
            json={"user_id": user_id, "symbol": "PTT.BK", "name": "PTT"},
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["status"] == "already_exists"
        assert second.json()["symbol"] == "PTT.BK"

    def test_post_unresolvable_symbol_returns_422(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unresolvable symbol is rejected with an actionable Thai error."""
        user_id = _first_user_id(temp_db)
        monkeypatch.setattr("finance_ai.tools.symbol_guard.search_asset_symbols", lambda query: [])

        response = client.post(
            "/assets/watchlist",
            json={"user_id": user_id, "symbol": "APPL", "name": "Apple"},
        )

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert "APPL" in detail
        assert "ไม่พบสัญลักษณ์" in detail

    def test_post_single_strong_match(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """appl with a lone AAPL candidate resolves and stores AAPL."""
        user_id = _first_user_id(temp_db)
        monkeypatch.setattr(
            "finance_ai.tools.symbol_guard.search_asset_symbols",
            lambda query: [
                AssetSymbolMatch(
                    symbol="AAPL",
                    name="Apple Inc.",
                    exchange="NASDAQ",
                    quote_type="EQUITY",
                )
            ],
        )

        response = client.post(
            "/assets/watchlist", json={"user_id": user_id, "symbol": "appl", "name": "Apple"}
        )

        assert response.status_code == 200
        assert response.json()["symbol"] == "AAPL"

    def test_delete_removes_asset(
        self,
        client: TestClient,
        temp_db: sessionmaker[Session],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """DELETE by symbol removes the asset and returns {"status": "ok"}."""
        user_id = _first_user_id(temp_db)
        monkeypatch.setattr(
            "finance_ai.tools.symbol_guard.search_asset_symbols",
            lambda query: [
                AssetSymbolMatch(
                    symbol="PTT.BK",
                    name="PTT Public Company Limited",
                    exchange="SET",
                    quote_type="EQUITY",
                )
            ],
        )
        client.post(
            "/assets/watchlist", json={"user_id": user_id, "symbol": "PTT.BK", "name": "PTT"}
        )

        response = client.delete("/assets/watchlist/PTT.BK", params={"user_id": user_id})

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        listed = client.get("/assets/watchlist", params={"user_id": user_id})
        assert listed.json() == []

    def test_delete_unknown_returns_404(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """DELETE for an asset that does not exist returns 404."""
        user_id = _first_user_id(temp_db)

        response = client.delete("/assets/watchlist/nope", params={"user_id": user_id})

        assert response.status_code == 404


class TestNotificationsThaiDate:
    """Tests for Thai-formatted created_at in GET /assets/notifications."""

    def test_created_at_is_thai_formatted(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """created_at (stored UTC) renders in Bangkok time, e.g. '13 ก.ย. 2026, 17:45'."""
        session = temp_db()
        user = session.query(User).first()
        assert user is not None
        notification = AssetNotification(
            user_id=user.id,
            symbol="GC=F",
            content="ราคาทอง: 2,350 USD",
            is_read=False,
            created_at=datetime(2026, 9, 13, 10, 45),
        )
        session.add(notification)
        session.commit()
        user_id = user.id
        session.close()

        response = client.get("/assets/notifications", params={"user_id": user_id})

        assert response.status_code == 200
        assert response.json()[0]["created_at"] == "13 ก.ย. 2026, 17:45"


class TestHoldingsEndpoint:
    """GET /assets/holdings lists what the user owns (shown beside the watchlist)."""

    def test_empty(self, client: TestClient, temp_db: sessionmaker[Session]) -> None:
        """A user without holdings gets an empty list."""
        response = client.get("/assets/holdings", params={"user_id": _first_user_id(temp_db)})
        assert response.status_code == 200
        assert response.json() == []

    def test_lists_holdings(self, client: TestClient, temp_db: sessionmaker[Session]) -> None:
        """Each holding has symbol, quantity, value and gain/loss (money as strings)."""
        from datetime import date  # pylint: disable=import-outside-toplevel

        from finance_ai.database.models.investment_holding import (  # pylint: disable=import-outside-toplevel
            InvestmentHolding,
        )

        user_id = _first_user_id(temp_db)
        with temp_db() as session:
            session.add(
                InvestmentHolding(
                    user_id=user_id,
                    asset_type="stock",
                    symbol="PTT.BK",
                    name="PTT",
                    quantity=Decimal("100"),
                    average_cost_per_unit=Decimal("35.50"),
                    total_cost=Decimal("3550.00"),
                    purchase_date=date(2025, 1, 15),
                    current_price_per_unit=Decimal("42"),
                    current_value=Decimal("4200"),
                    unrealized_gain_loss=Decimal("650"),
                )
            )
            session.commit()
        rows = client.get("/assets/holdings", params={"user_id": user_id}).json()
        assert rows[0].pop("holding_id")  # each row carries its id for edit/delete
        assert rows == [
            {
                "symbol": "PTT.BK",
                "name": "PTT",
                "quantity": "100",
                "average_cost_per_unit": "35.50",
                "current_price_per_unit": "42.00",
                "current_value": "4200.00",
                "unrealized_gain_loss": "650.00",
                "gain_loss_percentage": "18.3099",
            }
        ]


def _watch_ptt(client: TestClient, user_id: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """Put PTT.BK on the user's watchlist; return its asset id."""
    monkeypatch.setattr(
        "finance_ai.tools.symbol_guard.search_asset_symbols",
        lambda query: [
            AssetSymbolMatch(symbol="PTT.BK", name="PTT", exchange="SET", quote_type="EQUITY")
        ],
    )
    client.post("/assets/watchlist", json={"user_id": user_id, "symbol": "PTT.BK", "name": "PTT"})
    listed = client.get("/assets/watchlist", params={"user_id": user_id}).json()
    return str(listed[0]["id"])


class TestWatchlistAlertEndpoint:
    """PUT /assets/watchlist/{id}/alert sets or clears the daily news time."""

    def test_set_alert_shows_in_list(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A set time comes back normalized and in GET /assets/watchlist."""
        user_id = _first_user_id(temp_db)
        asset_id = _watch_ptt(client, user_id, monkeypatch)

        response = client.put(
            f"/assets/watchlist/{asset_id}/alert",
            json={"user_id": user_id, "alert_time": "8.30"},
        )

        assert response.json() == {"symbol": "PTT.BK", "alert_time": "08:30"}
        listed = client.get("/assets/watchlist", params={"user_id": user_id}).json()
        assert listed[0]["alert_time"] == "08:30"

    def test_null_clears_alert(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """alert_time null turns the alert off."""
        user_id = _first_user_id(temp_db)
        _watch_ptt(client, user_id, monkeypatch)
        url = "/assets/watchlist/PTT.BK/alert"
        client.put(url, json={"user_id": user_id, "alert_time": "08:30"})

        response = client.put(url, json={"user_id": user_id, "alert_time": None})

        assert response.json() == {"symbol": "PTT.BK", "alert_time": None}
        listed = client.get("/assets/watchlist", params={"user_id": user_id}).json()
        assert listed[0]["alert_time"] is None

    def test_invalid_time_returns_422(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A time outside 00:00-23:59 is rejected."""
        user_id = _first_user_id(temp_db)
        _watch_ptt(client, user_id, monkeypatch)

        response = client.put(
            "/assets/watchlist/PTT.BK/alert", json={"user_id": user_id, "alert_time": "25:00"}
        )

        assert response.status_code == 422
        assert "24-hour" in response.json()["detail"]

    def test_unwatched_asset_returns_404(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """Only watched assets can get an alert."""
        user_id = _first_user_id(temp_db)

        response = client.put(
            "/assets/watchlist/AOT.BK/alert", json={"user_id": user_id, "alert_time": "08:00"}
        )

        assert response.status_code == 404

    def test_delete_by_id_turns_off_alert(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting by UUID removes the asset and its alert."""
        user_id = _first_user_id(temp_db)
        asset_id = _watch_ptt(client, user_id, monkeypatch)
        client.put(
            f"/assets/watchlist/{asset_id}/alert",
            json={"user_id": user_id, "alert_time": "08:30"},
        )

        response = client.delete(f"/assets/watchlist/{asset_id}", params={"user_id": user_id})

        assert response.status_code == 200
        _watch_ptt(client, user_id, monkeypatch)
        listed = client.get("/assets/watchlist", params={"user_id": user_id}).json()
        assert listed[0]["alert_time"] is None


class TestWatchlistSymbolTypedAnyWay:
    """The web API finds watched Thai stocks without the .BK suffix."""

    def test_alert_and_delete_by_plain_ticker(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'ptt' works for both the alert and the delete endpoint."""
        user_id = _first_user_id(temp_db)
        _watch_ptt(client, user_id, monkeypatch)

        alert = client.put(
            "/assets/watchlist/ptt/alert", json={"user_id": user_id, "alert_time": "8:30 pm"}
        )
        deleted = client.delete("/assets/watchlist/ptt", params={"user_id": user_id})

        assert alert.json() == {"symbol": "PTT.BK", "alert_time": "20:30"}
        assert deleted.status_code == 200


class TestAddHoldingEndpoint:
    """POST /assets/holdings (asset page form) and price refresh."""

    def test_blank_price_uses_latest(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Saved at the latest price and listed with its gain."""
        user_id = _first_user_id(temp_db)
        monkeypatch.setattr(
            "finance_ai.tools.holding_entry_service.fetch_current_price", lambda s: Decimal("35")
        )

        response = client.post(
            "/assets/holdings",
            json={"user_id": user_id, "symbol": "PTT.BK", "name": "PTT", "quantity": "100"},
        )

        assert response.status_code == 200
        assert response.json()["used_latest_price"] is True
        assert response.json()["total_cost"] == "3500.00"
        listed = client.get("/assets/holdings", params={"user_id": user_id}).json()
        assert listed[0]["gain_loss_percentage"] is not None

    def test_foreign_stock_returns_422(
        self, client: TestClient, temp_db: sessionmaker[Session]
    ) -> None:
        """Only Thai stocks can be added from the asset page."""
        user_id = _first_user_id(temp_db)
        response = client.post(
            "/assets/holdings",
            json={"user_id": user_id, "symbol": "AAPL", "quantity": "1", "price_per_unit": "200"},
        )
        assert response.status_code == 422
        assert "หุ้นไทย" in response.json()["detail"]

    def test_refresh_updates_gain(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Refresh stores the latest price and returns the new gain."""
        user_id = _first_user_id(temp_db)
        monkeypatch.setattr(
            "finance_ai.tools.holding_entry_service.fetch_current_price", lambda s: None
        )
        client.post(
            "/assets/holdings",
            json={"user_id": user_id, "symbol": "PTT.BK", "quantity": "10", "price_per_unit": "30"},
        )
        monkeypatch.setattr(
            "finance_ai.tools.price_client.fetch_multiple_prices",
            lambda symbols: {"PTT.BK": Decimal("33")},
        )

        rows = client.post("/assets/holdings/refresh", params={"user_id": user_id}).json()

        assert rows[0]["unrealized_gain_loss"] == "30.00"
        assert rows[0]["current_price_per_unit"] == "33.00"


class TestFundsOnAssetPage:
    """Thai mutual funds from SEC data on the asset page."""

    def test_search_includes_fund_classes(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fund classes come back as type FUND with Thai name, house and NAV."""
        from datetime import date

        from finance_ai.tools.fund_search_service import FundSearchOutcome, FundSearchResult

        monkeypatch.setattr(
            "finance_ai.tools.symbol_search_service.search_asset_symbols", lambda query: []
        )
        fund = FundSearchResult(
            symbol="K-USA-A(A)",
            proj_id="M1",
            fund_code="K-USA",
            name_th="กองทุนเปิดเค ยูเอสเอ หุ้นทุน",
            name_en="K USA Equity",
            amc="กสิกรไทย",
            payout="A",
            nav=Decimal("16.6925"),
            nav_date=date(2026, 10, 29),
        )
        monkeypatch.setattr(
            "finance_ai.tools.asset_search_service.search_funds",
            lambda query, expand=None: FundSearchOutcome(results=[fund]),
        )

        body = client.get("/assets/search", params={"query": "k-usa"}).json()

        assert body["results"] == [
            {
                "symbol": "K-USA-A(A)",
                "name": "กองทุนเปิดเค ยูเอสเอ หุ้นทุน",
                "name_en": "K USA Equity",
                "exchange": "SEC",
                "type": "FUND",
                "category": "fund",
                "fund_code": "K-USA",
                "amc": "กสิกรไทย",
                "payout": "A",
                "nav": "16.6925",
                "nav_date": "2026-10-29",
            }
        ]
        assert body["clarify"] is None
        assert body["expanded_terms"] == []

    def test_add_fund_with_date(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fund bought in the past at a typed NAV shows its gain at today's NAV."""
        user_id = _first_user_id(temp_db)
        monkeypatch.setattr(
            "finance_ai.tools.holding_entry_service.fetch_latest_nav", lambda s: Decimal("16")
        )

        response = client.post(
            "/assets/holdings",
            json={
                "user_id": user_id,
                "symbol": "K-USA-A(A)",
                "quantity": "100",
                "price_per_unit": "15",
                "purchase_date": "1/10/2569",
                "asset_type": "mutual_fund",
            },
        )

        body = response.json()
        assert response.status_code == 200
        assert (body["purchase_date"], body["gain_loss"]) == ("2026-10-01", "100.00")

    def test_refresh_prices_funds_from_sec(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Price refresh asks the SEC for funds and Yahoo for stocks."""
        user_id = _first_user_id(temp_db)
        monkeypatch.setattr(
            "finance_ai.tools.holding_entry_service.fetch_latest_nav", lambda s: None
        )
        client.post(
            "/assets/holdings",
            json={
                "user_id": user_id,
                "symbol": "SCBSET50",
                "quantity": "10",
                "price_per_unit": "20",
                "asset_type": "mutual_fund",
            },
        )
        yahoo_symbols: list[list[str]] = []

        def fake_yahoo(symbols: list[str]) -> dict[str, Decimal]:
            yahoo_symbols.append(symbols)
            return {}

        monkeypatch.setattr("finance_ai.tools.price_client.fetch_multiple_prices", fake_yahoo)
        monkeypatch.setattr(
            "finance_ai.tools.fund_nav_client.fetch_latest_nav", lambda s: Decimal("24")
        )

        rows = client.post("/assets/holdings/refresh", params={"user_id": user_id}).json()

        assert rows[0]["unrealized_gain_loss"] == "40.00"
        assert not yahoo_symbols


class TestEditDeleteHoldingEndpoints:
    """PATCH/DELETE /assets/holdings/{holding_id} (edit or remove a portfolio row)."""

    @staticmethod
    def _add_ptt(client: TestClient, user_id: str, monkeypatch: pytest.MonkeyPatch) -> str:
        """Add 500 PTT at 42 and return its holding id."""
        monkeypatch.setattr(
            "finance_ai.tools.holding_entry_service.fetch_current_price", lambda s: Decimal("42")
        )
        client.post(
            "/assets/holdings",
            json={"user_id": user_id, "symbol": "PTT.BK", "name": "PTT_PTT", "quantity": "500"},
        )
        return str(
            client.get("/assets/holdings", params={"user_id": user_id}).json()[0]["holding_id"]
        )

    def test_edit_quantity_and_cost(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The row shows the new quantity and cost after the edit."""
        user_id = _first_user_id(temp_db)
        holding_id = self._add_ptt(client, user_id, monkeypatch)

        response = client.patch(
            f"/assets/holdings/{holding_id}",
            json={"user_id": user_id, "quantity": "600", "price_per_unit": "40"},
        )

        assert response.status_code == 204
        [row] = client.get("/assets/holdings", params={"user_id": user_id}).json()
        assert (row["quantity"], row["average_cost_per_unit"]) == ("600", "40.00")

    def test_edit_with_bad_number_is_422(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Thai message explains what to fix."""
        user_id = _first_user_id(temp_db)
        holding_id = self._add_ptt(client, user_id, monkeypatch)

        response = client.patch(
            f"/assets/holdings/{holding_id}",
            json={"user_id": user_id, "quantity": "0", "price_per_unit": "40"},
        )

        assert response.status_code == 422
        assert "จำนวนหุ้น/หน่วย" in response.json()["detail"]

    def test_delete_the_only_holding(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting the last row leaves an empty portfolio (the reported bug)."""
        user_id = _first_user_id(temp_db)
        holding_id = self._add_ptt(client, user_id, monkeypatch)

        response = client.delete(f"/assets/holdings/{holding_id}", params={"user_id": user_id})

        assert response.status_code == 204
        assert client.get("/assets/holdings", params={"user_id": user_id}).json() == []

    def test_unknown_or_foreign_holding_is_404(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Another user's id cannot edit or delete the row."""
        user_id = _first_user_id(temp_db)
        holding_id = self._add_ptt(client, user_id, monkeypatch)

        edit = client.patch(
            f"/assets/holdings/{holding_id}",
            json={"user_id": "someone-else", "quantity": "1", "price_per_unit": "1"},
        )
        delete = client.delete(f"/assets/holdings/{holding_id}", params={"user_id": "someone-else"})

        assert (edit.status_code, delete.status_code) == (404, 404)

    def test_yahoo_ticker_prefix_is_hidden(
        self, client: TestClient, temp_db: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A name saved as 'PTT_PTT' (Yahoo's SET format) is listed as 'PTT'."""
        user_id = _first_user_id(temp_db)
        self._add_ptt(client, user_id, monkeypatch)

        [row] = client.get("/assets/holdings", params={"user_id": user_id}).json()

        assert row["name"] == "PTT"
