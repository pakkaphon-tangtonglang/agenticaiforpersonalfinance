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
            },
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "type": "EQUITY",
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
        """created_at renders as e.g. '13 ก.ย. 2026, 17:45'."""
        session = temp_db()
        user = session.query(User).first()
        assert user is not None
        notification = AssetNotification(
            user_id=user.id,
            symbol="GC=F",
            content="ราคาทอง: 2,350 USD",
            is_read=False,
            created_at=datetime(2026, 9, 13, 17, 45),
        )
        session.add(notification)
        session.commit()
        user_id = user.id
        session.close()

        response = client.get("/assets/notifications", params={"user_id": user_id})

        assert response.status_code == 200
        assert response.json()[0]["created_at"] == "13 ก.ย. 2026, 17:45"
