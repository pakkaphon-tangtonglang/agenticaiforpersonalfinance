"""Tests for sec_fund_api and fund_nav_client (SEC Fund API V2)."""

from collections.abc import Callable
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import httpx

from finance_ai.tools import sec_fund_api
from finance_ai.tools.fund_nav_client import (
    fetch_latest_nav,
    is_fund_data_available,
    latest_navs_for_project,
)

FakeSec = Callable[[Callable[[httpx.Request], httpx.Response]], list[httpx.Request]]


def page(items: list[dict[str, Any]], cursor: str | None = None) -> httpx.Response:
    """One JSON page in the SEC response shape."""
    return httpx.Response(200, json={"message": "success", "items": items, "next_cursor": cursor})


def nav_row(name: str, nav_date: str, nav: Any, proj_id: str = "M1") -> dict[str, Any]:
    """One NAV row."""
    return {"proj_id": proj_id, "fund_class_name": name, "nav_date": nav_date, "last_val": nav}


class TestFetchItems:
    """sec_fund_api.fetch_items: key, paging, 204, errors, cache."""

    def test_sends_key_and_page_size(self, fake_sec_api: FakeSec) -> None:
        """The key header and page_size are sent with the filters."""
        calls = fake_sec_api(lambda request: page([{"a": 1}]))
        assert sec_fund_api.fetch_items(sec_fund_api.AMCS_PATH, {"x": "y"}) == [{"a": 1}]
        assert calls[0].headers["Ocp-Apim-Subscription-Key"] == "test-key"
        assert calls[0].url.params["page_size"] == "100"
        assert calls[0].url.path == sec_fund_api.AMCS_PATH

    def test_follows_cursor_up_to_max_pages(self, fake_sec_api: FakeSec) -> None:
        """Pages are read until next_cursor is empty or max_pages is reached."""
        calls = fake_sec_api(lambda request: page([{"n": len(calls)}], cursor="more"))
        items = sec_fund_api.fetch_items(sec_fund_api.NAV_PATH, {}, max_pages=3)
        assert len(items) == 3
        assert calls[1].url.params["next_cursor"] == "more"

    def test_204_is_empty_and_cached(self, fake_sec_api: FakeSec) -> None:
        """No match (204) is a cached empty answer, not an error."""
        calls = fake_sec_api(lambda request: httpx.Response(204))
        assert not sec_fund_api.fetch_items(sec_fund_api.NAV_PATH, {"q": "x"})
        assert not sec_fund_api.fetch_items(sec_fund_api.NAV_PATH, {"q": "x"})
        assert len(calls) == 1

    def test_error_is_empty_and_not_cached(self, fake_sec_api: FakeSec) -> None:
        """A failure returns [] and is retried next time."""
        calls = fake_sec_api(lambda request: httpx.Response(500))
        assert not sec_fund_api.fetch_items(sec_fund_api.NAV_PATH, {"q": "x"})
        assert not sec_fund_api.fetch_items(sec_fund_api.NAV_PATH, {"q": "x"})
        assert len(calls) == 2

    def test_no_key_no_request(self) -> None:
        """Without a key nothing is requested (conftest removes the key)."""
        with patch.object(httpx, "Client") as client:
            assert not sec_fund_api.fetch_items(sec_fund_api.AMCS_PATH, {})
            assert is_fund_data_available() is False
        client.assert_not_called()


class TestLatestNav:
    """fund_nav_client: newest NAV per class, single-class funds."""

    def test_newest_row_per_class(self, fake_sec_api: FakeSec) -> None:
        """Older rows and rows without a positive NAV are ignored."""
        fake_sec_api(
            lambda request: page(
                [
                    nav_row("K-VALUE-A(A)", "2026-10-29", 7.1),
                    nav_row("K-VALUE-A(A)", "2026-10-30", 7.3609),
                    nav_row("K-VALUE-P(A)", "2026-10-30", 0),
                    nav_row("K-VALUE-Z(A)", "bad-date", 7),
                    {"fund_class_name": "BROKEN"},
                ]
            )
        )
        navs = latest_navs_for_project("M0023_2538")
        assert {name: quote.nav for name, quote in navs.items()} == {
            "K-VALUE-A(A)": Decimal("7.3609")
        }

    def test_class_code_lookup(self, fake_sec_api: FakeSec) -> None:
        """'SCBSET50' does not take SCBSET50E's NAV."""
        fake_sec_api(
            lambda request: page(
                [
                    nav_row("SCBSET50E", "2026-10-30", 25.0147),
                    nav_row("SCBSET50", "2026-10-30", 24.044),
                ]
            )
        )
        assert fetch_latest_nav("scbset50") == Decimal("24.044")

    def test_single_class_fund_via_profile(self, fake_sec_api: FakeSec) -> None:
        """A fund whose only class is 'main' is found through its profile's proj_id."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == sec_fund_api.PROFILES_PATH:
                return page([{"proj_id": "M0405_2549", "proj_abbr_name": "K-VALUERMF"}])
            if request.url.params.get("proj_id") == "M0405_2549":
                return page([nav_row("main", "2026-10-30", 34.443, "M0405_2549")])
            return httpx.Response(204)

        fake_sec_api(handler)
        assert fetch_latest_nav("K-VALUERMF") == Decimal("34.443")

    def test_unknown_fund(self, fake_sec_api: FakeSec) -> None:
        """Nothing anywhere → None."""
        fake_sec_api(lambda request: httpx.Response(204))
        assert fetch_latest_nav("NOPE") is None
        assert fetch_latest_nav("  ") is None
