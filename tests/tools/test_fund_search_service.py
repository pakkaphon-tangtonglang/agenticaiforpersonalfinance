"""Tests for fund_search_service (Thai fund search by name/house/code)."""

from collections.abc import Callable
from typing import Any

import httpx

from finance_ai.tools import sec_fund_api
from finance_ai.tools.fund_search_service import find_amc, payout_type, search_funds

FakeSec = Callable[[Callable[[httpx.Request], httpx.Response]], list[httpx.Request]]

AMCS = [
    {
        "unique_id": "C21",
        "comp_name_th": "บริษัทหลักทรัพย์จัดการกองทุน กสิกรไทย จำกัด",
        "comp_name_en": "KASIKORN ASSET MANAGEMENT",
    },
    {
        "unique_id": "C709",
        "comp_name_th": "บริษัทหลักทรัพย์จัดการกองทุน กรุงศรี จำกัด",
        "comp_name_en": "KRUNGSRI ASSET MANAGEMENT",
    },
    {
        "unique_id": "C460",
        "comp_name_th": "บริษัทหลักทรัพย์จัดการกองทุน กรุงไทย จำกัด (มหาชน)",
        "comp_name_en": "KRUNG THAI ASSET MANAGEMENT",
    },
    {
        "unique_id": "C271",
        "comp_name_th": "บริษัท ไทยจัดการทรัพย์ จำกัด",
        "comp_name_en": "THAI ASSET MANAGEMENT",
    },
    {
        "unique_id": "C182",
        "comp_name_th": (
            "บริษัทหลักทรัพย์จัดการกองทุน ทหารไทย จำกัด ยกเลิกการใช้งานเนื่องจากควบรวมกิจการ"
        ),
        "comp_name_en": "TMB",
    },
]


def page(items: list[dict[str, Any]]) -> httpx.Response:
    """One JSON page."""
    return httpx.Response(200, json={"items": items, "next_cursor": None})


def profile(
    proj_id: str, code: str, klass: str, name: str, house: str, uid: str, desc: str = ""
) -> dict[str, Any]:  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """One profile row (one class of a fund)."""
    return {
        "proj_id": proj_id, "proj_abbr_name": code, "fund_class_name": klass,
        "proj_name_th": name, "proj_name_en": "", "comp_name_th": house,
        "unique_id": uid, "fund_class_description": desc,
    }  # fmt: skip


def nav(proj_id: str, klass: str, value: float) -> dict[str, Any]:
    """One NAV row."""
    return {
        "proj_id": proj_id,
        "fund_class_name": klass,
        "nav_date": "2026-10-30",
        "last_val": value,
    }


KASIKORN = "บริษัทหลักทรัพย์จัดการกองทุน กสิกรไทย จำกัด"
KRUNGSRI = "บริษัทหลักทรัพย์จัดการกองทุน กรุงศรี จำกัด"
PROFILES = {
    "ปันผล": [
        profile("P1", "K-VALUE", "K-VALUE-A(A)", "กองทุนเปิดเค หุ้นปันผล", KASIKORN, "C21"),
        profile("P1", "K-VALUE", "K-VALUE-A(D)", "กองทุนเปิดเค หุ้นปันผล", KASIKORN, "C21"),
        profile("P1", "K-VALUE", "K-VALUE-P(A)", "กองทุนเปิดเค หุ้นปันผล", KASIKORN, "C21"),
        profile("P2", "KFSDIV", "main", "กองทุนเปิดกรุงศรีหุ้นปันผล", KRUNGSRI, "C709"),
    ],
}
COMPANY = {
    "C709": [
        profile("P2", "KFSDIV", "main", "กองทุนเปิดกรุงศรีหุ้นปันผล", KRUNGSRI, "C709"),
        profile("P3", "KF-GOLD", "main", "กองทุนเปิดกรุงศรีโกลด์", KRUNGSRI, "C709"),
    ],
}
NAVS = {
    "P1": [nav("P1", "K-VALUE-A(A)", 7.36), nav("P1", "K-VALUE-A(D)", 6.5)],
    "P2": [nav("P2", "main", 5.23)],
    "P3": [nav("P3", "main", 12.0)],
    "P9": [nav("P9", "K-US500X-A(A)", 16.0)],
}


def sec_handler(
    extra_profiles: dict[str, list[dict[str, Any]]] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """Answer AMC, profile and NAV requests from the tables above."""
    names = {**PROFILES, **(extra_profiles or {})}

    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if request.url.path == sec_fund_api.AMCS_PATH:
            return page(AMCS)
        if request.url.path == sec_fund_api.PROFILES_PATH:
            rows = COMPANY.get(params.get("company_info", "")) or names.get(
                params.get("project_info", "")
            )
            return page(rows) if rows else httpx.Response(204)
        rows = NAVS.get(params.get("proj_id", ""))
        return page(rows) if rows else httpx.Response(204)

    return handler


class TestFindAmc:
    """Fund-house detection in a query."""

    def test_thai_name_with_or_without_space(self, fake_sec_api: FakeSec) -> None:
        """'กรุงศรี ปันผล' and 'กรุงศรีปันผล' both split into house + 'ปันผล'."""
        fake_sec_api(sec_handler())
        for query in ("กรุงศรี ปันผล", "กรุงศรีปันผล"):
            amc, rest = find_amc(query)
            assert amc is not None and amc.short_name_th == "กรุงศรี"
            assert rest == "ปันผล"

    def test_alias_and_english_name(self, fake_sec_api: FakeSec) -> None:
        """'kbank' and 'krungsri' are recognized; 'krungsri' is not Krung Thai."""
        fake_sec_api(sec_handler())
        amc, rest = find_amc("kbank หุ้น")
        assert amc is not None and (amc.short_name_th, rest) == ("กสิกรไทย", "หุ้น")
        amc, _ = find_amc("krungsri gold")
        assert amc is not None and amc.short_name_th == "กรุงศรี"

    def test_generic_words_are_not_houses(self, fake_sec_api: FakeSec) -> None:
        """'thai stock' must not pick Thai Asset Management; merged houses are skipped."""
        fake_sec_api(sec_handler())
        assert find_amc("thai stock") == (None, "thai stock")
        assert find_amc("ทหารไทย")[0] is None


class TestSearchFunds:
    """search_funds end to end against a fake SEC API."""

    def test_name_search_attaches_nav_and_skips_inactive_classes(
        self, fake_sec_api: FakeSec
    ) -> None:
        """Classes without a recent NAV (K-VALUE-P) are dropped; 'main' uses the fund code."""
        fake_sec_api(sec_handler())
        results = search_funds("ปันผล").results
        assert [(r.symbol, str(r.nav), r.payout, r.amc) for r in results] == [
            ("K-VALUE-A(A)", "7.36", "A", "กสิกรไทย"),
            ("K-VALUE-A(D)", "6.5", "D", "กสิกรไทย"),
            ("KFSDIV", "5.23", "", "กรุงศรี"),
        ]

    def test_house_plus_word_filters_that_house(self, fake_sec_api: FakeSec) -> None:
        """'กรุงศรีปันผล' → Krungsri funds whose name contains 'ปันผล'."""
        fake_sec_api(sec_handler())
        assert [r.symbol for r in search_funds("กรุงศรีปันผล").results] == ["KFSDIV"]

    def test_leading_fund_word_is_ignored(self, fake_sec_api: FakeSec) -> None:
        """'กองทุนปันผล' searches 'ปันผล'."""
        fake_sec_api(sec_handler())
        assert search_funds("กองทุนปันผล").results

    def test_expansion_only_when_nothing_found(self, fake_sec_api: FakeSec) -> None:
        """The expander runs only for empty results; 'US' must not match 'PLUS'."""
        extra = {
            "US": [
                profile(
                    "P9", "K-US500X", "K-US500X-A(A)", "กองทุนเปิดเค หุ้นยูเอส", KASIKORN, "C21"
                ),
                profile(
                    "P8", "K-SFPLUS", "K-SFPLUS-A", "กองทุนเปิดเค เอสเอฟ พลัส", KASIKORN, "C21"
                ),
            ]
        }
        fake_sec_api(sec_handler(extra))
        asked: list[str] = []

        def expand(query: str) -> list[str]:
            asked.append(query)
            return ["US"]

        assert search_funds("ปันผล", expand=expand).results
        assert not asked
        outcome = search_funds("หุ้นอเมริกา", expand=expand)
        assert asked == ["หุ้นอเมริกา"]
        assert outcome.expanded_terms == ["US"]
        assert [r.symbol for r in outcome.results] == ["K-US500X-A(A)"]

    def test_short_query_or_no_key(self, fake_sec_api: FakeSec) -> None:
        """One character, or no key, returns nothing."""
        fake_sec_api(sec_handler())
        assert not search_funds("ก").results


def test_search_without_key_makes_no_request() -> None:
    """conftest removes the key: nothing found, nothing requested."""
    assert not search_funds("ปันผล").results


def test_payout_type() -> None:
    """(A)/(D) suffix first, then the Thai class description."""
    assert payout_type("K-VALUE-A(D)", "") == "D"
    assert payout_type("KFSEQ2-A", "สะสมผลประโยชน์จากการลงทุน") == "A"
    assert payout_type("KFSEQ-D", "รับผลตอบแทนจากเงินปันผล") == "D"
    assert payout_type("X", "") == ""
