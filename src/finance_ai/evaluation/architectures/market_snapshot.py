"""Fixed market data for the architecture study.

Replaces the four network sources the tools use (yfinance, Google News RSS,
Yahoo symbol search, SEC fund search) with canned data, so every design
sees the same prices and headlines and runs are repeatable. Patched at the
lowest network layer; the production tool and service code still runs.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal
from email.utils import format_datetime
from typing import Any, Optional
from unittest.mock import patch
from xml.sax.saxutils import escape

import pandas as pd

from finance_ai.tools.fund_search_service import FundSearchOutcome, FundSearchResult

PRICES: dict[str, Decimal] = {
    "PTT.BK": Decimal("33.25"),
    "AOT.BK": Decimal("58.50"),
    "KBANK.BK": Decimal("152.00"),
    "CPALL.BK": Decimal("54.75"),
    "ADVANC.BK": Decimal("301.00"),
    "KT-RMFX": Decimal("14.20"),
}
NAMES: dict[str, str] = {
    "PTT.BK": "PTT Public Company Limited",
    "AOT.BK": "Airports of Thailand Public Company Limited",
    "KBANK.BK": "Kasikornbank Public Company Limited",
    "CPALL.BK": "CP ALL Public Company Limited",
    "ADVANC.BK": "Advanced Info Service Public Company Limited",
    "KT-RMFX": "Krungthai RMF",
}
HEADLINES: dict[str, str] = {
    "PTT.BK": "ปตท.รายงานกำไรไตรมาส 3 ลดลงตามราคาน้ำมันโลก",
    "AOT.BK": "ทอท.เผยผู้โดยสารสนามบินสุวรรณภูมิเดือนตุลาคมเพิ่มขึ้น 8%",
    "KBANK.BK": "กสิกรไทยตั้งสำรองหนี้เพิ่ม หลังสินเชื่อรายย่อยชะลอตัว",
    "CPALL.BK": "ซีพี ออลล์ขยายสาขาเซเว่นฯ ต่อเนื่อง ยอดขายสาขาเดิมโต 4%",
    "ADVANC.BK": "เอไอเอสเพิ่มผู้ใช้ 5G ทะลุ 15 ล้านราย",
}
_SEARCH_KEYS: dict[str, str] = {
    "PTT": "PTT.BK",
    "ปตท": "PTT.BK",
    "AOT": "AOT.BK",
    "ท่าอากาศยาน": "AOT.BK",
    "KBANK": "KBANK.BK",
    "กสิกร": "KBANK.BK",
    "CPALL": "CPALL.BK",
    "ADVANC": "ADVANC.BK",
}
NEWS_SOURCE = ("Reuters", "https://www.reuters.com")
SET50_FUNDS: tuple[FundSearchResult, ...] = (
    FundSearchResult(
        symbol="K-SET50",
        proj_id="M0001_2549",
        fund_code="K-SET50",
        name_th="กองทุนเปิดเค เซ็ท 50",
        name_en="K SET50 Index Fund",
        amc="กสิกรไทย",
        payout="A",
        nav=Decimal("21.45"),
        nav_date=date(2026, 10, 30),
    ),
    FundSearchResult(
        symbol="SCBSET50",
        proj_id="M0002_2550",
        fund_code="SCBSET50",
        name_th="กองทุนเปิดไทยพาณิชย์ SET50",
        name_en="SCB SET50 Index Fund",
        amc="ไทยพาณิชย์",
        payout="A",
        nav=Decimal("18.02"),
        nav_date=date(2026, 10, 30),
    ),
)


class _FastInfo:
    """Stand-in for yfinance FastInfo."""

    def __init__(self, price: Optional[Decimal]) -> None:
        """Store the snapshot price."""
        self.last_price = float(price) if price is not None else None


class SnapshotTicker:
    """Stand-in for ``yfinance.Ticker`` that answers from PRICES."""

    def __init__(self, symbol: str, *_args: Any, **_kwargs: Any) -> None:
        """Look up the snapshot price for a symbol."""
        self.symbol = symbol.upper()
        self._price = PRICES.get(self.symbol)
        self.fast_info = _FastInfo(self._price)

    @property
    def info(self) -> dict[str, Any]:
        """Return a minimal quote block (empty for unknown symbols)."""
        if self._price is None:
            return {}
        return {
            "currentPrice": float(self._price),
            "currency": "THB",
            "longName": NAMES.get(self.symbol, self.symbol),
        }

    def history(self, *_args: Any, **_kwargs: Any) -> pd.DataFrame:
        """Return one closing price (empty for unknown symbols)."""
        closes = [float(self._price)] if self._price is not None else []
        return pd.DataFrame({"Close": closes})


def snapshot_news_rss(symbol: str) -> str:
    """Return an RSS feed with one recent allowlisted headline for a symbol.

    Args:
        symbol: Ticker symbol.

    Returns:
        RSS XML text (no items for unknown symbols).
    """
    headline = HEADLINES.get(symbol.upper())
    if headline is None:
        return "<rss><channel></channel></rss>"
    published = format_datetime(datetime.now(timezone.utc))
    source_name, source_url = NEWS_SOURCE
    item = (
        f"<item><title>{escape(headline)}</title><link>{source_url}/th/{symbol}</link>"
        f'<pubDate>{published}</pubDate><source url="{source_url}">{source_name}</source></item>'
    )
    return f"<rss><channel>{item}</channel></rss>"


def snapshot_symbol_search(query: str) -> dict[str, Any]:
    """Return a Yahoo-search-shaped payload for known names in the query.

    Args:
        query: Free-text search query.

    Returns:
        ``{"quotes": [...]}`` with at most one match.
    """
    upper = query.upper()
    for key, symbol in _SEARCH_KEYS.items():
        if key in upper or key in query:
            quote = {"symbol": symbol, "shortname": NAMES[symbol], "quoteType": "EQUITY"}
            return {"quotes": [quote | {"exchange": "SET"}]}
    return {"quotes": []}


def snapshot_fund_search(query: str, *_args: Any, **_kwargs: Any) -> FundSearchOutcome:
    """Return the two SET50 index funds for SET50 queries, otherwise nothing.

    Args:
        query: Fund search text.
        *_args: Ignored (expand callback).
        **_kwargs: Ignored.

    Returns:
        FundSearchOutcome from the snapshot.
    """
    results = list(SET50_FUNDS) if "50" in query else []
    return FundSearchOutcome(results=results)


@contextmanager
def market_snapshot() -> Iterator[None]:
    """Patch every market-data network call with the snapshot.

    Example:
        >>> with market_snapshot():
        ...     import yfinance
        ...     yfinance.Ticker("PTT.BK").fast_info.last_price
        33.25
    """
    targets = {
        "yfinance.Ticker": SnapshotTicker,
        "finance_ai.tools.market_data_service._fetch_news_rss": snapshot_news_rss,
        "finance_ai.tools.symbol_search_service._fetch_search_payload": snapshot_symbol_search,
        "finance_ai.agents.fund_search_tools.search_funds": snapshot_fund_search,
        "finance_ai.tools.asset_search_service.search_funds": snapshot_fund_search,
    }
    patchers = [patch(target, replacement) for target, replacement in targets.items()]
    for patcher in patchers:
        patcher.start()
    try:
        yield
    finally:
        for patcher in reversed(patchers):
            patcher.stop()
