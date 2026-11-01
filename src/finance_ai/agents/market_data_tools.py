"""LangGraph tools for market data: symbol search, stock prices, news.

These tools require no database access — they fetch external data from
free public sources (Yahoo Finance search, yfinance, Google News RSS).
"""

from decimal import Decimal
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.citation_sources import (
    CitationSource,
    build_news_citations,
    collect_turn_citations,
)


@tool
def get_stock_price(symbol: str) -> dict[str, Any]:
    """ดึงข้อมูลราคาและสถิติสินทรัพย์ทางการเงิน.

    ข้อมูลที่ได้: ชื่อ, ราคาปัจจุบัน (พร้อมหน่วย เช่น บาท/USD), P/E, Market Cap,
    52-Week Range, Dividend Yield, เป้าหมายนักวิเคราะห์, คำแนะนำ
    แสดงราคาด้วยช่อง price_display เสมอ (เช่น "35.50 บาท") เพื่อให้เห็นหน่วยชัดเจน

    Args:
        symbol: สัญลักษณ์สินทรัพย์ เช่น 'PTT.BK', 'AAPL', 'BTC-USD', 'GC=F'

    Returns:
        Dict with price and statistics fields.

    Example:
        >>> get_stock_price("PTT.BK")
    """
    from finance_ai.tools.market_data_service import (  # noqa: PLC0415
        fetch_stock_dashboard,
        format_price_with_unit,
    )

    result = fetch_stock_dashboard(symbol)
    data = result.model_dump()
    output: dict[str, Any] = {"action": "stock_price", "symbol": symbol}
    for key, value in data.items():
        output[key] = str(value) if value is not None else None
    if isinstance(result.current_price, Decimal):
        output["price_display"] = format_price_with_unit(result.current_price, symbol)
    return output


def _numbered_news_digest(symbol: str, citations: list[CitationSource]) -> str:
    """List news headlines with their citation numbers (no URLs)."""
    lines = [f"[{c.index}] {c.title} — {c.publisher or ''} {c.snippet}".rstrip() for c in citations]
    return f"ข่าวล่าสุดสำหรับ {symbol} (อ้างอิงด้วยเลขในวงเล็บ):\n" + "\n".join(lines)


@tool
def search_finance_news(
    symbol: str,
    messages: Annotated[list[Any] | None, InjectedState("messages")] = None,
) -> dict[str, Any]:
    """ค้นหาข่าวสารล่าสุดเกี่ยวกับสินทรัพย์ทางการเงิน.

    รองรับหุ้น, คริปโต, ดัชนี เช่น 'AAPL', 'BTC-USD', 'PTT.BK', 'GC=F'
    ข่าวแต่ละข่าวมีเลขกำกับ [n] ให้อ้างอิงด้วยเลขนั้น ห้ามพิมพ์ลิงก์เอง
    (ระบบแสดงลิงก์ ชื่อสำนักข่าว และวันที่ให้ผู้ใช้เอง)

    Args:
        symbol: ตัวย่อสินทรัพย์ที่ต้องการหาข่าว
        messages: Conversation so far (injected from graph state) for citation numbering.

    Returns:
        Dict with numbered news content, citations and availability flag.

    Example:
        >>> search_finance_news("PTT.BK")
    """
    from finance_ai.tools.market_data_service import (  # noqa: PLC0415
        fetch_finance_news,
    )

    result = fetch_finance_news(symbol)
    citations = build_news_citations(result.articles, collect_turn_citations(messages or []))
    return {
        "action": "finance_news",
        "symbol": result.symbol,
        "news_content": (
            _numbered_news_digest(result.symbol, citations) if citations else result.news_content
        ),
        "has_news": result.has_news,
        "citations": [citation.model_dump() for citation in citations],
    }


@tool
def resolve_asset_symbol(query: str) -> dict[str, Any]:
    """ค้นหาสัญลักษณ์สินทรัพย์จากข้อความอิสระ (free text).

    ใช้เมื่อผู้ใช้พูดถึงสินทรัพย์ด้วยชื่อเรียกทั่วไป เช่น "หุ้นปตท", "Apple",
    "ทองคำ", "BTC" — ระบบจะค้นหาและคืนรายการสัญลักษณ์ที่ตรงกันพร้อมชื่อและตลาด
    หากผู้ใช้พิมพ์ภาษาไทย ให้แปลเป็นชื่อภาษาอังกฤษหรือตัวย่อก่อนเรียกใช้

    Args:
        query: ชื่อสินทรัพย์ที่ต้องการค้นหา

    Returns:
        Dict with "action": "resolve_symbol" and "candidates":
        list of {"symbol", "name", "exchange", "type"} (top 5).

    Example:
        >>> resolve_asset_symbol("PTT")
    """
    from finance_ai.tools.symbol_search_service import (  # noqa: PLC0415
        search_asset_symbols,
    )

    matches = search_asset_symbols(query)[:5]
    candidates = [
        {
            "symbol": match.symbol,
            "name": match.name,
            "exchange": match.exchange,
            "type": match.quote_type,
        }
        for match in matches
    ]
    return {"action": "resolve_symbol", "candidates": candidates}


MARKET_DATA_TOOLS = [
    get_stock_price,
    search_finance_news,
    resolve_asset_symbol,
]
