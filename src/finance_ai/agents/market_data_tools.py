"""LangGraph tools for market data: symbol search, stock prices, news.

These tools require no database access — they fetch external data from
free public sources (Yahoo Finance search, yfinance, Google News RSS).
"""

from typing import Any

from langchain_core.tools import tool


@tool
def get_stock_price(symbol: str) -> dict[str, Any]:
    """ดึงข้อมูลราคาและสถิติสินทรัพย์ทางการเงิน.

    ข้อมูลที่ได้: ชื่อ, ราคาปัจจุบัน, P/E, Market Cap, 52-Week Range,
    Dividend Yield, เป้าหมายนักวิเคราะห์, คำแนะนำ

    Args:
        symbol: สัญลักษณ์สินทรัพย์ เช่น 'PTT.BK', 'AAPL', 'BTC-USD', 'GC=F'

    Returns:
        Dict with price and statistics fields.

    Example:
        >>> get_stock_price("PTT.BK")
    """
    from finance_ai.tools.market_data_service import (  # noqa: PLC0415
        fetch_stock_dashboard,
    )

    result = fetch_stock_dashboard(symbol)
    data = result.model_dump()
    output: dict[str, Any] = {"action": "stock_price", "symbol": symbol}
    for key, value in data.items():
        output[key] = str(value) if value is not None else None
    return output


@tool
def search_finance_news(symbol: str) -> dict[str, Any]:
    """ค้นหาข่าวสารล่าสุดเกี่ยวกับสินทรัพย์ทางการเงิน.

    รองรับหุ้น, คริปโต, ดัชนี เช่น 'AAPL', 'BTC-USD', 'PTT.BK', 'GC=F'

    Args:
        symbol: ตัวย่อสินทรัพย์ที่ต้องการหาข่าว

    Returns:
        Dict with news content and availability flag.

    Example:
        >>> search_finance_news("PTT.BK")
    """
    from finance_ai.tools.market_data_service import (  # noqa: PLC0415
        fetch_finance_news,
    )

    result = fetch_finance_news(symbol)
    return {
        "action": "finance_news",
        "symbol": result.symbol,
        "news_content": result.news_content,
        "has_news": result.has_news,
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
