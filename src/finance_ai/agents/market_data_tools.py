"""LangGraph tools for market data: stock prices and financial news.

These tools require no database access — they fetch external data
from Bright Data Web Scraper API only.
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


MARKET_DATA_TOOLS = [
    get_stock_price,
    search_finance_news,
]
