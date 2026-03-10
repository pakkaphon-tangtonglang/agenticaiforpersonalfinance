"""LangGraph tool wrappers for market data: dashboard, forex, news.

These tools require no database access or InjectedState — they fetch
external data from yfinance and Yahoo Finance News only.
"""

from decimal import Decimal, InvalidOperation
from typing import Any

from langchain_core.tools import tool


@tool
def get_stock_dashboard(symbol: str) -> dict[str, Any]:
    """ดึงข้อมูลภาพรวมสินทรัพย์ (Dashboard) ทั้งหุ้นและคริปโต.

    ข้อมูลที่ได้: ชื่อ, ราคา, P/E, Market Cap, 52-Week Range,
    Dividend Yield, เป้าหมายนักวิเคราะห์, คำแนะนำ

    Args:
        symbol: สัญลักษณ์สินทรัพย์ เช่น 'PTT.BK', 'AAPL', 'BTC-USD'

    Returns:
        Dict with dashboard fields (values as strings for LLM).
    """
    from finance_ai.tools.market_data_service import (  # noqa: PLC0415
        fetch_stock_dashboard,
    )

    result = fetch_stock_dashboard(symbol)
    return _serialize_dashboard(result, symbol)


def _serialize_dashboard(result: Any, symbol: str) -> dict[str, Any]:
    """Convert StockDashboardResult to LLM-friendly dict.

    Args:
        result: StockDashboardResult from service layer.
        symbol: Original symbol for reference.

    Returns:
        Dict with all values as strings.
    """
    data = result.model_dump()
    output: dict[str, Any] = {"action": "stock_dashboard", "symbol": symbol}
    for key, value in data.items():
        output[key] = str(value) if value is not None else None
    return output


@tool
def convert_currency_tool(
    from_currency: str,
    to_currency: str,
    amount: str = "1",
) -> dict[str, Any]:
    """แปลงค่าเงินระหว่างสกุลเงิน ใช้อัตราแลกเปลี่ยน Real-time.

    รองรับสกุลเงินหลัก เช่น THB, USD, EUR, JPY, GBP, CNY, SGD

    Args:
        from_currency: รหัสสกุลเงินต้นทาง เช่น 'USD'
        to_currency: รหัสสกุลเงินปลายทาง เช่น 'THB'
        amount: จำนวนเงินที่ต้องการแปลง (ค่าเริ่มต้น '1')

    Returns:
        Dict with conversion details (values as strings for LLM).
    """
    from finance_ai.tools.market_data_service import (  # noqa: PLC0415
        convert_currency,
    )

    parsed_amount = _parse_amount(amount)
    result = convert_currency(from_currency, to_currency, parsed_amount)
    return _serialize_conversion(result)


def _parse_amount(amount: str) -> Decimal:
    """Parse amount string to Decimal.

    Args:
        amount: String representation of a number.

    Returns:
        Decimal value.

    Raises:
        ValueError: If amount cannot be parsed.
    """
    try:
        return Decimal(amount.strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid amount: '{amount}'. Must be a number.") from exc


def _serialize_conversion(result: Any) -> dict[str, Any]:
    """Convert CurrencyConversionResult to LLM-friendly dict.

    Args:
        result: CurrencyConversionResult from service layer.

    Returns:
        Dict with all values as strings.
    """
    return {
        "action": "convert_currency",
        "from_currency": result.from_currency,
        "to_currency": result.to_currency,
        "amount": str(result.amount),
        "exchange_rate": str(result.exchange_rate),
        "converted_amount": str(result.converted_amount),
    }


@tool
def get_finance_news(symbol: str) -> dict[str, Any]:
    """ค้นหาข่าวสารล่าสุดเกี่ยวกับสินทรัพย์ทางการเงิน.

    รองรับหุ้น, คริปโต, ดัชนี เช่น 'AAPL', 'BTC-USD', 'PTT.BK'

    Args:
        symbol: ตัวย่อสินทรัพย์ที่ต้องการหาข่าว

    Returns:
        Dict with news content and availability flag.
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
    get_stock_dashboard,
    convert_currency_tool,
    get_finance_news,
]
