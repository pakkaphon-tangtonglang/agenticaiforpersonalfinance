"""LangGraph tool for the user's investment portfolio (what they own).

Holdings are not the watchlist: the watchlist is assets the user only
follows; holdings have quantity, cost and gain/loss.
"""

from datetime import date
from decimal import Decimal
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.clarification import clarification, symbol_problem
from finance_ai.agents.expense_tools import parse_date_value
from finance_ai.agents.session_helper import get_tool_session
from finance_ai.agents.tax_tools import parse_decimal_value
from finance_ai.tools.deduction_service import detect_tax_saving_fund_type
from finance_ai.tools.foreign_stock_service import (
    FOREIGN_STOCK,
    is_foreign_stock_symbol,
    price_in_thb,
)
from finance_ai.tools.investment_calculator import HoldingSummary
from finance_ai.tools.investment_service import add_investment_holding, get_portfolio_summary
from finance_ai.tools.investment_calculator import validate_mutual_fund_symbol
from finance_ai.tools.investment_constants import STOCK_SYMBOL_SUFFIX
from finance_ai.tools.symbol_search_service import clean_listing_name
from finance_ai.tools.symbol_guard import (
    SymbolResolutionError,
    SymbolSearchUnavailableError,
    resolve_and_validate_symbol,
)

EMPTY_PORTFOLIO_MESSAGE: str = (
    "ยังไม่มีรายการถือครองในพอร์ต (นำเข้าได้จากหน้าสินทรัพย์ หรือบอกว่าซื้ออะไรไว้)"
)

ASSET_TYPE_ALIASES: dict[str, str] = {
    "fund": "mutual_fund",
    "mutual fund": "mutual_fund",
    "กองทุน": "mutual_fund",
    "กองทุนรวม": "mutual_fund",
    "หุ้น": "stock",
    "หุ้นต่างประเทศ": FOREIGN_STOCK,
    "foreign": FOREIGN_STOCK,
    "stocks": "stock",
}
ASSET_TYPE_OPTIONS: list[str] = ["หุ้นไทย", "กองทุนรวม"]
ASK_ASSET_TYPE = "เป็นหุ้นไทยหรือกองทุนรวม"
ASK_IS_FUND = "ไม่พบหุ้นชื่อ {code} — เป็นกองทุนรวมใช่ไหม"
ASK_FUND_CODE = "ขอรหัสกองทุนอีกครั้ง เช่น K-USA-A(A), KFSDIV, SCBSET50"
ASK_QUANTITY = "ซื้อจำนวนกี่หุ้น/หน่วย"
ASK_PRICE = "ซื้อมาราคาต่อหุ้น/หน่วยเท่าไร (บาท)"
ASK_DATE = "ซื้อวันไหน (เช่น 2026-10-01 หรือ 1/10/2569)"
ASK_POSITIVE_VALUES = "ขอจำนวนและราคาที่มากกว่า 0 อีกครั้ง"
_CENTS = Decimal("0.01")
UNSUPPORTED_ASSET = (
    "'{symbol}' is not a stock (crypto, gold, index or currency). The portfolio holds"
    " Thai stocks, foreign stocks and Thai mutual funds; add it to the watchlist instead."
)


@tool
def get_portfolio_holdings(
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """ดูพอร์ตการลงทุน: สินทรัพย์ที่ผู้ใช้ถือจริง (จำนวน ต้นทุน มูลค่า กำไร/ขาดทุน).

    ใช้เมื่อผู้ใช้ถามถึง "รายการถือหุ้น", "หุ้นที่ถืออยู่", "พอร์ต", "ถืออะไรอยู่",
    "กำไรขาดทุนพอร์ต". ไม่ใช่ Watchlist (รายการที่แค่ติดตาม) — ใช้ manage_watchlist แทน
    เฉพาะเมื่อผู้ใช้พูดถึงการติดตาม/watchlist.

    Args:
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict with holding_count, totals, holdings (money as strings, None when
        no price is stored) and a message when the portfolio is empty.

    Example:
        >>> get_portfolio_holdings.invoke({"user_id": uid, "db_session_factory": f})
    """
    with get_tool_session(db_session_factory) as session:
        summary = get_portfolio_summary(session, user_id)
    return {
        "holding_count": summary.holding_count,
        "total_cost": _money(summary.total_cost),
        "total_current_value": _money(summary.total_current_value),
        "total_unrealized_gain_loss": _money(summary.total_unrealized_gain_loss),
        "holdings": [holding_to_dict(holding) for holding in summary.holdings],
        "message": "" if summary.holdings else EMPTY_PORTFOLIO_MESSAGE,
    }


@tool
def add_portfolio_holding(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    symbol: str,
    quantity: str,
    price_per_unit: str,
    purchase_date: str = "",
    asset_type: str = "stock",
    name: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """เพิ่มรายการถือครองในพอร์ต: สินทรัพย์ที่ผู้ใช้ซื้อจริง (ไม่ใช่ Watchlist).

    ใช้เมื่อผู้ใช้บอกว่า "ซื้อ", "ถือ", "เพิ่มเข้าพอร์ต", "เพิ่มรายการถือครอง" พร้อมจำนวนและราคา.
    ถ้าผู้ใช้ยังไม่บอกจำนวนหรือราคาต่อหน่วย ให้ถามก่อน ห้ามเดา.
    รองรับหุ้นไทย (.BK), หุ้นต่างประเทศ (เช่น NVDA, AAPL — ใช้ asset_type="stock"
    ระบบแยกให้เอง) และกองทุนรวมทุกบริษัทจัดการ (asset_type="mutual_fund"
    เช่น K-USA-A(A), KFSDIV, SCBSET50). คริปโต/ทองคำ เพิ่มเข้าพอร์ตไม่ได้.
    ถ้าผลลัพธ์มี needs_clarification ให้ถามผู้ใช้ตาม ask_user ห้ามเดาแทน.

    Args:
        symbol: สัญลักษณ์ เช่น "PTT", "PTT.BK", "K-USA-A(A)".
        quantity: จำนวนหน่วย/หุ้น เช่น "100" หรือ "1,000".
        price_per_unit: ราคาซื้อต่อหน่วย เช่น "35.50" (บาท; หุ้นต่างประเทศใช้สกุลเงินของหุ้น
            เช่น USD แล้วระบบแปลงเป็นบาทตามอัตราวันนี้).
        purchase_date: วันที่ซื้อ YYYY-MM-DD หรือ DD/MM/YYYY (พ.ศ. ได้) ค่าเริ่มต้น = วันนี้.
        asset_type: "stock" หรือ "mutual_fund".
        name: ชื่อที่แสดง เช่น "PTT".
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict with status "added" and the saved values, an {"error": ...},
        or a clarification (needs_clarification, ask_user, options).

    Example:
        >>> add_portfolio_holding.invoke({"symbol": "PTT", "quantity": "100",
        ...     "price_per_unit": "35.50", "user_id": uid, "db_session_factory": f})
    """
    typed_type = ASSET_TYPE_ALIASES.get(asset_type.strip().lower(), asset_type.strip().lower())
    normalized_type, holding_symbol, problem = _holding_identity(symbol, typed_type)
    if problem:
        return problem
    values = _parse_holding_values(quantity, price_per_unit, purchase_date)
    if "error" in values:
        return values
    conversion = _convert_foreign_price(normalized_type, holding_symbol, values)
    if "error" in conversion:
        return conversion
    try:
        with get_tool_session(db_session_factory) as session:
            holding = add_investment_holding(
                session, user_id, normalized_type, holding_symbol, name or holding_symbol,
                values["quantity"], values["price"], values["bought_on"],
            )  # fmt: skip
            return {
                "status": "added",
                **_saved_holding(holding.symbol, holding.total_cost),
                **conversion,
                **_follow_up_after_add(normalized_type, holding_symbol, name, holding.total_cost),
            }
    except ValueError as exc:
        return clarification(str(exc), ASK_POSITIVE_VALUES)


def _follow_up_after_add(
    asset_type: str, symbol: str, name: str, total_cost: Decimal
) -> dict[str, Any]:
    """Related step after a purchase: record an RMF/SSF/ThaiESG deduction, or watch a stock's news.

    Args:
        asset_type: "stock" or "mutual_fund".
        symbol: Canonical holding symbol.
        name: Display name the agent passed (fund names reveal RMF/SSF/ThaiESG).
        total_cost: Purchase cost in THB.

    Returns:
        {"tax_deduction_hint": {...}} for an RMF/SSF/ThaiESG, {"watchlist_hint": {...}}
        for a stock, else {}.
    """
    if asset_type in ("stock", FOREIGN_STOCK):
        return {"watchlist_hint": {"symbol": symbol, "name": name or symbol}}
    fund_type = detect_tax_saving_fund_type(symbol, name)
    if fund_type is None:
        return {}
    amount = _money(total_cost)
    return {"tax_deduction_hint": {"deduction_type": fund_type, "amount": amount}}


def _fund_identity(symbol: str) -> tuple[str, str, dict[str, Any] | None]:
    """A Thai fund code, or a question asking for it again."""
    try:
        return "mutual_fund", validate_mutual_fund_symbol(symbol), None
    except ValueError as exc:
        return "mutual_fund", "", clarification(str(exc), ASK_FUND_CODE)


def _holding_identity(symbol: str, asset_type: str) -> tuple[str, str, dict[str, Any] | None]:
    """Asset type and canonical symbol (Thai/foreign stock or fund), or a problem."""
    if asset_type == "mutual_fund":
        return _fund_identity(symbol)
    if asset_type not in ("stock", FOREIGN_STOCK):
        return (
            asset_type,
            "",
            clarification(
                f"Unknown asset_type '{asset_type}'.", ASK_ASSET_TYPE, ASSET_TYPE_OPTIONS
            ),
        )
    try:
        canonical = resolve_and_validate_symbol(symbol)
    except SymbolResolutionError as exc:
        return asset_type, "", _stock_not_found(exc, symbol)
    if canonical.endswith(STOCK_SYMBOL_SUFFIX):
        return "stock", canonical, None
    if is_foreign_stock_symbol(canonical):
        return FOREIGN_STOCK, canonical, None
    return (
        asset_type,
        "",
        {"error": UNSUPPORTED_ASSET.format(symbol=canonical), "unsupported": True},
    )


def _convert_foreign_price(asset_type: str, symbol: str, values: dict[str, Any]) -> dict[str, Any]:
    """Turn a foreign stock's price (USD...) into baht in values; report the rate used."""
    if asset_type != FOREIGN_STOCK:
        return {}
    try:
        converted = price_in_thb(symbol, values["price"])
    except ValueError as exc:
        return {"error": str(exc)}
    values["price"] = converted.amount_thb
    return {
        "currency": converted.currency,
        "price_in_currency": str((values["price"] / converted.exchange_rate).quantize(_CENTS)),
        "exchange_rate": str(converted.exchange_rate),
    }


def _stock_not_found(error: SymbolResolutionError, typed: str) -> dict[str, Any]:
    """A stock search miss on something shaped like a fund code asks "is it a fund?"."""
    if error.candidates or isinstance(error, SymbolSearchUnavailableError):
        return symbol_problem(error)
    try:
        fund_code = validate_mutual_fund_symbol(typed)
    except ValueError:
        return symbol_problem(error)
    return clarification(str(error), ASK_IS_FUND.format(code=fund_code), ASSET_TYPE_OPTIONS)


def _parse_holding_values(quantity: str, price: str, purchase_date: str) -> dict[str, Any]:
    """Parse quantity, price and date; a clarification names the value that is unclear."""
    try:
        parsed_quantity = parse_decimal_value(quantity, "quantity")
    except ValueError as exc:
        return clarification(str(exc), ASK_QUANTITY)
    try:
        parsed_price = parse_decimal_value(price, "price_per_unit")
    except ValueError as exc:
        return clarification(str(exc), ASK_PRICE)
    try:
        bought_on = (
            parse_date_value(purchase_date, "purchase_date") if purchase_date else date.today()
        )
    except ValueError as exc:
        return clarification(str(exc), ASK_DATE)
    if bought_on > date.today():
        return clarification(f"purchase_date {bought_on} is in the future.", ASK_DATE)
    return {"quantity": parsed_quantity, "price": parsed_price, "bought_on": bought_on}


def _saved_holding(symbol: str, total_cost: Decimal) -> dict[str, str]:
    """Display values for a saved holding."""
    return {"symbol": symbol, "total_cost": str(total_cost.quantize(Decimal("0.01")))}


def holding_to_dict(holding: HoldingSummary) -> dict[str, Any]:
    """Convert one holding to plain values (money as strings, quantity unpadded).

    Used by the agent tool and the /assets/holdings endpoint.

    Args:
        holding: Holding summary from the investment service.

    Returns:
        Dict of display-ready fields (None where no price is stored).
    """
    return {
        "holding_id": holding.holding_id,
        "symbol": holding.symbol,
        "name": clean_listing_name(holding.symbol, holding.name),
        "asset_type": holding.asset_type_label,
        "quantity": _plain(holding.quantity),
        "average_cost_per_unit": _money(holding.average_cost_per_unit),
        "total_cost": _money(holding.total_cost),
        "current_price_per_unit": _money(holding.current_price_per_unit),
        "current_value": _money(holding.current_value),
        "unrealized_gain_loss": _money(holding.unrealized_gain_loss),
        "gain_loss_percentage": _plain(holding.gain_loss_percentage),
    }


def _money(amount: Decimal | None) -> str | None:
    """Money with 2 decimals, or None when unknown."""
    return None if amount is None else str(amount.quantize(Decimal("0.01")))


def _plain(value: Decimal | None) -> str | None:
    """Decimal without trailing zeros (100.0000 → 100), or None."""
    if value is None:
        return None
    text = format(value.normalize(), "f")
    return text
