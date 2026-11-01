"""Add a Thai stock or Thai mutual fund to the portfolio from the asset page form.

The form sends a quantity, an optional buy price and an optional purchase
date. A blank price means "bought at the latest price" (the stock's latest
price, or the fund's latest NAV from the SEC). The latest price is also stored
on the new holding so its gain/loss shows right away.
"""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy.orm import Session

from finance_ai.database.crud.investment_crud import InvestmentHoldingCRUD
from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.tools.fund_nav_client import fetch_latest_nav
from finance_ai.tools.input_normalizer import parse_decimal_text, parse_flexible_date
from finance_ai.tools.investment_calculator import validate_mutual_fund_symbol
from finance_ai.tools.investment_constants import STOCK_SYMBOL_SUFFIX
from finance_ai.tools.investment_service import add_investment_holding
from finance_ai.tools.price_client import fetch_current_price

_MONEY = Decimal("0.01")
_PERCENT = Decimal("0.01")
_HUNDRED = Decimal("100")
STOCK = "stock"
MUTUAL_FUND = "mutual_fund"
THAI_STOCKS_ONLY = "พอร์ตรองรับเฉพาะหุ้นไทย (.BK) — '{symbol}' ไม่ใช่หุ้นในตลาดหลักทรัพย์ฯ"
BAD_FUND_CODE = "รหัสกองทุน '{symbol}' ไม่ถูกต้อง (ตัวอย่าง: K-USA-A(A), SCBSET50)"
UNKNOWN_ASSET_TYPE = "ประเภทสินทรัพย์ต้องเป็นหุ้นไทยหรือกองทุนรวม (ได้รับ: '{value}')"
NO_LATEST_PRICE = "ไม่พบราคาล่าสุดของ {symbol} ตอนนี้ กรุณากรอกราคาซื้อเอง"
BAD_QUANTITY = "จำนวนหุ้น/หน่วยต้องเป็นตัวเลขมากกว่า 0 (ได้รับ: '{value}')"
BAD_PRICE = "ราคาซื้อต้องเป็นตัวเลขมากกว่า 0 หรือเว้นว่างเพื่อใช้ราคาล่าสุด (ได้รับ: '{value}')"
BAD_DATE = "วันที่ซื้ออ่านไม่ออก ใช้รูปแบบ 2026-10-01 หรือ 1/10/2569 (ได้รับ: '{value}')"
FUTURE_DATE = "วันที่ซื้อต้องไม่เกินวันนี้ (ได้รับ: {value})"


class HoldingEntryError(ValueError):
    """The form input cannot be saved; the message is shown to the user in Thai."""


class HoldingForm(BaseModel):
    """What the user typed into the asset page form.

    Example:
        >>> HoldingForm(symbol="PTT.BK", quantity="100").asset_type
        'stock'
    """

    symbol: str
    name: str = ""
    quantity: str
    price_per_unit: str | None = None
    purchase_date: str | None = None
    asset_type: str = STOCK


class HoldingEntry(BaseModel):
    """What was saved, with the gain/loss at the latest price (None without a price).

    Example:
        >>> HoldingEntry(symbol="PTT.BK", quantity="100", price_per_unit="35",
        ...     used_latest_price=True, total_cost="3500.00",
        ...     purchase_date="2026-11-01").gain_loss is None
        True
    """

    symbol: str
    quantity: str
    price_per_unit: str
    used_latest_price: bool
    total_cost: str
    purchase_date: str
    current_price: str | None = None
    current_value: str | None = None
    gain_loss: str | None = None
    gain_loss_percent: str | None = None


def add_holding_from_form(session: Session, user_id: str, form: HoldingForm) -> HoldingEntry:
    """Save a Thai stock or fund purchase typed into the asset page.

    Args:
        session: Database session.
        user_id: Owner's UUID.
        form: Symbol, quantity, optional price/date, and asset type.

    Returns:
        The saved entry with gain/loss at the latest price.

    Raises:
        HoldingEntryError: Unsupported asset, bad numbers or date, or no price.

    Example:
        >>> add_holding_from_form(session, uid, HoldingForm(symbol="PTT.BK", quantity="100"))
    """
    asset_type, symbol = _asset_identity(form.symbol, form.asset_type)
    quantity = _positive(form.quantity, BAD_QUANTITY)
    bought_on = _purchase_date(form.purchase_date)
    latest = latest_unit_price(asset_type, symbol)
    price = _buy_price(form.price_per_unit, latest, symbol)
    holding = _save(session, user_id, asset_type, symbol, form.name, quantity, price, bought_on)
    if latest is not None:
        InvestmentHoldingCRUD().update_market_price(session, holding.id, latest)
    used_latest = not (form.price_per_unit or "").strip()
    return _entry(symbol, quantity, price, latest, used_latest, bought_on)


def latest_unit_price(asset_type: str, symbol: str) -> Decimal | None:
    """Latest price per unit: SEC NAV for funds, market price for stocks.

    Example:
        >>> latest_unit_price("mutual_fund", "K-USA-A(A)")  # doctest: +SKIP
        Decimal('16.6925')
    """
    if asset_type == MUTUAL_FUND:
        return fetch_latest_nav(symbol)
    return fetch_current_price(symbol)


def _asset_identity(symbol: str, asset_type: str) -> tuple[str, str]:
    """Validate the asset type and normalize the symbol for it."""
    kind = asset_type.strip().lower()
    if kind == MUTUAL_FUND:
        try:
            return kind, validate_mutual_fund_symbol(symbol)
        except ValueError as exc:
            raise HoldingEntryError(BAD_FUND_CODE.format(symbol=symbol)) from exc
    if kind != STOCK:
        raise HoldingEntryError(UNKNOWN_ASSET_TYPE.format(value=asset_type))
    canonical = symbol.strip().upper()
    if not canonical.endswith(STOCK_SYMBOL_SUFFIX):
        raise HoldingEntryError(THAI_STOCKS_ONLY.format(symbol=symbol))
    return kind, canonical


def _purchase_date(text: str | None) -> date:
    """Typed purchase date (AD or Buddhist era), today when blank; never in the future."""
    if not (text or "").strip():
        return date.today()
    try:
        bought_on = parse_flexible_date(str(text))
    except ValueError as exc:
        raise HoldingEntryError(BAD_DATE.format(value=text)) from exc
    if bought_on > date.today():
        raise HoldingEntryError(FUTURE_DATE.format(value=bought_on.isoformat()))
    return bought_on


def _save(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    session: Session,
    user_id: str,
    asset_type: str,
    symbol: str,
    name: str,
    quantity: Decimal,
    price: Decimal,
    bought_on: date,
) -> InvestmentHolding:
    """Store the holding; validation errors become user-facing messages."""
    try:
        return add_investment_holding(
            session, user_id, asset_type, symbol, name or symbol, quantity, price, bought_on
        )
    except ValueError as exc:
        raise HoldingEntryError(str(exc)) from exc


def _positive(text: str | None, message: str) -> Decimal:
    """Parse a number that must be above zero."""
    try:
        value = parse_decimal_text(text or "", "value")
    except ValueError as exc:
        raise HoldingEntryError(message.format(value=text)) from exc
    if value <= 0:
        raise HoldingEntryError(message.format(value=text))
    return value


def _buy_price(price_text: str | None, latest_price: Decimal | None, symbol: str) -> Decimal:
    """The typed price, or the latest price when the box was left blank."""
    if (price_text or "").strip():
        return _positive(price_text, BAD_PRICE)
    if latest_price is None:
        raise HoldingEntryError(NO_LATEST_PRICE.format(symbol=symbol))
    return latest_price


def _entry(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    symbol: str,
    quantity: Decimal,
    price: Decimal,
    latest: Decimal | None,
    used_latest: bool,
    bought_on: date,
) -> HoldingEntry:
    """Build the result, computing value and gain/loss when a latest price exists."""
    total_cost = (quantity * price).quantize(_MONEY)
    entry = HoldingEntry(
        symbol=symbol,
        quantity=_plain(quantity),
        price_per_unit=_plain(price),
        used_latest_price=used_latest,
        total_cost=str(total_cost),
        purchase_date=bought_on.isoformat(),
    )
    if latest is None:
        return entry
    current_value = (quantity * latest).quantize(_MONEY)
    gain = current_value - total_cost
    entry.current_price = _plain(latest)
    entry.current_value = str(current_value)
    entry.gain_loss = str(gain)
    if total_cost > 0:  # a tiny lot can round its cost to 0.00
        entry.gain_loss_percent = str((gain / total_cost * _HUNDRED).quantize(_PERCENT))
    return entry


def _plain(value: Decimal) -> str:
    """Decimal without trailing zeros; fund NAVs keep their 4 decimals (16.6925)."""
    return format(value.normalize(), "f")
