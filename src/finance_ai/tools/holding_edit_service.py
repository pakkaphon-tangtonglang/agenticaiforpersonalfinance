"""Edit or delete one portfolio row (asset page ✎ / 🗑 buttons).

Editing corrects the quantity and the average cost per unit. The total cost,
the value at the stored latest price and the gain/loss are recalculated, and
the holding's buy transaction is kept in step. Deleting removes the holding
and (by cascade) its transactions. A holding that belongs to someone else is
treated exactly like a missing one.
"""

from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy.orm import Session

from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.tools.holding_entry_service import BAD_QUANTITY, HoldingEntryError
from finance_ai.tools.input_normalizer import parse_decimal_text

MONEY = Decimal("0.01")
BAD_COST = "ต้นทุนต่อหน่วยต้องเป็นตัวเลขมากกว่า 0 (ได้รับ: '{value}')"


class HoldingNotFoundError(LookupError):
    """No holding with this id for this user."""


class HoldingEditForm(BaseModel):
    """The corrected numbers typed into the row.

    Example:
        >>> HoldingEditForm(quantity="600", price_per_unit="40").quantity
        '600'
    """

    quantity: str
    price_per_unit: str


def update_holding(
    session: Session, user_id: str, holding_id: str, form: HoldingEditForm
) -> InvestmentHolding:
    """Correct a holding's quantity and average cost.

    Args:
        session: Database session.
        user_id: Owner of the holding.
        holding_id: Holding to change.
        form: New quantity and cost per unit (Thai number formats accepted).

    Returns:
        The updated holding.

    Raises:
        HoldingNotFoundError: The holding does not exist or is not the user's.
        HoldingEntryError: A number is missing, unreadable or not above zero.

    Example:
        >>> form = HoldingEditForm(quantity="600", price_per_unit="40")
        >>> update_holding(session, uid, hid, form)  # doctest: +SKIP
    """
    quantity = _positive(form.quantity, BAD_QUANTITY)
    cost_per_unit = _positive(form.price_per_unit, BAD_COST)
    holding = _owned_holding(session, user_id, holding_id)
    holding.quantity = quantity
    holding.average_cost_per_unit = cost_per_unit
    holding.total_cost = (quantity * cost_per_unit).quantize(MONEY)
    _revalue(holding)
    _sync_buy_transaction(holding)
    session.commit()
    session.refresh(holding)
    return holding


def delete_holding(session: Session, user_id: str, holding_id: str) -> None:
    """Delete a holding and its transactions.

    Raises:
        HoldingNotFoundError: The holding does not exist or is not the user's.

    Example:
        >>> delete_holding(session, uid, hid)  # doctest: +SKIP
    """
    session.delete(_owned_holding(session, user_id, holding_id))
    session.commit()


def _owned_holding(session: Session, user_id: str, holding_id: str) -> InvestmentHolding:
    """The user's holding, or HoldingNotFoundError."""
    holding = session.get(InvestmentHolding, holding_id)
    if holding is None or holding.user_id != user_id:
        raise HoldingNotFoundError(f"Holding not found: {holding_id}")
    return holding


def _positive(text: str, message: str) -> Decimal:
    """Parse a number that must be above zero."""
    try:
        value = parse_decimal_text(text or "", "value")
    except ValueError as exc:
        raise HoldingEntryError(message.format(value=text)) from exc
    if value <= 0:
        raise HoldingEntryError(message.format(value=text))
    return value


def _revalue(holding: InvestmentHolding) -> None:
    """Value and gain at the stored latest price (left empty without one)."""
    price = holding.current_price_per_unit
    if price is None:
        holding.current_value = None
        holding.unrealized_gain_loss = None
        return
    holding.current_value = (holding.quantity * price).quantize(MONEY)
    holding.unrealized_gain_loss = holding.current_value - holding.total_cost


def _sync_buy_transaction(holding: InvestmentHolding) -> None:
    """Keep a single linked buy record equal to the corrected purchase."""
    if len(holding.transactions) != 1:
        return
    buy = holding.transactions[0]
    buy.quantity = holding.quantity
    buy.price_per_unit = holding.average_cost_per_unit
    buy.amount = holding.total_cost
