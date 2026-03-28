"""Database-integrated investment portfolio service.

Orchestrates investment operations by querying holdings from the database
and delegating computation to pure calculator functions.
"""

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from finance_ai.database.crud.investment_crud import InvestmentHoldingCRUD
from finance_ai.database.crud.transaction_crud import TransactionCRUD
from finance_ai.database.models.investment_holding import InvestmentHolding
from finance_ai.tools.csv_parser import parse_portfolio_csv
from finance_ai.tools.investment_calculator import (
    HoldingRecord,
    PortfolioSummaryResult,
    summarize_portfolio,
    validate_asset_type,
    validate_price_per_unit,
    validate_quantity,
    validate_symbol,
)
from finance_ai.tools.investment_constants import INVESTMENT_BUY_TRANSACTION_TYPE


def convert_holding_to_record(
    holding: InvestmentHolding,
) -> HoldingRecord:
    """Convert an InvestmentHolding ORM model to a HoldingRecord.

    Args:
        holding: InvestmentHolding model instance.

    Returns:
        HoldingRecord suitable for pure calculator functions.

    Example:
        >>> record = convert_holding_to_record(holding)
        >>> record.symbol
        'PTT.BK'
    """
    return HoldingRecord(
        symbol=holding.symbol,
        name=holding.name or "",
        asset_type=holding.asset_type,
        quantity=holding.quantity,
        average_cost_per_unit=holding.average_cost_per_unit,
        total_cost=holding.total_cost,
        current_price_per_unit=holding.current_price_per_unit,
        current_value=holding.current_value,
        unrealized_gain_loss=holding.unrealized_gain_loss,
    )


def get_user_holdings(
    session: Session,
    user_id: str,
) -> list[HoldingRecord]:
    """Get all investment holdings for a user.

    Args:
        session: Database session.
        user_id: UUID of the user.

    Returns:
        List of HoldingRecord instances.

    Example:
        >>> holdings = get_user_holdings(session, "abc-123")
    """
    crud = InvestmentHoldingCRUD()
    holdings = crud.get_by_user(session, user_id)
    return [convert_holding_to_record(h) for h in holdings]


def get_user_holdings_by_asset_type(
    session: Session,
    user_id: str,
    asset_type: str,
) -> list[HoldingRecord]:
    """Get holdings filtered by asset type.

    Args:
        session: Database session.
        user_id: UUID of the user.
        asset_type: Asset type key (stock, mutual_fund).

    Returns:
        List of matching HoldingRecord instances.

    Raises:
        ValueError: If asset type is invalid.

    Example:
        >>> stocks = get_user_holdings_by_asset_type(session, uid, "stock")
    """
    normalized_type = validate_asset_type(asset_type)
    crud = InvestmentHoldingCRUD()
    holdings = crud.get_by_asset_type(session, user_id, normalized_type)
    return [convert_holding_to_record(h) for h in holdings]


def get_user_holding_by_symbol(
    session: Session,
    user_id: str,
    symbol: str,
) -> HoldingRecord | None:
    """Get a specific holding by symbol.

    Args:
        session: Database session.
        user_id: UUID of the user.
        symbol: Asset symbol (e.g., "PTT.BK").

    Returns:
        HoldingRecord if found, None otherwise.

    Example:
        >>> holding = get_user_holding_by_symbol(session, uid, "PTT.BK")
    """
    crud = InvestmentHoldingCRUD()
    holding = crud.get_by_symbol(session, user_id, symbol.strip().upper())
    if holding is None:
        return None
    return convert_holding_to_record(holding)


def add_investment_holding(  # noqa: PLR0913
    session: Session,
    user_id: str,
    asset_type: str,
    symbol: str,
    name: str,
    quantity: Decimal,
    price_per_unit: Decimal,
    purchase_date: date,
) -> InvestmentHolding:
    """Validate inputs and create an investment holding with buy transaction.

    Args:
        session: Database session.
        user_id: UUID of the user.
        asset_type: Asset type key (stock, mutual_fund).
        symbol: Asset symbol.
        name: Display name.
        quantity: Number of units purchased.
        price_per_unit: Purchase price per unit in THB.
        purchase_date: Date of purchase.

    Returns:
        Created InvestmentHolding instance.

    Raises:
        ValueError: If any input is invalid.

    Example:
        >>> holding = add_investment_holding(
        ...     session, uid, "stock", "PTT.BK", "PTT",
        ...     Decimal("100"), Decimal("35.50"), date(2025, 1, 15),
        ... )
    """
    normalized_type = validate_asset_type(asset_type)
    normalized_symbol = validate_symbol(symbol, normalized_type)
    validate_quantity(quantity)
    validate_price_per_unit(price_per_unit)
    total_cost = (quantity * price_per_unit).quantize(Decimal("0.01"))
    holding_crud = InvestmentHoldingCRUD()
    holding = holding_crud.create(
        session,
        user_id=user_id,
        asset_type=normalized_type,
        symbol=normalized_symbol,
        name=name,
        quantity=quantity,
        average_cost_per_unit=price_per_unit,
        total_cost=total_cost,
        purchase_date=purchase_date,
    )
    _create_buy_transaction(
        session,
        user_id,
        holding.id,
        total_cost,
        quantity,
        price_per_unit,
        purchase_date,
    )
    return holding


def _create_buy_transaction(  # noqa: PLR0913
    session: Session,
    user_id: str,
    holding_id: str,
    amount: Decimal,
    quantity: Decimal,
    price_per_unit: Decimal,
    transaction_date: date,
) -> None:
    """Create a buy transaction for a new holding.

    Args:
        session: Database session.
        user_id: UUID of the user.
        holding_id: UUID of the holding.
        amount: Total transaction amount.
        quantity: Number of units.
        price_per_unit: Price per unit.
        transaction_date: Date of transaction.
    """
    txn_crud = TransactionCRUD()
    txn_crud.create(
        session,
        user_id=user_id,
        holding_id=holding_id,
        transaction_type=INVESTMENT_BUY_TRANSACTION_TYPE,
        amount=amount,
        quantity=quantity,
        price_per_unit=price_per_unit,
        transaction_date=transaction_date,
    )


def import_holdings_from_csv(
    session: Session,
    user_id: str,
    csv_content: str,
) -> list[InvestmentHolding]:
    """Parse CSV content and create multiple holdings.

    Args:
        session: Database session.
        user_id: UUID of the user.
        csv_content: Raw CSV string with header row.

    Returns:
        List of created InvestmentHolding instances.

    Raises:
        ValueError: If CSV format or data is invalid.

    Example:
        >>> holdings = import_holdings_from_csv(session, uid, csv_string)
    """
    parsed_rows = parse_portfolio_csv(csv_content)
    created = []
    for row in parsed_rows:
        holding = add_investment_holding(
            session,
            user_id,
            asset_type=str(row["asset_type"]),
            symbol=str(row["symbol"]),
            name=str(row["name"]),
            quantity=row["quantity"],  # type: ignore[arg-type]
            price_per_unit=row["price_per_unit"],  # type: ignore[arg-type]
            purchase_date=row["purchase_date"],  # type: ignore[arg-type]
        )
        created.append(holding)
    return created


def refresh_portfolio_prices(
    session: Session,
    user_id: str,
) -> int:
    """Fetch latest prices from Bright Data API and update all holdings.

    Args:
        session: Database session.
        user_id: UUID of the user.

    Returns:
        Number of holdings successfully updated.

    Example:
        >>> updated = refresh_portfolio_prices(session, uid)
    """
    from finance_ai.tools.price_client import fetch_multiple_prices  # noqa: PLC0415

    crud = InvestmentHoldingCRUD()
    holdings = crud.get_by_user(session, user_id)
    if not holdings:
        return 0
    symbols = [h.symbol for h in holdings]
    prices = fetch_multiple_prices(symbols)
    updated_count = 0
    for holding in holdings:
        price = prices.get(holding.symbol)
        if price is not None:
            crud.update_market_price(session, holding.id, price)
            updated_count += 1
    return updated_count


def get_portfolio_summary(
    session: Session,
    user_id: str,
) -> PortfolioSummaryResult:
    """Build a complete portfolio summary for a user.

    Fetches all holdings from DB and delegates to pure calculator.

    Args:
        session: Database session.
        user_id: UUID of the user.

    Returns:
        PortfolioSummaryResult with all breakdowns.

    Example:
        >>> summary = get_portfolio_summary(session, uid)
    """
    holdings = get_user_holdings(session, user_id)
    return summarize_portfolio(holdings)
