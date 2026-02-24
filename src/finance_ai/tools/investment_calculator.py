"""Pure investment calculation functions for portfolio analysis.

All functions accept typed values directly with no database dependency,
making them easy to test and reuse across different contexts.
"""

from decimal import Decimal

from pydantic import BaseModel, Field

from finance_ai.tools.investment_constants import (
    ASSET_TYPES,
    MAX_PRICE_PER_UNIT,
    MIN_PRICE_PER_UNIT,
    MIN_QUANTITY,
    MUTUAL_FUND_PREFIXES,
    STOCK_SYMBOL_SUFFIX,
    VALID_ASSET_TYPES,
)


class HoldingRecord(BaseModel):
    """A single investment holding for calculation purposes.

    Attributes:
        symbol: Asset symbol (e.g., "PTT.BK").
        name: Display name of the asset.
        asset_type: Asset type key (stock, mutual_fund).
        quantity: Number of units held.
        average_cost_per_unit: Average purchase price per unit.
        total_cost: Total cost basis.
        current_price_per_unit: Current market price per unit (None if unavailable).
        current_value: Current market value (None if price unavailable).
        unrealized_gain_loss: Unrealized gain/loss (None if price unavailable).
    """

    symbol: str
    name: str = ""
    asset_type: str
    quantity: Decimal
    average_cost_per_unit: Decimal
    total_cost: Decimal
    current_price_per_unit: Decimal | None = None
    current_value: Decimal | None = None
    unrealized_gain_loss: Decimal | None = None


class HoldingSummary(BaseModel):
    """Summary for a single holding with gain/loss and allocation.

    Attributes:
        symbol: Asset symbol.
        name: Display name.
        asset_type: Asset type key.
        asset_type_label: Thai display label.
        quantity: Number of units.
        average_cost_per_unit: Average cost per unit.
        total_cost: Total cost basis.
        current_price_per_unit: Current price (None if unavailable).
        current_value: Current value (None if unavailable).
        unrealized_gain_loss: Unrealized gain/loss (None if unavailable).
        gain_loss_percentage: Gain/loss as percentage (None if unavailable).
        allocation_percentage: Holding's share of total portfolio cost.
    """

    symbol: str
    name: str = ""
    asset_type: str
    asset_type_label: str
    quantity: Decimal
    average_cost_per_unit: Decimal
    total_cost: Decimal
    current_price_per_unit: Decimal | None = None
    current_value: Decimal | None = None
    unrealized_gain_loss: Decimal | None = None
    gain_loss_percentage: Decimal | None = None
    allocation_percentage: Decimal = Field(default=Decimal("0.0000"))


class AssetTypeSummary(BaseModel):
    """Summary of holdings grouped by asset type.

    Attributes:
        asset_type: Asset type key.
        asset_type_label: Thai display label.
        total_cost: Sum of cost basis for this type.
        current_value: Sum of current values (None if any holding lacks price).
        unrealized_gain_loss: Aggregate gain/loss (None if value unavailable).
        holding_count: Number of holdings of this type.
        allocation_percentage: This type's share of total portfolio cost.
    """

    asset_type: str
    asset_type_label: str
    total_cost: Decimal
    current_value: Decimal | None = None
    unrealized_gain_loss: Decimal | None = None
    holding_count: int
    allocation_percentage: Decimal = Field(default=Decimal("0.0000"))


class PortfolioSummaryResult(BaseModel):
    """Complete portfolio summary with breakdowns.

    Attributes:
        total_cost: Sum of all holdings' cost basis.
        total_current_value: Sum of all current values (None if any unavailable).
        total_unrealized_gain_loss: Aggregate gain/loss (None if value unavailable).
        total_gain_loss_percentage: Overall return percentage (None if unavailable).
        holdings: Per-holding summaries.
        asset_type_breakdown: Per-asset-type summaries.
        holding_count: Total number of holdings.
    """

    total_cost: Decimal
    total_current_value: Decimal | None = None
    total_unrealized_gain_loss: Decimal | None = None
    total_gain_loss_percentage: Decimal | None = None
    holdings: list[HoldingSummary]
    asset_type_breakdown: list[AssetTypeSummary]
    holding_count: int


def validate_asset_type(asset_type: str) -> str:
    """Validate and normalize an asset type key.

    Args:
        asset_type: Asset type to validate.

    Returns:
        Normalized (lowercase) asset type key.

    Raises:
        ValueError: If asset type is not recognized.

    Example:
        >>> validate_asset_type("Stock")
        'stock'
    """
    normalized = asset_type.strip().lower()
    if normalized not in VALID_ASSET_TYPES:
        raise ValueError(
            f"Unknown asset type: '{asset_type}'. " f"Valid types: {VALID_ASSET_TYPES}"
        )
    return normalized


def validate_stock_symbol(symbol: str) -> str:
    """Validate a SET/MAI stock symbol format.

    Args:
        symbol: Stock symbol to validate (must end with .BK).

    Returns:
        Normalized (uppercase, stripped) symbol.

    Raises:
        ValueError: If symbol is empty or missing .BK suffix.

    Example:
        >>> validate_stock_symbol("ptt.bk")
        'PTT.BK'
    """
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("Stock symbol cannot be empty.")
    if not normalized.endswith(STOCK_SYMBOL_SUFFIX):
        raise ValueError(
            f"Stock symbol must end with '{STOCK_SYMBOL_SUFFIX}'. " f"Received: '{symbol}'"
        )
    return normalized


def validate_mutual_fund_symbol(symbol: str) -> str:
    """Validate a Thai mutual fund symbol format.

    Args:
        symbol: Fund symbol to validate (must start with K-, KT-, or SCB-).

    Returns:
        Normalized (uppercase, stripped) symbol.

    Raises:
        ValueError: If symbol is empty or missing valid prefix.

    Example:
        >>> validate_mutual_fund_symbol("k-equity")
        'K-EQUITY'
    """
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValueError("Mutual fund symbol cannot be empty.")
    if not normalized.startswith(MUTUAL_FUND_PREFIXES):
        raise ValueError(
            f"Mutual fund symbol must start with one of {MUTUAL_FUND_PREFIXES}. "
            f"Received: '{symbol}'"
        )
    return normalized


def validate_symbol(symbol: str, asset_type: str) -> str:
    """Validate a symbol based on its asset type.

    Args:
        symbol: Asset symbol to validate.
        asset_type: Asset type key (stock or mutual_fund).

    Returns:
        Normalized symbol.

    Raises:
        ValueError: If symbol format is invalid for the asset type.

    Example:
        >>> validate_symbol("PTT.BK", "stock")
        'PTT.BK'
    """
    if asset_type == "stock":
        return validate_stock_symbol(symbol)
    if asset_type == "mutual_fund":
        return validate_mutual_fund_symbol(symbol)
    return symbol.strip().upper()


def validate_quantity(quantity: Decimal) -> None:
    """Validate that quantity meets minimum requirement.

    Args:
        quantity: Quantity to validate.

    Raises:
        ValueError: If quantity is below minimum.

    Example:
        >>> validate_quantity(Decimal("100"))
    """
    if quantity < MIN_QUANTITY:
        raise ValueError(f"Quantity must be at least {MIN_QUANTITY}. " f"Received: {quantity}")


def validate_price_per_unit(price: Decimal) -> None:
    """Validate that price per unit is within allowed range.

    Args:
        price: Price to validate.

    Raises:
        ValueError: If price is below minimum or above maximum.

    Example:
        >>> validate_price_per_unit(Decimal("35.50"))
    """
    if price < MIN_PRICE_PER_UNIT:
        raise ValueError(
            f"Price per unit must be at least {MIN_PRICE_PER_UNIT} THB. " f"Received: {price} THB"
        )
    if price > MAX_PRICE_PER_UNIT:
        raise ValueError(
            f"Price per unit exceeds maximum of {MAX_PRICE_PER_UNIT} THB. " f"Received: {price} THB"
        )


def calculate_total_cost(
    quantity: Decimal,
    average_cost_per_unit: Decimal,
) -> Decimal:
    """Calculate total cost basis.

    Args:
        quantity: Number of units.
        average_cost_per_unit: Average cost per unit.

    Returns:
        Total cost quantized to 2 decimal places.

    Example:
        >>> calculate_total_cost(Decimal("100"), Decimal("35.50"))
        Decimal('3550.00')
    """
    return (quantity * average_cost_per_unit).quantize(Decimal("0.01"))


def calculate_current_value(
    quantity: Decimal,
    current_price: Decimal,
) -> Decimal:
    """Calculate current market value.

    Args:
        quantity: Number of units.
        current_price: Current price per unit.

    Returns:
        Current value quantized to 2 decimal places.

    Example:
        >>> calculate_current_value(Decimal("100"), Decimal("42.50"))
        Decimal('4250.00')
    """
    return (quantity * current_price).quantize(Decimal("0.01"))


def calculate_unrealized_gain_loss(
    current_value: Decimal,
    total_cost: Decimal,
) -> Decimal:
    """Calculate unrealized gain or loss.

    Args:
        current_value: Current market value.
        total_cost: Total cost basis.

    Returns:
        Unrealized gain (positive) or loss (negative).

    Example:
        >>> calculate_unrealized_gain_loss(Decimal("4250.00"), Decimal("3550.00"))
        Decimal('700.00')
    """
    return current_value - total_cost


def calculate_gain_loss_percentage(
    unrealized_gain_loss: Decimal,
    total_cost: Decimal,
) -> Decimal:
    """Calculate gain/loss as a percentage of cost.

    Args:
        unrealized_gain_loss: Unrealized gain or loss amount.
        total_cost: Total cost basis.

    Returns:
        Percentage with 4 decimal places (e.g., 19.7183 = 19.7183%).

    Example:
        >>> calculate_gain_loss_percentage(Decimal("700"), Decimal("3550"))
        Decimal('19.7183')
    """
    if total_cost == Decimal("0"):
        return Decimal("0.0000")
    return ((unrealized_gain_loss / total_cost) * 100).quantize(Decimal("0.0001"))


def calculate_allocation_percentage(
    holding_value: Decimal,
    portfolio_total: Decimal,
) -> Decimal:
    """Calculate a holding's allocation as percentage of portfolio.

    Args:
        holding_value: Value of the holding (cost or market value).
        portfolio_total: Total portfolio value.

    Returns:
        Percentage with 4 decimal places.

    Example:
        >>> calculate_allocation_percentage(Decimal("3550"), Decimal("10000"))
        Decimal('35.5000')
    """
    if portfolio_total == Decimal("0"):
        return Decimal("0.0000")
    return ((holding_value / portfolio_total) * 100).quantize(Decimal("0.0001"))


def calculate_portfolio_total_cost(
    holdings: list[HoldingRecord],
) -> Decimal:
    """Sum total cost across all holdings.

    Args:
        holdings: List of holding records.

    Returns:
        Sum of all total_cost values.

    Example:
        >>> calculate_portfolio_total_cost([])
        Decimal('0')
    """
    return sum((h.total_cost for h in holdings), Decimal("0"))


def calculate_portfolio_current_value(
    holdings: list[HoldingRecord],
) -> Decimal | None:
    """Sum current value across all holdings.

    Args:
        holdings: List of holding records.

    Returns:
        Sum of all current_value, or None if any holding lacks a price.

    Example:
        >>> calculate_portfolio_current_value([])
    """
    if not holdings:
        return None
    values = [h.current_value for h in holdings]
    if any(v is None for v in values):
        return None
    return sum((v for v in values if v is not None), Decimal("0"))


def get_active_asset_types(
    holdings: list[HoldingRecord],
) -> list[str]:
    """Return sorted unique asset types present in holdings.

    Args:
        holdings: List of holding records.

    Returns:
        Sorted list of asset type keys.

    Example:
        >>> get_active_asset_types([])
        []
    """
    return sorted({h.asset_type for h in holdings})


def build_holding_summary(
    holding: HoldingRecord,
    portfolio_total_cost: Decimal,
) -> HoldingSummary:
    """Build a summary for one holding with allocation.

    Args:
        holding: The holding record.
        portfolio_total_cost: Total cost of the entire portfolio.

    Returns:
        HoldingSummary with calculated percentages.

    Example:
        >>> summary = build_holding_summary(holding, Decimal("10000"))
    """
    gain_loss_pct = None
    if holding.unrealized_gain_loss is not None and holding.total_cost > 0:
        gain_loss_pct = calculate_gain_loss_percentage(
            holding.unrealized_gain_loss, holding.total_cost
        )
    allocation = calculate_allocation_percentage(holding.total_cost, portfolio_total_cost)
    label = ASSET_TYPES.get(holding.asset_type, holding.asset_type)
    return HoldingSummary(
        symbol=holding.symbol,
        name=holding.name,
        asset_type=holding.asset_type,
        asset_type_label=label,
        quantity=holding.quantity,
        average_cost_per_unit=holding.average_cost_per_unit,
        total_cost=holding.total_cost,
        current_price_per_unit=holding.current_price_per_unit,
        current_value=holding.current_value,
        unrealized_gain_loss=holding.unrealized_gain_loss,
        gain_loss_percentage=gain_loss_pct,
        allocation_percentage=allocation,
    )


def build_asset_type_summary(
    holdings: list[HoldingRecord],
    asset_type: str,
    portfolio_total_cost: Decimal,
) -> AssetTypeSummary:
    """Build an aggregate summary for one asset type.

    Args:
        holdings: All holdings (will filter by asset_type).
        asset_type: Asset type key to summarize.
        portfolio_total_cost: Total portfolio cost for allocation.

    Returns:
        AssetTypeSummary with aggregated values.

    Example:
        >>> summary = build_asset_type_summary(holdings, "stock", Decimal("10000"))
    """
    filtered = [h for h in holdings if h.asset_type == asset_type]
    type_cost = sum((h.total_cost for h in filtered), Decimal("0"))
    values = [h.current_value for h in filtered]
    type_value = (
        None
        if any(v is None for v in values)
        else sum((v for v in values if v is not None), Decimal("0"))
    )
    type_gain_loss = None
    if type_value is not None:
        type_gain_loss = type_value - type_cost
    allocation = calculate_allocation_percentage(type_cost, portfolio_total_cost)
    label = ASSET_TYPES.get(asset_type, asset_type)
    return AssetTypeSummary(
        asset_type=asset_type,
        asset_type_label=label,
        total_cost=type_cost,
        current_value=type_value,
        unrealized_gain_loss=type_gain_loss,
        holding_count=len(filtered),
        allocation_percentage=allocation,
    )


def summarize_portfolio(
    holdings: list[HoldingRecord],
) -> PortfolioSummaryResult:
    """Build complete portfolio summary with all breakdowns.

    Args:
        holdings: List of all holding records for the user.

    Returns:
        PortfolioSummaryResult with per-holding and per-type summaries.

    Example:
        >>> result = summarize_portfolio([])
        >>> result.holding_count
        0
    """
    total_cost = calculate_portfolio_total_cost(holdings)
    total_value = calculate_portfolio_current_value(holdings)
    total_gain_loss = None
    total_gain_loss_pct = None
    if total_value is not None:
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = calculate_gain_loss_percentage(total_gain_loss, total_cost)
    holding_summaries = [build_holding_summary(h, total_cost) for h in holdings]
    active_types = get_active_asset_types(holdings)
    type_breakdowns = [build_asset_type_summary(holdings, at, total_cost) for at in active_types]
    return PortfolioSummaryResult(
        total_cost=total_cost,
        total_current_value=total_value,
        total_unrealized_gain_loss=total_gain_loss,
        total_gain_loss_percentage=total_gain_loss_pct,
        holdings=holding_summaries,
        asset_type_breakdown=type_breakdowns,
        holding_count=len(holdings),
    )
