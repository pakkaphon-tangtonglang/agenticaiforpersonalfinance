"""Investment tracking constants for asset type validation and display.

Contains asset types with Thai labels, symbol format rules,
validation limits, and CSV import column definitions.
"""

from decimal import Decimal

# Asset types: key -> Thai display label
ASSET_TYPES: dict[str, str] = {
    "stock": "หุ้น",
    "mutual_fund": "กองทุนรวม",
}

# Valid asset type keys for validation
VALID_ASSET_TYPES: list[str] = list(ASSET_TYPES.keys())

# Stock symbol format: SET/MAI listed stocks end with .BK
STOCK_SYMBOL_SUFFIX: str = ".BK"

# Mutual fund symbol prefixes (Thai fund houses)
MUTUAL_FUND_PREFIXES: tuple[str, ...] = ("K-", "KT-", "SCB-")

# Transaction types for investment operations
INVESTMENT_BUY_TRANSACTION_TYPE: str = "buy"
INVESTMENT_SELL_TRANSACTION_TYPE: str = "sell"

# Minimum quantity (supports fractional shares/units)
MIN_QUANTITY: Decimal = Decimal("0.0001")

# Price per unit validation range (THB)
MIN_PRICE_PER_UNIT: Decimal = Decimal("0.01")
MAX_PRICE_PER_UNIT: Decimal = Decimal("100000000.00")

# Required columns for CSV portfolio import
CSV_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "asset_type",
    "name",
    "quantity",
    "price_per_unit",
    "purchase_date",
)
