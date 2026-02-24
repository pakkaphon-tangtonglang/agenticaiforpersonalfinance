"""Tests for investment tracking constants."""

from decimal import Decimal

from finance_ai.tools.investment_constants import (
    ASSET_TYPES,
    CSV_REQUIRED_COLUMNS,
    INVESTMENT_BUY_TRANSACTION_TYPE,
    INVESTMENT_SELL_TRANSACTION_TYPE,
    MAX_PRICE_PER_UNIT,
    MIN_PRICE_PER_UNIT,
    MIN_QUANTITY,
    MUTUAL_FUND_PREFIXES,
    STOCK_SYMBOL_SUFFIX,
    VALID_ASSET_TYPES,
)


class TestAssetTypes:
    """Tests for ASSET_TYPES constant."""

    def test_contains_stock(self) -> None:
        """Stock type exists with Thai label."""
        assert "stock" in ASSET_TYPES
        assert ASSET_TYPES["stock"] == "หุ้น"

    def test_contains_mutual_fund(self) -> None:
        """Mutual fund type exists with Thai label."""
        assert "mutual_fund" in ASSET_TYPES
        assert ASSET_TYPES["mutual_fund"] == "กองทุนรวม"

    def test_valid_asset_types_matches_keys(self) -> None:
        """VALID_ASSET_TYPES matches ASSET_TYPES keys."""
        assert set(VALID_ASSET_TYPES) == set(ASSET_TYPES.keys())


class TestSymbolConstants:
    """Tests for symbol format constants."""

    def test_stock_symbol_suffix(self) -> None:
        """Stock suffix is .BK for SET/MAI."""
        assert STOCK_SYMBOL_SUFFIX == ".BK"

    def test_mutual_fund_prefixes(self) -> None:
        """Mutual fund prefixes include K-, KT-, SCB-."""
        assert "K-" in MUTUAL_FUND_PREFIXES
        assert "KT-" in MUTUAL_FUND_PREFIXES
        assert "SCB-" in MUTUAL_FUND_PREFIXES

    def test_mutual_fund_prefixes_is_tuple(self) -> None:
        """Prefixes stored as tuple for startswith compatibility."""
        assert isinstance(MUTUAL_FUND_PREFIXES, tuple)


class TestTransactionTypes:
    """Tests for transaction type constants."""

    def test_buy_transaction_type(self) -> None:
        """Buy transaction type is 'buy'."""
        assert INVESTMENT_BUY_TRANSACTION_TYPE == "buy"

    def test_sell_transaction_type(self) -> None:
        """Sell transaction type is 'sell'."""
        assert INVESTMENT_SELL_TRANSACTION_TYPE == "sell"


class TestValidationLimits:
    """Tests for validation limit constants."""

    def test_min_quantity_is_decimal(self) -> None:
        """MIN_QUANTITY is a Decimal."""
        assert isinstance(MIN_QUANTITY, Decimal)

    def test_min_quantity_value(self) -> None:
        """MIN_QUANTITY allows fractional shares."""
        assert MIN_QUANTITY == Decimal("0.0001")

    def test_min_price_is_decimal(self) -> None:
        """MIN_PRICE_PER_UNIT is a Decimal."""
        assert isinstance(MIN_PRICE_PER_UNIT, Decimal)

    def test_max_price_is_decimal(self) -> None:
        """MAX_PRICE_PER_UNIT is a Decimal."""
        assert isinstance(MAX_PRICE_PER_UNIT, Decimal)

    def test_min_price_is_positive(self) -> None:
        """MIN_PRICE_PER_UNIT is positive."""
        assert MIN_PRICE_PER_UNIT > Decimal("0")

    def test_max_price_greater_than_min(self) -> None:
        """MAX_PRICE_PER_UNIT exceeds MIN_PRICE_PER_UNIT."""
        assert MAX_PRICE_PER_UNIT > MIN_PRICE_PER_UNIT


class TestCsvColumns:
    """Tests for CSV import column definitions."""

    def test_required_columns_is_tuple(self) -> None:
        """CSV_REQUIRED_COLUMNS is a tuple."""
        assert isinstance(CSV_REQUIRED_COLUMNS, tuple)

    def test_contains_symbol(self) -> None:
        """Required columns include symbol."""
        assert "symbol" in CSV_REQUIRED_COLUMNS

    def test_contains_asset_type(self) -> None:
        """Required columns include asset_type."""
        assert "asset_type" in CSV_REQUIRED_COLUMNS

    def test_contains_quantity(self) -> None:
        """Required columns include quantity."""
        assert "quantity" in CSV_REQUIRED_COLUMNS

    def test_contains_price_per_unit(self) -> None:
        """Required columns include price_per_unit."""
        assert "price_per_unit" in CSV_REQUIRED_COLUMNS

    def test_contains_purchase_date(self) -> None:
        """Required columns include purchase_date."""
        assert "purchase_date" in CSV_REQUIRED_COLUMNS
