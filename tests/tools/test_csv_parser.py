"""Tests for CSV portfolio parser."""

from datetime import date
from decimal import Decimal

import pytest

from finance_ai.tools.csv_parser import (
    parse_csv_row,
    parse_portfolio_csv,
    validate_csv_columns,
)

VALID_CSV = (
    "symbol,asset_type,name,quantity,price_per_unit,purchase_date\n"
    "PTT.BK,stock,PTT,100,35.50,2025-01-15\n"
    "K-EQUITY,mutual_fund,K Equity Fund,500,14.50,2025-03-01\n"
)


class TestValidateCsvColumns:
    """Tests for validate_csv_columns."""

    def test_valid_columns(self) -> None:
        """Accepts header with all required columns."""
        validate_csv_columns(
            ["symbol", "asset_type", "name", "quantity", "price_per_unit", "purchase_date"]
        )

    def test_extra_columns_ok(self) -> None:
        """Accepts header with extra columns."""
        validate_csv_columns(
            ["symbol", "asset_type", "name", "quantity", "price_per_unit", "purchase_date", "notes"]
        )

    def test_case_insensitive(self) -> None:
        """Normalizes column names to lowercase."""
        validate_csv_columns(
            ["Symbol", "Asset_Type", "Name", "Quantity", "Price_Per_Unit", "Purchase_Date"]
        )

    def test_missing_column_raises(self) -> None:
        """Raises ValueError for missing required column."""
        with pytest.raises(ValueError, match="missing required columns"):
            validate_csv_columns(["symbol", "asset_type", "name"])

    def test_empty_header_raises(self) -> None:
        """Raises ValueError for empty header."""
        with pytest.raises(ValueError, match="missing required columns"):
            validate_csv_columns([])


class TestParseCsvRow:
    """Tests for parse_csv_row."""

    def test_valid_row(self) -> None:
        """Parses a complete valid row."""
        row = {
            "symbol": "PTT.BK",
            "asset_type": "stock",
            "name": "PTT",
            "quantity": "100",
            "price_per_unit": "35.50",
            "purchase_date": "2025-01-15",
        }
        result = parse_csv_row(row, 1)
        assert result["symbol"] == "PTT.BK"
        assert result["asset_type"] == "stock"
        assert result["name"] == "PTT"
        assert result["quantity"] == Decimal("100")
        assert result["price_per_unit"] == Decimal("35.50")
        assert result["purchase_date"] == date(2025, 1, 15)

    def test_empty_symbol_raises(self) -> None:
        """Raises ValueError for empty symbol."""
        row = {
            "symbol": "",
            "asset_type": "stock",
            "name": "PTT",
            "quantity": "100",
            "price_per_unit": "35.50",
            "purchase_date": "2025-01-15",
        }
        with pytest.raises(ValueError, match="Row 1: symbol cannot be empty"):
            parse_csv_row(row, 1)

    def test_empty_asset_type_raises(self) -> None:
        """Raises ValueError for empty asset_type."""
        row = {
            "symbol": "PTT.BK",
            "asset_type": "",
            "name": "PTT",
            "quantity": "100",
            "price_per_unit": "35.50",
            "purchase_date": "2025-01-15",
        }
        with pytest.raises(ValueError, match="Row 2: asset_type cannot be empty"):
            parse_csv_row(row, 2)

    def test_invalid_quantity_raises(self) -> None:
        """Raises ValueError for non-numeric quantity."""
        row = {
            "symbol": "PTT.BK",
            "asset_type": "stock",
            "name": "PTT",
            "quantity": "abc",
            "price_per_unit": "35.50",
            "purchase_date": "2025-01-15",
        }
        with pytest.raises(ValueError, match="Cannot parse quantity"):
            parse_csv_row(row, 1)

    def test_invalid_date_raises(self) -> None:
        """Raises ValueError for bad date format."""
        row = {
            "symbol": "PTT.BK",
            "asset_type": "stock",
            "name": "PTT",
            "quantity": "100",
            "price_per_unit": "35.50",
            "purchase_date": "15/01/2025",
        }
        with pytest.raises(ValueError, match="Cannot parse purchase_date"):
            parse_csv_row(row, 1)

    def test_empty_quantity_raises(self) -> None:
        """Raises ValueError for empty quantity."""
        row = {
            "symbol": "PTT.BK",
            "asset_type": "stock",
            "name": "PTT",
            "quantity": "",
            "price_per_unit": "35.50",
            "purchase_date": "2025-01-15",
        }
        with pytest.raises(ValueError, match="quantity cannot be empty"):
            parse_csv_row(row, 1)

    def test_normalizes_asset_type(self) -> None:
        """Normalizes asset_type to lowercase."""
        row = {
            "symbol": "PTT.BK",
            "asset_type": "STOCK",
            "name": "PTT",
            "quantity": "100",
            "price_per_unit": "35.50",
            "purchase_date": "2025-01-15",
        }
        result = parse_csv_row(row, 1)
        assert result["asset_type"] == "stock"


class TestParsePortfolioCsv:
    """Tests for parse_portfolio_csv."""

    def test_valid_csv(self) -> None:
        """Parses multi-row CSV correctly."""
        result = parse_portfolio_csv(VALID_CSV)
        assert len(result) == 2
        assert result[0]["symbol"] == "PTT.BK"
        assert result[1]["symbol"] == "K-EQUITY"

    def test_empty_content_raises(self) -> None:
        """Raises ValueError for empty content."""
        with pytest.raises(ValueError, match="empty"):
            parse_portfolio_csv("")

    def test_header_only_raises(self) -> None:
        """Raises ValueError for header without data."""
        csv_content = "symbol,asset_type,name,quantity,price_per_unit,purchase_date\n"
        with pytest.raises(ValueError, match="no data rows"):
            parse_portfolio_csv(csv_content)

    def test_missing_columns_raises(self) -> None:
        """Raises ValueError for missing required columns."""
        csv_content = "symbol,name\nPTT.BK,PTT\n"
        with pytest.raises(ValueError, match="missing required columns"):
            parse_portfolio_csv(csv_content)

    def test_single_row(self) -> None:
        """Parses single-row CSV."""
        csv_content = (
            "symbol,asset_type,name,quantity,price_per_unit,purchase_date\n"
            "PTT.BK,stock,PTT,100,35.50,2025-01-15\n"
        )
        result = parse_portfolio_csv(csv_content)
        assert len(result) == 1
        assert result[0]["quantity"] == Decimal("100")
