"""Tests for market data Pydantic models."""

from decimal import Decimal

import pytest

from finance_ai.tools.market_data_models import (
    CurrencyConversionResult,
    FinanceNewsResult,
    StockDashboardResult,
)


class TestStockDashboardResult:
    """Tests for StockDashboardResult model."""

    def test_create_with_all_fields(self) -> None:
        """Should accept all fields with Decimal values."""
        result = StockDashboardResult(
            name="PTT Public Company Limited",
            current_price=Decimal("35.50"),
            currency="THB",
            fifty_two_week_high=Decimal("42.00"),
            fifty_two_week_low=Decimal("28.00"),
            pe_ratio=Decimal("12.5"),
            market_cap=Decimal("1000000000"),
            dividend_yield_percent=Decimal("3.50"),
            analyst_target_price=Decimal("40.00"),
            recommendation="buy",
        )
        assert result.name == "PTT Public Company Limited"
        assert result.current_price == Decimal("35.50")
        assert result.currency == "THB"

    def test_defaults_to_none_for_optional_fields(self) -> None:
        """Optional fields should default to None."""
        result = StockDashboardResult()
        assert result.name is None
        assert result.current_price is None
        assert result.pe_ratio is None
        assert result.market_cap is None
        assert result.recommendation is None

    def test_dividend_yield_defaults_to_zero(self) -> None:
        """Dividend yield should default to 0, not None."""
        result = StockDashboardResult()
        assert result.dividend_yield_percent == Decimal("0")

    def test_model_dump_returns_dict(self) -> None:
        """model_dump should return a serializable dict."""
        result = StockDashboardResult(name="AAPL", current_price=Decimal("180.00"))
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["name"] == "AAPL"
        assert data["current_price"] == Decimal("180.00")


class TestCurrencyConversionResult:
    """Tests for CurrencyConversionResult model."""

    def test_create_valid_conversion(self) -> None:
        """Should accept all required fields."""
        result = CurrencyConversionResult(
            from_currency="USD",
            to_currency="THB",
            amount=Decimal("100"),
            exchange_rate=Decimal("34.50"),
            converted_amount=Decimal("3450.00"),
        )
        assert result.from_currency == "USD"
        assert result.to_currency == "THB"
        assert result.converted_amount == Decimal("3450.00")

    def test_requires_all_fields(self) -> None:
        """Should raise ValidationError if required fields are missing."""
        with pytest.raises(Exception):
            CurrencyConversionResult()  # type: ignore[call-arg]

    def test_model_dump_has_all_keys(self) -> None:
        """model_dump should include all 5 fields."""
        result = CurrencyConversionResult(
            from_currency="EUR",
            to_currency="THB",
            amount=Decimal("50"),
            exchange_rate=Decimal("37.80"),
            converted_amount=Decimal("1890.00"),
        )
        data = result.model_dump()
        expected_keys = {
            "from_currency",
            "to_currency",
            "amount",
            "exchange_rate",
            "converted_amount",
        }
        assert set(data.keys()) == expected_keys

    def test_decimal_precision_preserved(self) -> None:
        """Decimal precision should be maintained."""
        result = CurrencyConversionResult(
            from_currency="JPY",
            to_currency="THB",
            amount=Decimal("10000"),
            exchange_rate=Decimal("0.2345"),
            converted_amount=Decimal("2345.0000"),
        )
        assert result.exchange_rate == Decimal("0.2345")


class TestFinanceNewsResult:
    """Tests for FinanceNewsResult model."""

    def test_create_with_news(self) -> None:
        """Should store news content and mark has_news=True."""
        result = FinanceNewsResult(
            symbol="PTT.BK",
            news_content="PTT announces quarterly results...",
            has_news=True,
        )
        assert result.symbol == "PTT.BK"
        assert result.has_news is True
        assert "PTT" in result.news_content

    def test_create_without_news(self) -> None:
        """Should store empty content and mark has_news=False."""
        result = FinanceNewsResult(
            symbol="UNKNOWN",
            news_content="ไม่พบข่าว",
            has_news=False,
        )
        assert result.has_news is False

    def test_requires_all_fields(self) -> None:
        """Should raise ValidationError if fields are missing."""
        with pytest.raises(Exception):
            FinanceNewsResult()  # type: ignore[call-arg]

    def test_model_dump_returns_dict(self) -> None:
        """model_dump should return a serializable dict."""
        result = FinanceNewsResult(symbol="AAPL", news_content="Apple news...", has_news=True)
        data = result.model_dump()
        assert data["symbol"] == "AAPL"
        assert data["has_news"] is True
