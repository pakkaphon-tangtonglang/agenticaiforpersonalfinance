"""Tests for market data constants."""

from finance_ai.tools.market_data_constants import (
    CURRENCY_CODE_LENGTH,
    DASHBOARD_PRICE_FIELDS,
    DEFAULT_CURRENCY,
    DIVIDEND_YIELD_TO_PERCENT,
    FOREX_SYMBOL_TEMPLATE,
    NEWS_NOT_FOUND_MESSAGE,
    NEWS_USER_AGENT,
    SUPPORTED_CURRENCIES,
)


class TestSupportedCurrencies:
    """Tests for SUPPORTED_CURRENCIES constant."""

    def test_is_tuple(self) -> None:
        """Supported currencies should be an immutable tuple."""
        assert isinstance(SUPPORTED_CURRENCIES, tuple)

    def test_contains_thb(self) -> None:
        """THB must be included for Thai users."""
        assert "THB" in SUPPORTED_CURRENCIES

    def test_contains_usd(self) -> None:
        """USD must be included as the global reserve currency."""
        assert "USD" in SUPPORTED_CURRENCIES

    def test_all_codes_are_three_chars(self) -> None:
        """All currency codes must be 3-character uppercase strings."""
        for code in SUPPORTED_CURRENCIES:
            assert len(code) == CURRENCY_CODE_LENGTH
            assert code == code.upper()


class TestForexSymbolTemplate:
    """Tests for FOREX_SYMBOL_TEMPLATE constant."""

    def test_format_produces_valid_symbol(self) -> None:
        """Template should produce Yahoo Finance forex format."""
        result = FOREX_SYMBOL_TEMPLATE.format(from_currency="USD", to_currency="THB")
        assert result == "USDTHB=X"


class TestNewsConstants:
    """Tests for news-related constants."""

    def test_not_found_message_contains_placeholder(self) -> None:
        """Message template must have {symbol} placeholder."""
        assert "{symbol}" in NEWS_NOT_FOUND_MESSAGE

    def test_user_agent_is_nonempty_string(self) -> None:
        """User agent must be a non-empty string."""
        assert isinstance(NEWS_USER_AGENT, str)
        assert len(NEWS_USER_AGENT) > 0


class TestDashboardConstants:
    """Tests for dashboard-related constants."""

    def test_price_fields_contains_current_price(self) -> None:
        """Must include currentPrice as primary price field."""
        assert "currentPrice" in DASHBOARD_PRICE_FIELDS

    def test_price_fields_contains_fallback(self) -> None:
        """Must include regularMarketPrice as fallback."""
        assert "regularMarketPrice" in DASHBOARD_PRICE_FIELDS

    def test_dividend_yield_multiplier(self) -> None:
        """Dividend yield multiplier should be 100 (fraction → %)."""
        assert DIVIDEND_YIELD_TO_PERCENT == 100


class TestMiscConstants:
    """Tests for other constants."""

    def test_default_currency_is_thb(self) -> None:
        """Default currency should be THB for Thai users."""
        assert DEFAULT_CURRENCY == "THB"

    def test_currency_code_length(self) -> None:
        """Currency code length should be 3."""
        assert CURRENCY_CODE_LENGTH == 3
