"""Tests for market data constants."""

from finance_ai.tools.market_data_constants import (
    CURRENCY_CODE_LENGTH,
    DEFAULT_CURRENCY,
    DIVIDEND_YIELD_TO_PERCENT,
    NEWS_NOT_FOUND_MESSAGE,
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


class TestNewsConstants:
    """Tests for news-related constants."""

    def test_not_found_message_contains_placeholder(self) -> None:
        """Message template must have {symbol} placeholder."""
        assert "{symbol}" in NEWS_NOT_FOUND_MESSAGE


class TestMiscConstants:
    """Tests for other constants."""

    def test_default_currency_is_thb(self) -> None:
        """Default currency should be THB for Thai users."""
        assert DEFAULT_CURRENCY == "THB"

    def test_currency_code_length(self) -> None:
        """Currency code length should be 3."""
        assert CURRENCY_CODE_LENGTH == 3

    def test_dividend_yield_multiplier(self) -> None:
        """Dividend yield multiplier should be 100 (fraction → %)."""
        assert DIVIDEND_YIELD_TO_PERCENT == 100
