"""Tests for crypto_service (Yahoo crypto pairs like BTC-USD in the portfolio)."""

import pytest

from finance_ai.tools.crypto_service import is_crypto_symbol, validate_crypto_symbol


class TestIsCryptoSymbol:
    """Yahoo crypto pairs end in -USD (or -USDT)."""

    @pytest.mark.parametrize("symbol", ["BTC-USD", "eth-usd", "SOL-USDT"])
    def test_crypto_pairs(self, symbol: str) -> None:
        """Coin + quote currency."""
        assert is_crypto_symbol(symbol) is True

    @pytest.mark.parametrize("symbol", ["AAPL", "PTT.BK", "GC=F", "-USD", "BTC-THB"])
    def test_not_crypto(self, symbol: str) -> None:
        """Stocks, futures and pairs Yahoo does not price in USD."""
        assert is_crypto_symbol(symbol) is False


class TestValidateCryptoSymbol:
    """Normalize or reject a crypto symbol."""

    def test_normalizes(self) -> None:
        """Upper case, no spaces."""
        assert validate_crypto_symbol(" btc-usd ") == "BTC-USD"

    def test_rejects_stock(self) -> None:
        """A stock ticker gets an example of the right format."""
        with pytest.raises(ValueError, match="BTC-USD"):
            validate_crypto_symbol("AAPL")
