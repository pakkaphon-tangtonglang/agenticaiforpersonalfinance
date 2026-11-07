"""Tests for thai_gold_service (96.5% gold bar priced per baht-weight)."""

from decimal import Decimal
from unittest.mock import patch

import pytest

from finance_ai.tools.thai_gold_service import (
    THAI_GOLD_SYMBOL,
    is_thai_gold_query,
    thai_gold_price_per_baht_weight,
    thai_gold_search_result,
    validate_thai_gold_symbol,
)

_PRICE = "finance_ai.tools.thai_gold_service.fetch_current_price"


def _prices(gold: str | None, usd_thb: str | None) -> dict[str, Decimal | None]:
    """World gold (USD/oz) and the USD→THB rate, as Yahoo would return them."""
    return {
        "GC=F": None if gold is None else Decimal(gold),
        "USDTHB=X": None if usd_thb is None else Decimal(usd_thb),
    }


class TestThaiGoldPrice:
    """Baht per baht-weight from world gold and the exchange rate."""

    def test_converts_ounce_usd_to_baht_weight(self) -> None:
        """4,000 USD/oz at 32.5 ฿: 15.244 g of 96.5% gold out of a 31.1035 g ounce."""
        with patch(_PRICE, side_effect=_prices("4000", "32.5").get):
            price = thai_gold_price_per_baht_weight()
        assert price == Decimal("61483.75")

    @pytest.mark.parametrize(("gold", "usd_thb"), [(None, "32.5"), ("4000", None)])
    def test_missing_input_gives_none(self, gold: str | None, usd_thb: str | None) -> None:
        """No world price or no rate: no estimate instead of a wrong one."""
        with patch(_PRICE, side_effect=_prices(gold, usd_thb).get):
            assert thai_gold_price_per_baht_weight() is None


class TestGoldQuery:
    """Which searches should offer the Thai gold bar."""

    @pytest.mark.parametrize("query", ["ทองคำ", "ทอง", "ทองคำแท่ง", "Gold", "gold bar", "XAU"])
    def test_gold_words_match(self, query: str) -> None:
        """Thai and English gold words."""
        assert is_thai_gold_query(query) is True

    @pytest.mark.parametrize("query", ["goldman", "ปตท", "Apple", ""])
    def test_other_words_do_not_match(self, query: str) -> None:
        """'goldman' is a bank, not gold."""
        assert is_thai_gold_query(query) is False


class TestGoldSymbol:
    """The one symbol the portfolio stores for gold bars."""

    def test_accepts_symbol_in_any_case(self) -> None:
        """Lower case and spaces are normalized."""
        assert validate_thai_gold_symbol(" thgold965 ") == THAI_GOLD_SYMBOL

    def test_rejects_other_symbol(self) -> None:
        """A futures ticker is not the Thai gold bar."""
        with pytest.raises(ValueError, match=THAI_GOLD_SYMBOL):
            validate_thai_gold_symbol("GC=F")


class TestGoldSearchResult:
    """The search card for the Thai gold bar."""

    def test_card_has_price_and_category(self) -> None:
        """Category 'gold', type GOLD, price as plain text."""
        result = thai_gold_search_result(Decimal("61483.75"))
        assert (result["symbol"], result["type"], result["category"]) == (
            THAI_GOLD_SYMBOL,
            "GOLD",
            "gold",
        )
        assert result["price"] == "61483.75"

    def test_card_without_price(self) -> None:
        """No estimate right now: the card still shows, price None."""
        assert thai_gold_search_result(None)["price"] is None
