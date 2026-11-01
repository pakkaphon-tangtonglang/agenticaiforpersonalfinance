"""Tests for fund_query_expander (LLM suggests fund-name search terms)."""

from typing import Any
from unittest.mock import MagicMock

from finance_ai.tools.fund_query_expander import FundSearchTerms, clean_terms, expand_fund_query


def model_returning(result: Any) -> MagicMock:
    """A fake chat model whose structured output returns `result`."""
    model = MagicMock()
    model.with_structured_output.return_value.invoke.return_value = result
    return model


def test_returns_cleaned_terms() -> None:
    """Terms are trimmed, de-duplicated and capped at five."""
    model = model_returning(
        FundSearchTerms(terms=[" US ", "us", "S&P 500", "x", "A", "B1", "C2", "D3", "E4"])
    )
    assert expand_fund_query("หุ้นอเมริกา", lambda: model) == ["US", "S&P 500", "B1", "C2", "D3"]
    prompt = model.with_structured_output.return_value.invoke.call_args.args[0]
    assert "หุ้นอเมริกา" in prompt


def test_gibberish_returns_nothing() -> None:
    """meaningful=False → no terms, so the page asks instead of guessing."""
    model = model_returning(FundSearchTerms(meaningful=False, terms=["QQQ"]))
    assert not expand_fund_query("zzqqxx", lambda: model)


def test_model_failure_returns_nothing() -> None:
    """Any model error means no extra terms."""
    model = MagicMock()
    model.with_structured_output.side_effect = RuntimeError("quota")
    assert not expand_fund_query("ทองคำ", lambda: model)


def test_unexpected_output_returns_nothing() -> None:
    """A non-schema answer is ignored."""
    assert not expand_fund_query("ทองคำ", lambda: model_returning({"terms": ["GOLD"]}))


def test_clean_terms_drops_query_and_lengths() -> None:
    """The original query and too short/long terms are removed."""
    assert clean_terms(["ทองคำ", "GOLD", "G", "X" * 41], "ทองคำ") == ["GOLD"]
