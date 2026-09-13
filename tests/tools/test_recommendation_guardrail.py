"""Tests for the recommendation response guardrail (post-generation checks)."""

from finance_ai.tools.recommendation_constants import HIGH_RISK_INSTRUMENT_KEYWORDS
from finance_ai.tools.recommendation_guardrail import (
    RISK_MISMATCH_WARNING,
    UNBACKED_NUMBERS_WARNING,
    check_recommendation_response,
)

CRYPTO_ANSWER = "แนะนำลงทุนในคริปโต เช่น Bitcoin เพื่อความหลากหลาย"
RETURN_PROMISE_ANSWER = "กองทุนนี้มีผลตอบแทนเฉลี่ย 15% ต่อปี"
CLEAN_ANSWER = "ควรตั้งเป้าหมายการออมและทบทวนงบประมาณรายเดือน"


class TestRiskMismatchCheck:
    """Tests for the risk-profile mismatch check."""

    def test_low_risk_user_with_crypto_gets_warning(self) -> None:
        """Level-1 user and a crypto mention produce a mismatch warning."""
        result = check_recommendation_response(CRYPTO_ANSWER, 1, had_tool_results=True)
        assert "ความเสี่ยง" in result
        assert "ระดับ 1" in result

    def test_level_2_boundary_still_counts_as_low_risk(self) -> None:
        """Level 2 is still within the low-risk threshold."""
        result = check_recommendation_response(CRYPTO_ANSWER, 2, had_tool_results=True)
        assert result == RISK_MISMATCH_WARNING.format(risk_level=2)

    def test_medium_risk_user_with_crypto_no_warning(self) -> None:
        """Level-3 users are not warned about high-risk instruments."""
        result = check_recommendation_response(CRYPTO_ANSWER, 3, had_tool_results=True)
        assert result == ""

    def test_high_risk_user_with_crypto_no_warning(self) -> None:
        """Level-5 users are not warned about high-risk instruments."""
        result = check_recommendation_response(CRYPTO_ANSWER, 5, had_tool_results=True)
        assert result == ""

    def test_no_assessment_with_crypto_no_warning(self) -> None:
        """Without a questionnaire there is no mismatch to detect."""
        result = check_recommendation_response(CRYPTO_ANSWER, None, had_tool_results=True)
        assert result == ""

    def test_case_insensitive_english_keyword(self) -> None:
        """English keywords match regardless of case."""
        result = check_recommendation_response("Consider LEVERAGE products", 1, True)
        assert "ความเสี่ยง" in result


class TestUnbackedNumbersCheck:
    """Tests for the unbacked-return-numbers check."""

    def test_return_percent_without_tool_results_gets_disclaimer(self) -> None:
        """A % return claim with no tool calls that turn gets a disclaimer."""
        result = check_recommendation_response(RETURN_PROMISE_ANSWER, 3, had_tool_results=False)
        assert result == UNBACKED_NUMBERS_WARNING

    def test_return_percent_with_tool_results_no_disclaimer(self) -> None:
        """Tool-backed numbers are trusted and get no disclaimer."""
        result = check_recommendation_response(RETURN_PROMISE_ANSWER, 3, had_tool_results=True)
        assert result == ""

    def test_profit_percent_pattern(self) -> None:
        """Profit phrasing also triggers the disclaimer."""
        result = check_recommendation_response("อาจได้กำไรสูงสุด 20% ต่อปี", 3, False)
        assert result == UNBACKED_NUMBERS_WARNING

    def test_plain_percent_without_return_context_no_disclaimer(self) -> None:
        """Percentages not framed as returns (e.g. savings rate) are fine."""
        result = check_recommendation_response("ควรออมอย่างน้อย 20% ของรายได้", 3, False)
        assert result == ""


class TestCombinedChecks:
    """Tests for answers triggering multiple guardrails."""

    def test_both_violations_produce_both_warnings(self) -> None:
        """Low-risk crypto answer with % promises gets both warnings."""
        answer = "แนะนำคริปโต มีผลตอบแทนสูงสุด 25% ต่อปี"
        result = check_recommendation_response(answer, 1, had_tool_results=False)
        assert "ความเสี่ยง" in result
        assert "รับประกัน" in result

    def test_clean_answer_returns_empty_string(self) -> None:
        """A clean answer produces no warnings."""
        assert check_recommendation_response(CLEAN_ANSWER, 1, had_tool_results=False) == ""


class TestHighRiskKeywordsConstant:
    """Sanity checks for the keyword constant."""

    def test_keywords_nonempty_and_lowercase(self) -> None:
        """Keywords are non-empty and pre-lowercased for matching."""
        assert len(HIGH_RISK_INSTRUMENT_KEYWORDS) > 0
        assert all(k == k.lower() for k in HIGH_RISK_INSTRUMENT_KEYWORDS)
