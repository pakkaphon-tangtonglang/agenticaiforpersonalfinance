"""Tests for finance_ai.agents.llm_error_messages."""

from finance_ai.agents.llm_error_messages import (
    LLM_GENERIC_ERROR_MESSAGE,
    LLM_QUOTA_MESSAGE,
    describe_llm_error,
)


class TestDescribeLlmError:
    """Tests for describe_llm_error."""

    def test_quota_error_returns_quota_message(self) -> None:
        """A 429 RESOURCE_EXHAUSTED error maps to the quota message."""
        error = RuntimeError("429 RESOURCE_EXHAUSTED. Quota exceeded")
        assert describe_llm_error(error) == LLM_QUOTA_MESSAGE

    def test_other_error_returns_generic_message(self) -> None:
        """Any other error maps to the generic message."""
        assert describe_llm_error(RuntimeError("LLM down")) == LLM_GENERIC_ERROR_MESSAGE

    def test_messages_are_thai_and_distinct(self) -> None:
        """The two messages differ so users can tell the cases apart."""
        assert "โควตา" in LLM_QUOTA_MESSAGE
        assert "ข้อผิดพลาด" in LLM_GENERIC_ERROR_MESSAGE
        assert LLM_QUOTA_MESSAGE != LLM_GENERIC_ERROR_MESSAGE
