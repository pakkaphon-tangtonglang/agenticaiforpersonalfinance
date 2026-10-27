"""Tests for retrying embedding calls on provider rate limits."""

import pytest

from finance_ai.rag.rate_limit_retry import (
    MAX_RETRY_DELAY_SECONDS,
    call_with_rate_limit_retry,
    is_rate_limit_error,
    retry_delay_seconds,
)

GOOGLE_429 = RuntimeError(
    "Error embedding content (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "Quota exceeded ... Please retry in 27.551161481s."
)


class TestIsRateLimitError:
    """Tests for is_rate_limit_error."""

    @pytest.mark.parametrize(
        "message",
        ["429 Too Many Requests", "RESOURCE_EXHAUSTED", "Rate limit reached for requests"],
    )
    def test_detects_rate_limit_messages(self, message: str) -> None:
        """Provider-specific wording is all recognised."""
        assert is_rate_limit_error(RuntimeError(message))

    def test_other_errors_are_not_retried(self) -> None:
        """Auth or validation errors must fail immediately."""
        assert not is_rate_limit_error(ValueError("API key not valid"))


class TestRetryDelaySeconds:
    """Tests for retry_delay_seconds."""

    def test_uses_provider_hint_plus_margin(self) -> None:
        """Google's 'retry in 27.5s' hint is respected (rounded up, plus 1s)."""
        assert retry_delay_seconds(GOOGLE_429, attempt=0) == 29.0

    def test_exponential_backoff_without_hint(self) -> None:
        """Without a hint: 2, 4, 8 ... seconds, capped."""
        error = RuntimeError("429")

        assert [retry_delay_seconds(error, attempt) for attempt in range(3)] == [2.0, 4.0, 8.0]
        assert retry_delay_seconds(error, attempt=20) == MAX_RETRY_DELAY_SECONDS


class TestCallWithRateLimitRetry:
    """Tests for call_with_rate_limit_retry."""

    def test_retries_until_success(self) -> None:
        """Rate-limited calls are retried after waiting."""
        outcomes: list[Exception | str] = [GOOGLE_429, GOOGLE_429, "ok"]
        sleeps: list[float] = []

        def flaky() -> str:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        assert call_with_rate_limit_retry(flaky, sleep=sleeps.append) == "ok"
        assert sleeps == [29.0, 29.0]

    def test_non_rate_limit_error_is_raised_immediately(self) -> None:
        """Other errors are not retried."""
        sleeps: list[float] = []

        def broken() -> None:
            raise ValueError("API key not valid")

        with pytest.raises(ValueError, match="API key"):
            call_with_rate_limit_retry(broken, sleep=sleeps.append)
        assert not sleeps

    def test_gives_up_after_max_attempts(self) -> None:
        """A permanent quota problem eventually surfaces."""
        sleeps: list[float] = []

        def always_limited() -> None:
            raise RuntimeError("429")

        with pytest.raises(RuntimeError, match="429"):
            call_with_rate_limit_retry(always_limited, sleep=sleeps.append, max_attempts=3)
        assert len(sleeps) == 2
