"""Tests for the shared retry helper used by evaluation LLM calls."""

from unittest.mock import MagicMock, patch

import pytest

from finance_ai.evaluation.llm_retry import invoke_with_retry


class TestInvokeWithRetry:
    """Tests for invoke_with_retry."""

    def test_returns_result_first_try(self) -> None:
        """No retry needed when the call succeeds."""
        action = MagicMock(return_value="ok")
        assert invoke_with_retry(action) == "ok"
        action.assert_called_once()

    def test_retries_on_rate_limit_then_succeeds(self) -> None:
        """A 429-style error is retried until success."""
        action = MagicMock(
            side_effect=[
                Exception("too many concurrent requests (status code: 429)"),
                "ok",
            ]
        )
        with patch("finance_ai.evaluation.llm_retry.time.sleep") as mock_sleep:
            assert invoke_with_retry(action) == "ok"
        assert action.call_count == 2
        mock_sleep.assert_called_once()

    def test_raises_after_max_attempts(self) -> None:
        """The last error is raised once all attempts are exhausted."""
        action = MagicMock(side_effect=Exception("too many concurrent requests"))
        with (
            patch("finance_ai.evaluation.llm_retry.time.sleep"),
            pytest.raises(Exception, match="too many concurrent"),
        ):
            invoke_with_retry(action, max_attempts=2)
        assert action.call_count == 2

    def test_does_not_retry_unrelated_errors(self) -> None:
        """Non-transient errors propagate immediately."""
        action = MagicMock(side_effect=ValueError("income cannot be negative"))
        with pytest.raises(ValueError, match="income cannot be negative"):
            invoke_with_retry(action)
        action.assert_called_once()

    def test_backoff_grows_exponentially(self) -> None:
        """Delays double between attempts."""
        action = MagicMock(side_effect=Exception("429"))
        with (
            patch("finance_ai.evaluation.llm_retry.time.sleep") as mock_sleep,
            pytest.raises(Exception),
        ):
            invoke_with_retry(action, max_attempts=3, initial_delay=5.0)
        assert [call_item.args[0] for call_item in mock_sleep.call_args_list] == [5.0, 10.0]

    def test_retries_connection_abort(self) -> None:
        """Windows socket aborts from load-shedding are retried."""
        action = MagicMock(
            side_effect=[
                Exception("[WinError 10053] An established connection was aborted"),
                "ok",
            ]
        )
        with patch("finance_ai.evaluation.llm_retry.time.sleep"):
            assert invoke_with_retry(action) == "ok"
        assert action.call_count == 2
