"""Retry helper for transient LLM provider errors during evaluation.

Ollama Cloud sheds load with HTTP 429 responses and Windows clients
sometimes see socket aborts under heavy concurrency. Dataset evaluators
call this helper so a transient failure is retried with exponential
backoff before the case is marked failed.
"""

import time
from collections.abc import Callable
from typing import TypeVar

from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

TypeVarResult = TypeVar("TypeVarResult")

_RETRYABLE_MARKERS = (
    "429",
    "too many concurrent requests",
    "rate limit",
    "resource has been exhausted",
    "connection was aborted",
    "10053",
)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_INITIAL_DELAY_SECONDS = 5.0


def _is_retryable(error: Exception) -> bool:
    """Check whether an error looks like a transient rate/limit failure.

    Args:
        error: The exception raised by the LLM call.

    Returns:
        True when the error is worth retrying.
    """
    message = str(error).lower()
    return any(marker in message for marker in _RETRYABLE_MARKERS)


def invoke_with_retry(
    action: Callable[[], TypeVarResult],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_delay: float = DEFAULT_INITIAL_DELAY_SECONDS,
) -> TypeVarResult:
    """Run an action, retrying transient provider errors with backoff.

    Args:
        action: Zero-argument callable performing the LLM invocation.
        max_attempts: Total attempts before giving up.
        initial_delay: Seconds before the first retry (doubles each time).

    Returns:
        Whatever the action returns on success.

    Raises:
        Exception: The last error when attempts are exhausted, or the
            original error immediately when it is not retryable.

    Example:
        >>> response = invoke_with_retry(lambda: model.invoke(messages))
    """
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return action()
        except Exception as error:  # noqa: BLE001  # classify then re-raise
            if not _is_retryable(error) or attempt == max_attempts:
                raise
            logger.warning("Retryable LLM error (attempt %d): %s", attempt, error)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")
