"""Retry embedding calls when the provider rate-limits us.

Free tiers cap embedding requests per minute (Google: 100/min), so a full
knowledge-base rebuild hits 429 errors. Retries honour the provider's own
"retry in Ns" hint and fall back to exponential backoff.
"""

import math
import re
import time
from collections.abc import Callable
from typing import TypeVar

from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

ResultT = TypeVar("ResultT")

MAX_RETRY_ATTEMPTS = 6
MAX_RETRY_DELAY_SECONDS = 60.0
_RATE_LIMIT_MARKERS = ("429", "resource_exhausted", "rate limit", "too many requests")
_RETRY_HINT = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)


def is_rate_limit_error(error: Exception) -> bool:
    """True when an error message looks like a provider rate limit.

    Example:
        >>> is_rate_limit_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
        True
    """
    message = str(error).casefold()
    return any(marker in message for marker in _RATE_LIMIT_MARKERS)


def retry_delay_seconds(error: Exception, attempt: int) -> float:
    """Seconds to wait before the next attempt.

    Args:
        error: The rate-limit error (may contain "retry in 27.5s").
        attempt: Zero-based retry number.

    Returns:
        The provider hint rounded up plus 1s, else 2 ** (attempt + 1),
        capped at MAX_RETRY_DELAY_SECONDS.

    Example:
        >>> retry_delay_seconds(RuntimeError("Please retry in 27.5s"), 0)
        29.0
    """
    hint = _RETRY_HINT.search(str(error))
    delay = math.ceil(float(hint.group(1))) + 1 if hint else 2 ** (attempt + 1)
    return float(min(delay, MAX_RETRY_DELAY_SECONDS))


def call_with_rate_limit_retry(
    function: Callable[[], ResultT],
    sleep: Callable[[float], None] = time.sleep,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
) -> ResultT:
    """Call a function, waiting and retrying while it is rate-limited.

    Args:
        function: Zero-argument call to make (e.g. one embedding batch).
        sleep: Delay function (injectable for tests).
        max_attempts: Total attempts before giving up.

    Returns:
        The function's result.

    Raises:
        Exception: Non-rate-limit errors immediately; the last rate-limit
            error after max_attempts.

    Example:
        >>> call_with_rate_limit_retry(lambda: store.add_texts(texts))
    """
    for attempt in range(max_attempts):
        try:
            return function()
        except Exception as error:  # pylint: disable=broad-exception-caught
            if not is_rate_limit_error(error) or attempt == max_attempts - 1:
                raise
            delay = retry_delay_seconds(error, attempt)
            logger.warning("Rate limited, retrying in %.0fs (attempt %d)", delay, attempt + 1)
            sleep(delay)
    raise RuntimeError("call_with_rate_limit_retry: max_attempts must be at least 1")
