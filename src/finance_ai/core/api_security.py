"""API security middleware: API key guard + per-IP rate limiting.

Protects a publicly deployed FastAPI service from unauthenticated
abuse (costly LLM/OCR calls). Two layers:

1. API key guard - when ``settings.api_key`` is set, every request
   must carry the value in the ``X-API-Key`` header. ``/health`` and
   ``/line/webhook`` are exempt (the LINE webhook already verifies
   cryptographic signatures).
2. Per-IP rate limit - a sliding-window counter keyed by client IP,
   capped at ``settings.rate_limit_per_minute`` (0 disables).

The limiter is in-memory by design: each Render instance is a single
process, so instance-local state is sufficient and adds no external
dependency.

Example:
    >>> from finance_ai.core.api_security import install_api_security
    >>> install_api_security(app)  # doctest: +SKIP
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response as StarletteResponse

from finance_ai.core.config import get_settings
from finance_ai.core.logging import get_logger

logger = get_logger(__name__)

EXEMPT_PATHS: tuple[str, ...] = ("/health", "/line/webhook")

API_KEY_HEADER: str = "X-API-Key"
_RATE_LIMIT_HEADER: str = "X-Forwarded-For"

# IP -> deque of timestamps (within the current window)
_request_log: dict[str, deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    """Resolve the client IP, preferring proxy headers from Render.

    Args:
        request: Incoming request.

    Returns:
        Client IP string (first proxy hop when behind a router).
    """
    forwarded = request.headers.get(_RATE_LIMIT_HEADER)
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_exempt(path: str) -> bool:
    """Check whether a path is exempt from security checks.

    Args:
        path: Request path.

    Returns:
        True when the path should pass through unchecked.
    """
    return any(path == exempt or path.startswith(exempt) for exempt in EXEMPT_PATHS)


def _check_api_key(request: Request, api_key: str) -> bool:
    """Validate the X-API-Key header against the configured key.

    Args:
        request: Incoming request.
        api_key: Configured shared key.

    Returns:
        True when the supplied key matches.
    """
    supplied = request.headers.get(API_KEY_HEADER, "")
    return bool(supplied) and supplied == api_key


def _check_rate_limit(client_ip: str, limit_per_minute: int) -> bool:
    """Sliding-window rate limit per client IP.

    Args:
        client_ip: Resolved client IP.
        limit_per_minute: Max requests per minute; 0 disables.

    Returns:
        True when the request is allowed, False when limited.
    """
    if limit_per_minute <= 0:
        return True
    now = time.monotonic()
    window = _request_log[client_ip]
    while window and now - window[0] > 60.0:
        window.popleft()
    if len(window) >= limit_per_minute:
        return False
    window.append(now)
    return True


def reset_rate_limiter() -> None:
    """Clear all recorded request timestamps (for tests and resets)."""
    _request_log.clear()


def install_api_security(app: FastAPI) -> None:
    """Attach the API security middleware to a FastAPI application.

    Reads ``settings.api_key`` and ``settings.rate_limit_per_minute``
    at request time so configuration changes apply without reinstall.

    Args:
        app: FastAPI application to protect.
    """

    @app.middleware("http")
    async def api_security_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[StarletteResponse]]
    ) -> StarletteResponse:
        """Reject unauthenticated or excessive requests before handlers.

        Args:
            request: Incoming request.
            call_next: Next handler in the chain.

        Returns:
            401 for missing/invalid keys, 429 for rate-limited clients,
            or the upstream response.
        """
        settings = get_settings()
        path = request.url.path
        if path in EXEMPT_PATHS:
            return await call_next(request)

        if settings.api_key and not _check_api_key(request, settings.api_key):
            logger.warning("Rejected request with missing/invalid API key: %s", path)
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key"})

        client_ip = _client_ip(request)
        if not _check_rate_limit(client_ip, settings.rate_limit_per_minute):
            logger.warning("Rate limit exceeded for %s on %s", client_ip, path)
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})

        return await call_next(request)

    installed_settings = get_settings()
    logger.info(
        "API security installed (key_configured=%s, rate_limit=%s/min)",
        bool(installed_settings.api_key),
        installed_settings.rate_limit_per_minute,
    )
