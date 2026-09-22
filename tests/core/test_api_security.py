"""Tests for API security middleware (API key guard + per-IP rate limit)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from finance_ai.core.api_security import (
    EXEMPT_PATHS,
    install_api_security,
)


@pytest.fixture(name="client")
def client_fixture(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a FastAPI app with security middleware and a test endpoint."""
    monkeypatch.setenv("API_KEY", "secret-key-123")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    from finance_ai.core.config import get_settings

    app = FastAPI()
    install_api_security(app)

    @app.post("/chat")
    def chat() -> dict[str, str]:
        return {"ok": "true"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return TestClient(app)


class TestApiKeyGuard:
    """API key is required when settings.api_key is set."""

    def test_request_without_key_is_rejected(self, client: TestClient) -> None:
        response = client.post("/chat", json={})
        assert response.status_code == 401

    def test_request_with_wrong_key_is_rejected(self, client: TestClient) -> None:
        response = client.post("/chat", json={}, headers={"X-API-Key": "wrong"})
        assert response.status_code == 401

    def test_request_with_valid_key_passes(self, client: TestClient) -> None:
        response = client.post("/chat", json={}, headers={"X-API-Key": "secret-key-123"})
        assert response.status_code == 200

    def test_health_endpoint_is_exempt(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_exempt_paths_cover_line_webhook(self) -> None:
        assert "/line/webhook" in EXEMPT_PATHS


class TestRateLimit:
    """Per-IP sliding window rate limiting."""

    def test_excess_requests_get_429(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_KEY", "secret-key-123")
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
        from finance_ai.core.config import get_settings

        app = FastAPI()
        install_api_security(app)

        @app.get("/ping")
        def ping(request: Request) -> dict[str, str]:
            return {"client": request.headers.get("x-test", "")}

        client = TestClient(app)
        headers = {"X-API-Key": "secret-key-123", "X-Forwarded-For": "1.2.3.4"}
        assert client.get("/ping", headers=headers).status_code == 200
        assert client.get("/ping", headers=headers).status_code == 200
        assert client.get("/ping", headers=headers).status_code == 429

    def test_limit_disabled_when_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_KEY", "secret-key-123")
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
        from finance_ai.core.config import get_settings

        app = FastAPI()
        install_api_security(app)

        @app.get("/ping")
        def ping() -> dict[str, str]:
            return {"ok": "true"}

        client = TestClient(app)
        headers = {"X-API-Key": "secret-key-123"}
        for _ in range(10):
            assert client.get("/ping", headers=headers).status_code == 200


class TestDisabledGuard:
    """When api_key is unset the middleware allows all requests."""

    def test_open_when_no_key_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
        from finance_ai.core.config import get_settings

        app = FastAPI()
        install_api_security(app)

        @app.post("/chat")
        def chat() -> dict[str, str]:
            return {"ok": "true"}

        client = TestClient(app)
        assert client.post("/chat", json={}).status_code == 200
