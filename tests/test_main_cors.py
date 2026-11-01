"""Tests for cross-origin access.

The frontend is served by FastAPI itself (same origin on Render and in local
dev), so no other site needs browser access to the API. With no CORS
middleware, browsers block every cross-origin call — including the retired
iHost origin.
"""

from fastapi.testclient import TestClient

from finance_ai.main import app


class TestNoCrossOriginAccess:
    """No origin receives CORS allow headers."""

    client: TestClient = TestClient(app)

    def test_retired_ihost_origin_denied(self) -> None:
        """The old iHost frontend origin is no longer allowed."""
        response = self.client.get("/health", headers={"Origin": "https://www.it.kmitl.ac.th"})
        assert "access-control-allow-origin" not in response.headers

    def test_unknown_origin_denied(self) -> None:
        """Requests from other sites receive no allow-origin header."""
        response = self.client.get("/health", headers={"Origin": "https://evil.example.com"})
        assert "access-control-allow-origin" not in response.headers

    def test_preflight_not_answered_with_cors_headers(self) -> None:
        """A browser preflight from another site gets no permission to proceed."""
        response = self.client.options(
            "/chat",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" not in response.headers

    def test_same_origin_frontend_still_served(self) -> None:
        """The web UI is served by the API host itself."""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "<html" in response.text.lower()
