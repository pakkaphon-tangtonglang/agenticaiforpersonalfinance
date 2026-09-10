"""Tests for CORS middleware configuration.

The frontend is served from iHost (https://www.it.kmitl.ac.th) while the
API runs on a separate host, so CORS must allow exactly the iHost origin
plus localhost for development — and nothing else. No credentials are
used (user_id travels in request bodies), so credentialed CORS must be off.
"""

from fastapi.testclient import TestClient

from finance_ai.main import app


class TestCorsConfiguration:
    """CORS must allow only the iHost origin and localhost, without credentials."""

    client: TestClient = TestClient(app)

    def test_ihost_origin_allowed(self) -> None:
        """Requests from the iHost origin receive the matching allow-origin header."""
        response = self.client.get("/health", headers={"Origin": "https://www.it.kmitl.ac.th"})
        assert response.headers.get("access-control-allow-origin") == "https://www.it.kmitl.ac.th"

    def test_localhost_origin_allowed(self) -> None:
        """Requests from localhost:8080 receive the matching allow-origin header."""
        response = self.client.get("/health", headers={"Origin": "http://localhost:8080"})
        assert response.headers.get("access-control-allow-origin") == "http://localhost:8080"

    def test_unknown_origin_denied(self) -> None:
        """Requests from unknown origins receive no allow-origin header."""
        response = self.client.get("/health", headers={"Origin": "https://evil.example.com"})
        assert "access-control-allow-origin" not in response.headers

    def test_credentials_not_allowed(self) -> None:
        """CORS must not allow credentials (no cookies are used)."""
        response = self.client.get("/health", headers={"Origin": "https://www.it.kmitl.ac.th"})
        assert response.headers.get("access-control-allow-credentials") != "true"
