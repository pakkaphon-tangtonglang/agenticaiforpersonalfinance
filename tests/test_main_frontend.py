"""Tests for FastAPI frontend serving routes (root page + static assets)."""

from fastapi.testclient import TestClient

from finance_ai.main import app


class TestFrontendServing:
    """Tests for static frontend route handlers."""

    client: TestClient = TestClient(app)

    def test_root_returns_html(self) -> None:
        """GET / returns 200 with HTML content type."""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "<html" in response.text.lower()
        assert "การเงินส่วนบุคคล" in response.text

    def test_static_css_served(self) -> None:
        """GET /static/styles.css returns CSS."""
        response = self.client.get("/static/styles.css")
        assert response.status_code == 200
        assert "text/css" in response.headers.get("content-type", "")
        assert "--paper" in response.text

    def test_static_js_served(self) -> None:
        """GET /static/app.js returns JavaScript."""
        response = self.client.get("/static/app.js")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "javascript" in content_type or "text/plain" in content_type
        assert "INTENT_CONFIG" in response.text
