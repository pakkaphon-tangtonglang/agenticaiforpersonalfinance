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
        assert "--color-parchment" in response.text

    def test_static_js_served(self) -> None:
        """GET /static/app.js returns JavaScript."""
        response = self.client.get("/static/app.js")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "javascript" in content_type or "text/plain" in content_type
        assert "INTENT_CONFIG" in response.text


class TestUploadRecheckMarkup:
    """Tests for the document-scan + recheck UI markup."""

    client: TestClient = TestClient(app)

    def test_upload_view_has_scan_and_bank_tabs(self) -> None:
        """The upload view exposes scan and bank mode tabs."""
        response = self.client.get("/")
        assert 'data-mode="scan"' in response.text
        assert 'data-mode="bank"' in response.text

    def test_upload_view_has_recheck_table(self) -> None:
        """The upload view contains the recheck table and confirm button."""
        response = self.client.get("/")
        assert 'id="recheckArea"' in response.text
        assert 'id="recheckBody"' in response.text
        assert 'id="confirmBtn"' in response.text

    def test_file_input_accepts_images_and_pdf(self) -> None:
        """The file input default accept includes images and PDF."""
        response = self.client.get("/")
        assert 'accept="image/*,.pdf"' in response.text

    def test_app_js_wires_scan_and_confirm(self) -> None:
        """app.js wires the scan endpoint and the confirm flow."""
        response = self.client.get("/static/app.js")
        assert "/upload/receipt" in response.text
        assert "/transactions/confirm" in response.text
        assert "function scanReceipt" in response.text
        assert "function confirmRecheck" in response.text

    def test_styles_define_recheck_table(self) -> None:
        """styles.css defines the recheck table styles."""
        response = self.client.get("/static/styles.css")
        assert ".recheck-table" in response.text
        assert ".upload-tab" in response.text


class TestDashboardWiring:
    """Regression tests for the dashboard auto-load fix.

    Previously the dashboard never loaded after OCR confirmation because the
    Load button was unwired, switching to the dashboard view did not fetch
    data, and confirmRecheck only refreshed when the dashboard was already
    active. These tests pin the wiring that fixes that.
    """

    client: TestClient = TestClient(app)

    def test_dash_load_button_is_wired(self) -> None:
        """The โหลดข้อมูล button has a click handler that loads the dashboard."""
        response = self.client.get("/static/app.js")
        assert 'addEventListener("click", loadDashboard)' in response.text

    def test_dashboard_auto_loads_on_view_switch(self) -> None:
        """switchView("dashboard") triggers loadDashboard()."""
        response = self.client.get("/static/app.js")
        assert 'if (name === "dashboard") loadDashboard();' in response.text

    def test_confirm_recheck_switches_to_dashboard(self) -> None:
        """confirmRecheck navigates to the dashboard after saving."""
        response = self.client.get("/static/app.js")
        assert 'switchView("dashboard")' in response.text

    def test_confirm_recheck_aligns_filter_to_transactions(self) -> None:
        """confirmRecheck aligns the dashboard filter to saved transactions."""
        response = self.client.get("/static/app.js")
        assert "alignDashboardFilterTo" in response.text


class TestDeployableFrontendPaths:
    """Tests for iHost-deployable frontend (relative assets + configurable API base).

    The frontend must work both locally (FastAPI serves it at /) and on
    iHost KMITL (served under /~username/). Root-relative paths like
    /static/styles.css would 404 on iHost, so assets must be referenced
    relatively, and the backend URL must be configurable via config.js.
    """

    client: TestClient = TestClient(app)

    def test_index_uses_relative_asset_paths(self) -> None:
        """index.html references styles.css and app.js relatively, not /static/."""
        response = self.client.get("/")
        assert 'href="styles.css"' in response.text
        assert 'src="app.js"' in response.text
        assert 'href="/static/styles.css"' not in response.text
        assert 'src="/static/app.js"' not in response.text

    def test_index_loads_config_before_app(self) -> None:
        """config.js is loaded before app.js so FINANCE_API_BASE is set in time."""
        response = self.client.get("/")
        config_pos = response.text.index('src="config.js"')
        app_pos = response.text.index('src="app.js"')
        assert config_pos < app_pos

    def test_config_js_served(self) -> None:
        """GET /config.js returns JavaScript defining window.FINANCE_API_BASE."""
        response = self.client.get("/config.js")
        assert response.status_code == 200
        assert "window.FINANCE_API_BASE" in response.text

    def test_app_js_defines_api_base_with_fallback(self) -> None:
        """app.js derives API_BASE from window.FINANCE_API_BASE with origin fallback."""
        response = self.client.get("/static/app.js")
        assert "const API_BASE = window.FINANCE_API_BASE ?? location.origin" in response.text

    def test_app_js_api_helper_honors_api_base(self) -> None:
        """api() builds URLs from API_BASE instead of location.origin."""
        response = self.client.get("/static/app.js")
        assert "new URL(path, API_BASE)" in response.text

    def test_app_js_stream_honors_api_base(self) -> None:
        """sendChat opens the SSE stream against API_BASE."""
        response = self.client.get("/static/app.js")
        assert "new EventSource(`${API_BASE}/chat/stream?${params}`)" in response.text
