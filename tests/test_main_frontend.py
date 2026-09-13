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


class TestAssetSearchAndRiskOnboarding:
    """Tests for asset search UI + risk onboarding modal markup."""

    client: TestClient = TestClient(app)

    def test_assets_view_has_search_controls(self) -> None:
        """The assets view exposes the free-text search input, button, and results."""
        response = self.client.get("/")
        assert 'id="assetSearchInput"' in response.text
        assert 'id="assetSearchBtn"' in response.text
        assert 'id="assetSearchResults"' in response.text
        assert "พิมพ์ชื่อสินทรัพย์" in response.text

    def test_app_js_wires_asset_search(self) -> None:
        """app.js calls /assets/search and renders candidates with actions."""
        response = self.client.get("/static/app.js")
        assert '"/assets/search"' in response.text
        assert "function searchAssets" in response.text
        assert "function renderAssetSearchResults" in response.text
        assert "function fetchAssetData" in response.text

    def test_styles_define_asset_search(self) -> None:
        """styles.css defines the asset search card styles."""
        response = self.client.get("/static/styles.css")
        assert ".asset-result-card" in response.text
        assert ".asset-search-results" in response.text

    def test_risk_onboarding_modal_markup_exists(self) -> None:
        """The risk modal overlay, question box, and nav buttons exist."""
        response = self.client.get("/")
        assert 'id="riskModal"' in response.text
        assert 'id="riskQuestionBox"' in response.text
        assert 'id="riskBackBtn"' in response.text
        assert 'id="riskNextBtn"' in response.text
        assert 'id="riskStepLabel"' in response.text
        assert 'id="riskSkipLink"' in response.text

    def test_dashboard_has_risk_profile_card(self) -> None:
        """The dashboard shows the risk profile card and retake link."""
        response = self.client.get("/")
        assert 'id="riskProfileCard"' in response.text
        assert 'id="riskProfileLevel"' in response.text
        assert 'id="riskRetakeBtn"' in response.text
        assert "ทำแบบประเมินอีกครั้ง" in response.text

    def test_app_js_embeds_all_12_risk_questions(self) -> None:
        """app.js embeds the 12 SEC questions and wires both risk endpoints."""
        response = self.client.get("/static/app.js")
        assert '"/risk-assessment/latest"' in response.text
        assert '"/risk-assessment/submit"' in response.text
        assert "function maybeShowRiskAssessment" in response.text
        assert "function submitRiskAssessment" in response.text
        for question_id in range(1, 13):
            assert f"{{ id: {question_id}," in response.text

    def test_eval_tab_removed(self) -> None:
        """The ประเมินผล tab, view, and JS wiring are fully removed."""
        page = self.client.get("/")
        assert 'data-view="eval"' not in page.text
        assert 'id="view-eval"' not in page.text
        js = self.client.get("/static/app.js").text
        assert "setupEval" not in js
        assert "runEvaluation" not in js
        assert '"/evaluation/run"' not in js
        styles = self.client.get("/static/styles.css").text
        assert ".eval-result" not in styles


class TestAssetFetchCardsAndWatchlist:
    """Tests for asset fetch result cards, selected-asset display,
    and the watchlist section (task-2 asset page bugfix)."""

    client: TestClient = TestClient(app)

    def test_index_bumps_static_versions_to_v5(self) -> None:
        """Every static reference in index.html is cache-busted to v=5."""
        response = self.client.get("/")
        assert 'href="styles.css?v=5"' in response.text
        assert 'src="config.js?v=5"' in response.text
        assert 'src="app.js?v=5"' in response.text
        assert "?v=4" not in response.text

    def test_manual_symbol_form_removed(self) -> None:
        """The raw-symbol form is gone — symbols come only from search."""
        response = self.client.get("/")
        assert 'id="assetForm"' not in response.text
        assert 'id="assetSymbol"' not in response.text
        assert 'id="assetType"' not in response.text
        assert 'id="assetSearchForm"' in response.text

    def test_selected_asset_display_exists(self) -> None:
        """A selected-asset display sits next to the fetch form."""
        response = self.client.get("/")
        assert 'id="selectedAsset"' in response.text
        assert 'class="selected-asset"' in response.text

    def test_watchlist_section_markup_exists(self) -> None:
        """The watchlist section lists tracked assets with an empty state."""
        response = self.client.get("/")
        assert 'id="watchlistSection"' in response.text
        assert 'id="watchlistList"' in response.text
        assert "สินทรัพย์ที่ติดตาม" in response.text
        assert "ยังไม่มีสินทรัพย์ที่ติดตาม" in response.text

    def test_app_js_wires_watchlist_endpoints(self) -> None:
        """app.js loads, adds to, and removes watchlist entries."""
        response = self.client.get("/static/app.js")
        assert '"/assets/watchlist"' in response.text
        assert "function loadWatchlist" in response.text
        assert "function renderWatchlist" in response.text
        assert "function addToWatchlist" in response.text
        assert "function removeWatchlistAsset" in response.text

    def test_app_js_renders_fetch_results_as_cards(self) -> None:
        """app.js renders fetch results as cards and always fetches all data."""
        response = self.client.get("/static/app.js")
        assert "function renderAssetFetchResult" in response.text
        assert 'fetch_type: "all"' in response.text
        assert "JSON.stringify(res.result" not in response.text

    def test_app_js_search_cards_have_fetch_and_track_buttons(self) -> None:
        """Search cards offer ดึงข้อมูล and ติดตาม without a manual symbol form."""
        response = self.client.get("/static/app.js")
        assert '"ดึงข้อมูล"' in response.text
        assert '"ติดตาม"' in response.text
        assert "function buildAssetSearchCard" in response.text
        assert "fetchAssetData(result.symbol" in response.text

    def test_app_js_watchlist_rows_are_clickable_to_fetch(self) -> None:
        """Watchlist rows fetch data on click; ตัดออก does not trigger it."""
        response = self.client.get("/static/app.js")
        assert "fetchAssetData(row.symbol" in response.text
        assert "e.stopPropagation()" in response.text

    def test_app_js_formats_thai_dates_and_strips_markdown(self) -> None:
        """app.js formats timestamps with th-TH locale and strips markdown bold."""
        response = self.client.get("/static/app.js")
        assert "function formatThaiDateTime" in response.text
        assert '"th-TH"' in response.text
        assert "function stripMarkdownEmphasis" in response.text

    def test_app_js_renders_news_links_safely(self) -> None:
        """News links are anchors with hostname text, _blank, and noopener."""
        response = self.client.get("/static/app.js")
        assert 'target: "_blank"' in response.text
        assert 'rel: "noopener"' in response.text
        assert "function linkHostname" in response.text

    def test_app_js_loads_watchlist_on_assets_view(self) -> None:
        """Switching to the assets view loads notifications and the watchlist."""
        response = self.client.get("/static/app.js")
        assert 'if (name === "assets") { loadNotifications(); loadWatchlist(); }' in response.text

    def test_styles_define_fetch_cards_and_watchlist(self) -> None:
        """styles.css styles fetch result cards and watchlist rows."""
        response = self.client.get("/static/styles.css")
        assert ".asset-price-line" in response.text
        assert ".asset-news-card" in response.text
        assert ".asset-news-link" in response.text
        assert ".selected-asset" in response.text
        assert ".watchlist-item" in response.text
        assert ".watchlist-remove" in response.text


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
        assert 'href="styles.css?v=' in response.text
        assert 'src="app.js?v=' in response.text
        assert 'href="/static/styles.css"' not in response.text
        assert 'src="/static/app.js"' not in response.text

    def test_index_loads_config_before_app(self) -> None:
        """config.js is loaded before app.js so FINANCE_API_BASE is set in time."""
        response = self.client.get("/")
        config_pos = response.text.index("config.js?v=")
        app_pos = response.text.index('src="app.js?v=')
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


class TestRecommendationDisclaimer:
    """Tests for the Thai disclaimer rendered under recommendation chat replies."""

    client: TestClient = TestClient(app)

    def test_app_js_renders_disclaimer_for_recommendation_intent(self) -> None:
        """app.js defines the disclaimer helper and shows it once per session."""
        response = self.client.get("/static/app.js")
        assert "function intentDisclaimer" in response.text
        assert "ไม่ใช่คำแนะนำการลงทุน" in response.text
        assert "recommendationDisclaimerShown" in response.text
        assert "!state.recommendationDisclaimerShown" in response.text

    def test_styles_define_msg_disclaimer(self) -> None:
        """styles.css defines the chat disclaimer style."""
        response = self.client.get("/static/styles.css")
        assert ".msg-disclaimer" in response.text


class TestChatMessageFormatting:
    """Tests for chat bubble markdown rendering (stray blank-line fix)."""

    client: TestClient = TestClient(app)

    def test_marked_breaks_disabled(self) -> None:
        """Single newlines render as soft wraps (GPT/Claude behaviour).

        `breaks: true` forces every \n into a hard <br>, which shows stray
        blank lines wherever the model wraps a line oddly.
        """
        response = self.client.get("/static/app.js")
        assert "breaks: false" in response.text
        assert "breaks: true" not in response.text

    def test_normalize_for_chat_defined_and_used(self) -> None:
        """normalizeForChat trims edges and collapses blank-line runs."""
        response = self.client.get("/static/app.js")
        assert "function normalizeForChat" in response.text
        assert "normalizeForChat(text)" in response.text
        assert "\\n{3,}" in response.text
