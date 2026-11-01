"""Tests for FastAPI frontend serving routes (root page + static assets)."""

import re

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
        assert "--canvas:" in response.text

    def test_static_js_served(self) -> None:
        """GET /static/app.js returns JavaScript."""
        response = self.client.get("/static/app.js")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "javascript" in content_type or "text/plain" in content_type
        assert "INTENT_CONFIG" in response.text

    def test_intent_config_covers_clarify_intent(self) -> None:
        """The clarify intent has its own badge label, not the unknown fallback."""
        response = self.client.get("/static/app.js")
        assert response.status_code == 200
        assert "clarify:" in response.text


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

    def test_index_cache_busts_static_assets_with_one_version(self) -> None:
        """All local CSS/JS files share the same ?v= cache-buster."""
        response = self.client.get("/")
        versions = {
            re.search(rf'{asset}\?v=(\d+)"', response.text).group(1)  # type: ignore[union-attr]
            for asset in (
                r"styles\.css",
                r"chat_format\.js",
                r"citations\.js",
                r"app\.js",
                r"guru\.js",
                r"history\.js",
            )
        }
        assert len(versions) == 1, f"static assets use different versions: {versions}"
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

    def test_holdings_list_sits_in_watchlist_section(self) -> None:
        """Owned assets show in the watchlist panel, separate from tracked ones."""
        html = self.client.get("/").text
        section = html.split('id="watchlistSection"', 1)[1].split("</section>", 1)[0]
        assert 'id="holdingsList"' in section
        assert "รายการถือ (พอร์ต)" in section
        assert "ยังไม่มีรายการถือ" in section

    def test_app_js_loads_holdings(self) -> None:
        """app.js loads /assets/holdings when the assets page opens."""
        js = self.client.get("/static/app.js").text
        assert '"/assets/holdings"' in js
        assert "loadHoldings()" in js

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
        assert (
            'if (name === "assets") { loadNotifications(); loadHoldings(); loadWatchlist(); }'
            in response.text
        )

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


class TestSameOriginFrontend:
    """The web UI is served by the API host itself (Render or local uvicorn).

    Assets stay relative, and the frontend always calls the origin it was
    loaded from; there is no separately hosted copy and no config.js.
    """

    client: TestClient = TestClient(app)

    def test_index_uses_relative_asset_paths(self) -> None:
        """index.html references styles.css and app.js relatively, not /static/."""
        response = self.client.get("/")
        assert 'href="styles.css?v=' in response.text
        assert 'src="app.js?v=' in response.text
        assert 'href="/static/styles.css"' not in response.text
        assert 'src="/static/app.js"' not in response.text

    def test_index_has_no_config_js(self) -> None:
        """The separate-host config script is gone."""
        response = self.client.get("/")
        assert "config.js" not in response.text
        assert self.client.get("/config.js").status_code == 404

    def test_app_js_calls_its_own_origin(self) -> None:
        """app.js sends API calls to the origin that served the page."""
        response = self.client.get("/static/app.js")
        assert "const API_BASE = location.origin;" in response.text
        assert "FINANCE_API_BASE" not in response.text

    def test_app_js_api_helper_honors_api_base(self) -> None:
        """api() builds URLs from API_BASE."""
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


class TestGuruTab:
    """Tests for the Guru tab markup and script."""

    client: TestClient = TestClient(app)

    def test_guru_nav_and_view_exist(self) -> None:
        """The sidebar links to a Guru view with a persistent disclaimer."""
        html = self.client.get("/").text
        assert 'data-view="guru"' in html
        assert 'id="view-guru"' in html
        assert "ไม่ใช่คำพูดหรือคำแนะนำจากบุคคลจริง" in html
        assert 'data-mode="panel"' in html

    def test_guru_js_served_and_wired(self) -> None:
        """guru.js calls the guru endpoints and registers GuruView."""
        response = self.client.get("/static/guru.js")
        assert response.status_code == 200
        js = response.text
        assert "/guru/chat/stream" in js
        assert "/guru/panel/stream" in js
        assert "window.GuruView" in js
        assert 'kind: "guru"' in js

    def test_app_js_exposes_bridge_and_opens_guru_view(self) -> None:
        """app.js shares helpers with guru.js and opens it on view switch."""
        js = self.client.get("/static/app.js").text
        assert "window.FinanceApp" in js
        assert "window.GuruView.open()" in js


class TestChatAttachment:
    """Tests for attaching slips/receipts/bank files inside the chat."""

    client: TestClient = TestClient(app)

    def test_both_chat_boxes_have_attach_button_and_file_input(self) -> None:
        """Hero and thread composers each get a 📎 button sharing one file input."""
        html = self.client.get("/").text
        assert html.count("data-attach") == 2
        assert 'id="chatFileInput"' in html
        assert 'accept="image/*,.pdf,.csv,.xlsx"' in html

    def test_app_js_reuses_upload_endpoints_and_saves_history(self) -> None:
        """Chat attachments use the existing upload/confirm flow plus the history note."""
        js = self.client.get("/static/app.js").text
        assert "function attachFileInChat" in js
        assert "/upload/receipt" in js and "/upload/bank-statement" in js
        assert "/transactions/confirm" in js
        assert "/attachments" in js
        assert 'upload: { label: "แนบไฟล์"' in js

    def test_upload_tab_uses_plain_wording(self) -> None:
        """The original tab stays, renamed with easier words."""
        html = self.client.get("/").text
        assert 'id="view-upload"' in html
        assert "เพิ่มรายการจากไฟล์" in html
        assert "ถ่าย/แนบสลิป-ใบเสร็จ" in html
        assert "ไฟล์จากแอปธนาคาร" in html
        assert "เช็กให้ถูกก่อนบันทึก" in html
        assert "นำเข้าเอกสาร" not in html


class TestDashboardHistory:
    """The dashboard history card, form and undo toast are wired up."""

    client: TestClient = TestClient(app)

    def test_markup_present(self) -> None:
        """Card, filters, form and undo toast exist in the page."""
        html = self.client.get("/").text
        for element_id in (
            "historyCard",
            "historyList",
            "historyAdd",
            "historyModal",
            "historyForm",
            "historyDate",
            "historyCategory",
            "undoToast",
        ):
            assert f'id="{element_id}"' in html, element_id
        assert 'src="history.js?v=' in html

    def test_history_js_served_with_ten_second_undo(self) -> None:
        """history.js is served and waits 10 s before sending a delete."""
        js = self.client.get("/static/history.js").text
        assert "const UNDO_SECONDS = 10;" in js
        assert "keepalive: true" in js
        assert "window.HistoryView" in js

    def test_category_keys_match_backend(self) -> None:
        """The form offers exactly the categories the API accepts."""
        import re

        from finance_ai.tools.expense_constants import EXPENSE_CATEGORIES, INCOME_TYPES

        js = self.client.get("/static/history.js").text
        blocks = dict(re.findall(r"(expense|income): \{(.*?)\}", js, re.DOTALL))
        assert set(re.findall(r"(\w+):", blocks["expense"])) == set(EXPENSE_CATEGORIES)
        assert set(re.findall(r"(\w+):", blocks["income"])) == set(INCOME_TYPES)

    def test_dashboard_loads_history_for_selected_month(self) -> None:
        """app.js asks HistoryView to load the same year and month."""
        js = self.client.get("/static/app.js").text
        assert "window.HistoryView.load(year, month)" in js


class TestBankStatementPdfUpload:
    """The bank tab accepts PDF statements and asks for a password when locked."""

    client: TestClient = TestClient(app)

    def test_password_dialog_and_include_column(self) -> None:
        """The page has the password dialog and an include column in the review table."""
        html = self.client.get("/").text
        assert 'id="pdfPasswordModal"' in html
        assert 'type="password" id="pdfPasswordInput"' in html
        assert "<th>บันทึก</th>" in html

    def test_app_js_routes_pdf_to_statement_reader(self) -> None:
        """Bank-mode PDFs go to the statement endpoint; own transfers start unticked."""
        js = self.client.get("/static/app.js").text
        assert '".csv,.xlsx,.pdf"' in js
        assert "/upload/bank-statement/pdf" in js
        assert "pdf_password_required" in js
        assert "box.checked = !draft.own_transfer;" in js


class TestDesignPolish:
    """Accessibility and motion fixes from the design polish pass (4.9)."""

    client: TestClient = TestClient(app)

    def test_conversation_and_watchlist_rows_open_with_a_button(self) -> None:
        """Opening a chat or a watched asset works by keyboard (a real button)."""
        response = self.client.get("/static/app.js")
        assert 'el("button", { class: "conv-text", type: "button" }' in response.text
        assert 'class: "watchlist-open"' in response.text
        assert "onclick: () => fetchAssetData(row.symbol, row.name)" in response.text

    def test_escape_and_outside_tap_dismiss_overlays(self) -> None:
        """Escape closes the top overlay; tapping outside closes the mobile sidebar."""
        response = self.client.get("/static/app.js")
        assert "function closeTopOverlay()" in response.text
        assert "bindOverlayDismissal();" in response.text

    def test_styles_define_every_token_they_use(self) -> None:
        """No var(--x) refers to a custom property that styles.css never defines."""
        import re

        css = self.client.get("/static/styles.css").text
        defined = set(re.findall(r"(--[\w-]+)\s*:", css))
        used = set(re.findall(r"var\((--[\w-]+)", css))
        assert used - defined == set()

    def test_styles_have_base_focus_ring_and_coarse_targets(self) -> None:
        """Every control gets a focus ring; touch targets grow on coarse pointers."""
        css = self.client.get("/static/styles.css").text
        assert ":where(button, a, [tabindex], summary, select):focus-visible" in css
        assert "@media (pointer: coarse)" in css

    def test_styles_animate_entrances_with_starting_style(self) -> None:
        """Modals, undo toast and citation popover fade in via @starting-style."""
        css = self.client.get("/static/styles.css").text
        assert css.count("@starting-style") >= 4
        assert "@media (prefers-reduced-motion: reduce)" in css

    def test_index_declares_theme_color(self) -> None:
        """The browser chrome matches the light shell; JS switches it in dark mode."""
        response = self.client.get("/")
        assert '<meta name="theme-color" content="#ffffff" />' in response.text


class TestDesignV2:
    """v2 redesign (designref/v2_design.md): Orbita shell, Lucide icons, light/dark."""

    client: TestClient = TestClient(app)

    def test_emblem_and_favicon_are_served(self) -> None:
        """The full emblem is the logo; a small copy is the favicon."""
        html = self.client.get("/").text
        assert 'class="rail-logo" src="emblem.png"' in html
        assert 'rel="icon" type="image/png" href="favicon.png"' in html
        for asset in ("/emblem.png", "/favicon.png"):
            response = self.client.get(asset)
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/png"

    def test_icons_come_from_the_lucide_sprite(self) -> None:
        """Icons are Lucide symbols referenced with <use>, licence noted."""
        html = self.client.get("/").text
        assert "Lucide v0.469.0" in html and "ISC" in html
        assert '<symbol id="i-chat"' in html
        assert '<use href="#i-chat"/>' in html

    def test_theme_is_light_unless_user_picked_dark(self) -> None:
        """A head script starts light and switches only on a saved "dark" choice."""
        html = self.client.get("/").text
        head = html[: html.index("</head>")]
        assert 'var theme = "light";' in head
        assert 'localStorage.getItem("pfai_theme") === "dark"' in head
        assert "prefers-color-scheme" not in head
        assert 'id="themeToggle"' in html

    def test_every_light_colour_has_a_dark_value(self) -> None:
        """Each colour token in :root is redefined for the dark theme."""
        import re

        css = self.client.get("/static/styles.css").text
        light_block = css[css.index(":root {") : css.index("/* ── Typography ── */")]
        dark_block = css[css.index(':root[data-theme="dark"] {') :]
        dark_block = dark_block[: dark_block.index("}")]
        light = set(re.findall(r"(--[\w-]+)\s*:", light_block))
        dark = set(re.findall(r"(--[\w-]+)\s*:", dark_block))
        assert light - dark == set()

    def test_app_js_wires_theme_home_and_sidebar(self) -> None:
        """Theme toggle, home actions and the sidebar collapse are wired in app.js."""
        js = self.client.get("/static/app.js").text
        assert "function applyTheme" in js and "function setupThemeToggle" in js
        assert "function setupHomeActions" in js and "function loadHomeHealth" in js
        assert "function setupSidebarCollapse" in js
        assert "no-panel" not in js
        assert 'cssVar("--accent")' in js
        assert "#134611" not in js


class TestBeautifulUiComponents:
    """Phase 2: Beautiful UI behaviours wired into the real app."""

    client: TestClient = TestClient(app)

    def test_chat_answers_get_thinking_trace_actions_and_follow_ups(self) -> None:
        """Live answers show a loader, then a trace, a copy action and follow-ups."""
        js = self.client.get("/static/app.js").text
        assert "function thinkingRow" in js and "function traceDetails" in js
        assert "function answerActions" in js and "function followUps" in js
        assert "const FOLLOW_UPS = {" in js
        assert "finishLiveAnswer(assistantBubble" in js

    def test_review_table_save_button_counts_selected_rows(self) -> None:
        """The save button names how many rows will be saved."""
        js = self.client.get("/static/app.js").text
        assert "function bindSaveCount" in js
        assert "บันทึก ${count} รายการ" in js
        css = self.client.get("/static/styles.css").text
        assert ".recheck-row.is-excluded .recheck-input { text-decoration: line-through; }" in css

    def test_chat_review_card_header_matches_its_columns(self) -> None:
        """The in-chat review table has a header for the include column too."""
        js = self.client.get("/static/app.js").text
        assert '["บันทึก", "ประเภท", "จำนวนเงิน", "วันที่", "หมวดหมู่", "รายละเอียด", ""]' in js

    def test_file_imports_show_task_rows(self) -> None:
        """Uploads show send → read → result progress rows."""
        html = self.client.get("/").text
        assert 'id="importSteps"' in html
        js = self.client.get("/static/app.js").text
        assert "function setImportSteps" in js
        assert 'setImportSteps(["done", "failed", "waiting"])' in js

    def test_highlights_render_as_recommendation_cards(self) -> None:
        """Dashboard highlights get a type label and an ask-the-assistant action."""
        js = self.client.get("/static/app.js").text
        assert "function recommendationCard" in js
        assert "ถามผู้ช่วย" in js

    def test_guru_descriptions_are_not_clamped(self) -> None:
        """Guru taglines show in full so people can choose by reading them."""
        css = self.client.get("/static/styles.css").text
        start = css.index(".guru-card-tagline {")
        rule = css[start : css.index("}", start)]
        assert "line-clamp" not in rule


class TestMotionAdditions:
    """Animation-skill picks: guru answers fade in; saved confirmations draw a check."""

    client: TestClient = TestClient(app)

    def test_guru_panel_answers_fade_in_as_they_arrive(self) -> None:
        """Each guru card and the comparison summary fade in when filled."""
        js = self.client.get("/static/guru.js").text
        assert 'body.classList.add("fade-in")' in js
        assert '.classList.add("guru-synthesis", "fade-in")' in js

    def test_saved_confirmations_show_a_drawn_check(self) -> None:
        """Toasts and chat summaries for saved rows carry the check mark."""
        js = self.client.get("/static/app.js").text
        assert "function savedMark" in js
        assert js.count("{ saved: true }") >= 4
        css = self.client.get("/static/styles.css").text
        assert "@keyframes saved-tick" in css

    def test_chat_answer_loader_settles_into_the_trace(self) -> None:
        """The loader row is replaced by the trace in place (no layout jump)."""
        js = self.client.get("/static/app.js").text
        assert "live.thinking.settle(traceDetails(" in js
        assert "thinking.answering()" in js


class TestSingleSidebar:
    """One sidebar on every page (ChatGPT/Claude style), collapsible to icons."""

    client: TestClient = TestClient(app)

    def test_sidebar_holds_menu_history_and_account(self) -> None:
        """Menu labels, chat history and the account menu live in one sidebar."""
        html = self.client.get("/").text
        sidebar = html[
            html.index('<div class="sidebar" id="sidebar">') : html.index("<!-- Main canvas -->")
        ]
        assert '<span class="nav-label">แดชบอร์ด</span>' in sidebar
        assert 'id="convList"' in sidebar and 'id="newChatBtn"' in sidebar
        assert 'id="sidebarCollapse"' in sidebar and 'class="user-menu"' in sidebar

    def test_collapsed_state_is_applied_before_first_paint(self) -> None:
        """The saved collapsed state is set in <head>, so the layout never jumps."""
        html = self.client.get("/").text
        head = html[: html.index("</head>")]
        assert 'localStorage.getItem("pfai_sidebar") === "collapsed"' in head
        css = self.client.get("/static/styles.css").text
        assert ':root[data-sidebar="collapsed"] .app' in css
