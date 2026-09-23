/* Personal Finance AI — frontend logic
   Wires all FastAPI endpoints into a single-page app.
   Search-centric hero + thread layout, single accent #134611.
*/
(function () {
  "use strict";

  // Backend API base: configurable via config.js (iHost deploy), same-origin by default.
  const API_BASE = window.FINANCE_API_BASE ?? location.origin;

  // ─── Intent config (mirrors ui/app_constants.py INTENT_CONFIG) ───
  // One accent (#134611) for the primary intent; others muted neutrals.
  const INTENT_CONFIG = {
    tax: { label: "ภาษี", icon: "🧾", color: "#134611" },
    expense: { label: "ค่าใช้จ่าย", icon: "💸", color: "#8a7a6a" },
    asset_monitoring: { label: "ติดตามสินทรัพย์", icon: "📈", color: "#5a6a5a" },
    planning: { label: "วางแผนการเงิน", icon: "🗓️", color: "#6a7a8a" },
    recommendation: { label: "คำแนะนำการเงิน", icon: "💡", color: "#7a6a8a" },
    report: { label: "รายงานการเงิน", icon: "📊", color: "#4a6b3a" },
    general: { label: "ทั่วไป", icon: "💬", color: "#72706b" },
    general_chat: { label: "แชททั่วไป", icon: "💬", color: "#72706b" },
    clarify: { label: "ขอข้อมูลเพิ่ม", icon: "🤔", color: "#8a7a5a" },
    unknown: { label: "ไม่ทราบ", icon: "❓", color: "#92918b" },
    error: { label: "ข้อผิดพลาด", icon: "⚠️", color: "#a8442c" },
  };

  // ─── Suggestion cards (hero home) ───
  const SUGGESTIONS = [
    { cat: "tax", icon: "🧾", title: "คำนวณภาษี", desc: "เงินเดือน 50,000 บาท/เดือน มีลูก 1 คน ซื้อ SSF 100,000",
      query: "คำนวณภาษี เงินเดือน 50,000 บาท/เดือน มีลูก 1 คน ซื้อ SSF 100,000" },
    { cat: "tax", icon: "🧾", title: "ลดหย่อนภาษี", desc: "ซื้อกองทุน LTF/RMF ปีนี้เท่าไหร่ดี",
      query: "แนะนำการลงทุนลดหย่อนภาษีเท่าไหร่ดี" },
    { cat: "expense", icon: "💸", title: "บันทึกค่าอาหาร", desc: "จ่ายค่าอาหาร 350 บาท",
      query: "จ่ายค่าอาหาร 350 บาท" },
    { cat: "expense", icon: "📋", title: "สรุปรายจ่ายเดือนนี้", desc: "ดูรายจ่ายแยกหมวดหมู่ประจำเดือน",
      query: "สรุปรายจ่ายเดือนนี้" },
    { cat: "asset", icon: "📈", title: "เพิ่มหุ้น PTT", desc: "เพิ่ม PTT.BK 100 หุ้น ราคา 35 บาท",
      query: "เพิ่มหุ้น PTT.BK 100 หุ้น ราคา 35 บาท" },
    { cat: "asset", icon: "💼", title: "ดูพอร์ตการลงทุน", desc: "สรุปมูลค่าและกำไรขาดทุนพอร์ต",
      query: "ดูพอร์ตของฉัน" },
    { cat: "planning", icon: "🗓️", title: "วางแผนการออม", desc: "ช่วยวางแผนออมเพื่อเป้าหมาย",
      query: "ช่วยวางแผนการออมให้หน่อย" },
    { cat: "report", icon: "📊", title: "รายงานการเงิน", desc: "สรุปภาพรวมการเงินรายปี",
      query: "สรุปการเงินรายปี" },
  ];

  const THAI_MONTHS = [
    "มกราคม", "กุมภาพันธ์", "มีนาคม", "เมษายน", "พฤษภาคม", "มิถุนายน",
    "กรกฎาคม", "สิงหาคม", "กันยายน", "ตุลาคม", "พฤศจิกายน", "ธันวาคม",
  ];
  const THAI_CATEGORY_LABELS = {
    food: "อาหาร", transport: "เดินทาง", housing: "ที่พักอาศัย",
    health: "สุขภาพ", education: "การศึกษา", shopping: "ช้อปปิ้ง",
    utilities: "สาธารณูปโภค", entertainment: "บันเทิง",
    investment: "ลงทุน", other: "อื่นๆ",
  };

  // Chart palette — accent-led, muted
  const CHART_PALETTE = [
    "#134611", "#4a6b3a", "#7a8a6a", "#b8a86a",
    "#a8442c", "#6a7a8a", "#8a7a6a", "#5a6a5a",
    "#92918b", "#3f7d3f",
  ];
  const CHART_FONT = "'Inter','Sarabun',sans-serif";

  // ─── State ───
  const state = {
    userId: localStorage.getItem("pfai_uid") || "",
    conversationId: localStorage.getItem("pfai_conv") || "",
    streaming: false,
    charts: {},
    mode: "hero", // "hero" | "thread"
    riskProfile: null, // latest risk assessment {risk_category, risk_level, total_score}
    recommendationDisclaimerShown: false, // show the disclaimer once per session
  };

  // ─── DOM helpers ───
  const $ = (id) => document.getElementById(id);
  const el = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "dataset") Object.assign(node.dataset, v);
      else if (k.startsWith("on") && typeof v === "function")
        node.addEventListener(k.slice(2).toLowerCase(), v);
      else if (v !== null && v !== undefined) node.setAttribute(k, v);
    }
    for (const c of children) {
      if (c === null || c === undefined) continue;
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return node;
  };

  // ─── User ID ───
  function ensureUserId() {
    if (!state.userId) {
      state.userId = (crypto.randomUUID
        ? crypto.randomUUID()
        : "web-" + Date.now() + "-" + Math.random().toString(36).slice(2));
      localStorage.setItem("pfai_uid", state.userId);
    }
    const label = $("userIdLabel");
    label.textContent = state.userId.slice(0, 8) + "…";
    label.title = state.userId;
  }

  // ─── Connect LINE ───
  function openLineModal() {
    $("lineCommandText").textContent = "เชื่อมต่อ " + state.userId;
    $("copyLineCmdBtn").textContent = "คัดลอกคำสั่ง";
    $("lineModalNote").hidden = true;
    $("unlinkLineBtn").disabled = false;
    $("lineModal").hidden = false;
  }

  function closeLineModal() {
    $("lineModal").hidden = true;
  }

  async function copyLineCommand() {
    const command = "เชื่อมต่อ " + state.userId;
    try {
      await navigator.clipboard.writeText(command);
      $("copyLineCmdBtn").textContent = "คัดลอกแล้ว ✓";
    } catch (_err) {
      window.prompt("คัดลอกคำสั่ง:", command);
    }
  }

  async function unlinkLineFromWeb() {
    const btn = $("unlinkLineBtn");
    btn.disabled = true;
    try {
      const result = await api("POST", "/line/unlink", { json: { user_id: state.userId } });
      const count = result.unlinked.length;
      showLineModalNote(count > 0
        ? `ยกเลิกการเชื่อมต่อแล้ว (${count} แชท) — แชท LINE จะกลับไปใช้บัญชีเดิมตั้งแต่ข้อความถัดไป`
        : "ไม่พบแชท LINE ที่เชื่อมต่อกับบัญชีนี้อยู่");
    } catch (_err) {
      showLineModalNote("เกิดข้อผิดพลาด กรุณาลองใหม่อีกครั้ง");
      btn.disabled = false;
    }
  }

  function showLineModalNote(text) {
    const note = $("lineModalNote");
    note.textContent = text;
    note.hidden = false;
  }

  function resetUser() {
    localStorage.removeItem("pfai_uid");
    localStorage.removeItem("pfai_conv");
    state.userId = "";
    state.conversationId = "";
    ensureUserId();
    loadConversations();
    clearChat();
    toast("สร้างไอดีใหม่แล้ว");
  }

  // ─── Toast ───
  let toastTimer;
  function toast(message, isError = false) {
    const t = $("toast");
    t.textContent = message;
    t.classList.toggle("error", isError);
    t.hidden = false;
    requestAnimationFrame(() => t.classList.add("show"));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      t.classList.remove("show");
      setTimeout(() => { t.hidden = true; }, 300);
    }, 3200);
  }

  // ─── Format ───
  function thb(value) {
    const num = Number(value || 0);
    const sign = num < 0 ? "-" : "";
    return sign + num.toLocaleString("th-TH", {
      minimumFractionDigits: 0, maximumFractionDigits: 2,
    }) + " ฿";
  }
  function pct(value) {
    return Number(value || 0).toFixed(1) + "%";
  }
  function esc(text) {
    const d = document.createElement("div");
    d.textContent = String(text);
    return d.innerHTML;
  }

  // Thai-readable timestamp, e.g. "13 ก.ย. 2026, 17:45".
  // Non-date strings pass through unchanged (backend may pre-format them).
  function formatThaiDateTime(value) {
    if (value === null || value === undefined || value === "") return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString("th-TH", {
      day: "numeric", month: "short", year: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
  }

  // Remove markdown bold markers so titles never show raw asterisks.
  function stripMarkdownEmphasis(text) {
    return String(text == null ? "" : text).replace(/\*\*/g, "");
  }

  // Hostname of a news link (Google News redirect URLs are long — show just the host).
  function linkHostname(link) {
    try { return new URL(String(link)).hostname; }
    catch (_) { return String(link).slice(0, 60); }
  }

  // Normalize model output before markdown parsing so chat bubbles never
  // show stray blank space: collapse runs of 3+ newlines to one paragraph
  // break, normalize CRLF, and trim the edges. Matches how GPT/Claude
  // render — a single newline is a soft wrap, not a hard break.
  function normalizeForChat(text) {
    return String(text == null ? "" : text)
      .replace(/\r\n/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  // Render model output as markdown → Excel-style HTML tables, sanitized.
  // Falls back to escaped text + <br> if the libs are not yet loaded.
  function renderMarkdown(text) {
    const str = normalizeForChat(text);
    if (typeof marked === "undefined" || typeof DOMPurify === "undefined") {
      return esc(str).replace(/\n/g, "<br>");
    }
    // breaks: false — single newlines wrap softly (like GPT/Claude);
    // hard-break mode turned every wrapped line into a visible blank line.
    marked.setOptions({ gfm: true, breaks: false });
    const html = marked.parse(str);
    // Wrap tables so they can scroll on small screens and pick up grid styling.
    return DOMPurify.sanitize(
      html.replace(/<table/g, '<div class="table-wrap"><table').replace(/<\/table>/g, "</table></div>")
    );
  }

  // ─── API ───
  async function api(method, path, { json, params, form, signal } = {}) {
    const url = new URL(path, API_BASE);
    if (params) for (const [k, v] of Object.entries(params))
      if (v !== null && v !== undefined) url.searchParams.set(k, v);
    const opts = { method, headers: {} };
    if (signal) opts.signal = signal;
    if (json) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(json); }
    if (form) { opts.body = form; }
    const res = await fetch(url, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { const body = await res.json(); detail = body.detail || JSON.stringify(body); } catch (_) { /* noop */ }
      throw new Error(`${res.status}: ${detail}`);
    }
    const text = await res.text();
    return text ? JSON.parse(text) : null;
  }

  // ─── View switching ───
  const VIEW_TITLES = {
    chat: "แชท", dashboard: "แดชบอร์ด",
    upload: "นำเข้าเอกสาร", assets: "สินทรัพย์",
  };
  function switchView(name) {
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("is-active"));
    $("view-" + name).classList.add("is-active");
    document.querySelectorAll(".nav-item").forEach((n) =>
      n.classList.toggle("is-active", n.dataset.view === name));
    $("minibarTitle").textContent = VIEW_TITLES[name] || name;
    if (name === "assets") { loadNotifications(); loadWatchlist(); }
    if (name === "dashboard") loadDashboard();
    if (window.innerWidth <= 760) $("sidebar").classList.remove("open");
  }

  // ─── Hero / Thread mode ───
  function enterHeroMode() {
    state.mode = "hero";
    $("chatHero").hidden = false;
    $("chatThread").hidden = true;
    $("threadComposer").hidden = true;
    $("chatThread").innerHTML = "";
    const input = $("chatInput");
    input.value = "";
    $("chatSend").disabled = true;
    requestAnimationFrame(() => input.focus());
  }

  function enterThreadMode() {
    state.mode = "thread";
    $("chatHero").hidden = true;
    $("chatThread").hidden = false;
    $("threadComposer").hidden = false;
    requestAnimationFrame(() => $("threadInput").focus());
  }

  // ─── Conversations ───
  async function loadConversations() {
    try {
      const convs = await api("GET", "/conversations", { params: { user_id: state.userId } });
      const list = $("convList");
      list.innerHTML = "";
      if (!convs.length) {
        list.append(el("li", { class: "notif-empty" }, "ยังไม่มีบทสนทนา"));
        return;
      }
      for (const c of convs) {
        const li = el("li", {
          class: c.id === state.conversationId ? "is-active" : "",
          dataset: { id: c.id },
          onclick: () => selectConversation(c.id, c.title),
          title: c.title,
        }, c.title || "แชทใหม่");
        list.append(li);
      }
    } catch (e) { /* silent — sidebar not critical */ }
  }

  async function selectConversation(id, title) {
    state.conversationId = id;
    localStorage.setItem("pfai_conv", id);
    clearChat();
    switchView("chat");
    try {
      const msgs = await api("GET", `/conversations/${id}/messages`);
      if (msgs.length) {
        enterThreadMode();
        for (const m of msgs) appendMessage(m.role, m.content, m.intent);
      }
    } catch (e) { toast("โหลดข้อความไม่สำเร็จ", true); }
    loadConversations();
  }

  async function newConversation() {
    try {
      const conv = await api("POST", "/conversations", {
        json: { user_id: state.userId, title: "แชทใหม่" },
      });
      state.conversationId = conv.id;
      localStorage.setItem("pfai_conv", conv.id);
      clearChat();
      switchView("chat");
      loadConversations();
    } catch (e) { toast("สร้างแชทใหม่ไม่สำเร็จ", true); }
  }

  function clearChat() {
    $("chatThread").innerHTML = "";
    enterHeroMode();
  }

  // ─── Chat ───
  function appendMessage(role, content, intent) {
    const thread = $("chatThread");
    const cfg = intent ? INTENT_CONFIG[intent] || INTENT_CONFIG.unknown : null;
    const bubble = el("div", { class: "msg-bubble" });
    bubble.innerHTML = renderMarkdown(content);
    const msg = el("div", { class: `msg ${role}` },
      el("span", { class: "msg-role" }, role === "user" ? "คุณ" : "ผู้ช่วยการเงิน"),
      bubble,
    );
    if (cfg && role === "assistant") {
      appendIntentExtras(msg, intent);
    }
    thread.append(msg);
    thread.scrollTop = thread.scrollHeight;
    return bubble;
  }

  function intentBadge(cfg) {
    return el("span", {
      class: "msg-intent",
      style: `background:${cfg.color}1a;color:${cfg.color};`,
    }, `${cfg.icon} ${cfg.label}`);
  }

  function intentDisclaimer() {
    return el("p", { class: "msg-disclaimer" },
      "AI ให้ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน");
  }

  function appendIntentExtras(msgEl, intent) {
    const cfg = INTENT_CONFIG[intent] || INTENT_CONFIG.unknown;
    msgEl.append(intentBadge(cfg));
    if (intent === "recommendation" && !state.recommendationDisclaimerShown) {
      state.recommendationDisclaimerShown = true;
      msgEl.append(intentDisclaimer());
    }
  }

  async function sendChat(text) {
    if (!text.trim() || state.streaming) return;
    if (state.mode === "hero") enterThreadMode();
    appendMessage("user", text);
    const assistantBubble = appendMessage("assistant", "");
    assistantBubble.classList.add("msg-cursor");
    state.streaming = true;
    setComposerDisabled(true);

    let full = "";
    let intent = "unknown";
    try {
      const params = new URLSearchParams({ query: text, user_id: state.userId });
      if (state.conversationId) params.set("conversation_id", state.conversationId);
      const source = new EventSource(`${API_BASE}/chat/stream?${params}`);

      await new Promise((resolve, reject) => {
        source.addEventListener("message", (ev) => {
          full += JSON.parse(ev.data);
          assistantBubble.innerHTML = renderMarkdown(full);
          $("chatThread").scrollTop = $("chatThread").scrollHeight;
        });
        source.addEventListener("complete", (ev) => {
          try { intent = ev.data || "unknown"; } catch (_) { /* noop */ }
          source.close();
          resolve();
        });
        source.onerror = () => { source.close(); reject(new Error("stream")); };
      });

      assistantBubble.classList.remove("msg-cursor");
      appendIntentExtras(assistantBubble.closest(".msg"), intent);
      loadConversations();
    } catch (e) {
      assistantBubble.classList.remove("msg-cursor");
      assistantBubble.textContent = "เชื่อมต่อสตรีมไม่สำเร็จ กำลังลองใหม่แบบไม่สตรีม…";
      try {
        const res = await api("POST", "/chat", {
          json: {
            query: text, user_id: state.userId,
            conversation_id: state.conversationId || null,
          },
        });
        assistantBubble.innerHTML = renderMarkdown(res.response);
        appendIntentExtras(assistantBubble.closest(".msg"), res.intent);
        loadConversations();
      } catch (e2) {
        assistantBubble.textContent = "ส่งข้อความไม่สำเร็จ โปรดลองอีกครั้ง";
        toast("ส่งข้อความไม่สำเร็จ", true);
      }
    } finally {
      state.streaming = false;
      setComposerDisabled(false);
    }
  }

  function setComposerDisabled(disabled) {
    $("chatSend").disabled = disabled;
    $("chatInput").disabled = disabled;
    $("threadSend").disabled = disabled;
    $("threadInput").disabled = disabled;
    if (!disabled) {
      const focusId = state.mode === "thread" ? "threadInput" : "chatInput";
      $(focusId).focus();
    }
  }

  // ─── Suggestion cards + category filter ───
  function renderSuggestions(cat) {
    const grid = $("suggestionGrid");
    grid.innerHTML = "";
    const items = cat === "all" ? SUGGESTIONS
      : SUGGESTIONS.filter((s) => s.cat === cat);
    for (const s of items) {
      grid.append(el("button", {
        type: "button",
        class: "suggestion-card",
        onclick: () => {
          if (state.streaming) return;
          if (state.mode === "hero") enterThreadMode();
          sendChat(s.query);
        },
      },
        el("span", { class: "suggestion-icon", "aria-hidden": "true" }, s.icon),
        el("span", { class: "suggestion-title" }, s.title),
        el("span", { class: "suggestion-desc" }, s.desc),
      ));
    }
  }

  function setupCategoryNav() {
    document.querySelectorAll(".cat-link").forEach((link) =>
      link.addEventListener("click", () => {
        document.querySelectorAll(".cat-link").forEach((l) => l.classList.remove("is-active"));
        link.classList.add("is-active");
        renderSuggestions(link.dataset.cat);
      }));
  }

  // ─── Dashboard ───
  function initDashboardControls() {
    const now = new Date();
    const yearSel = $("dashYear");
    for (let y = now.getFullYear() - 3; y <= now.getFullYear() + 1; y++) {
      yearSel.append(el("option", { value: y }, String(y + 543)));
      yearSel.lastChild.dataset.greg = y;
    }
    yearSel.value = now.getFullYear();
    const monthSel = $("dashMonth");
    THAI_MONTHS.forEach((m, i) => {
      monthSel.append(el("option", { value: i + 1 }, m));
    });
    monthSel.value = now.getMonth() + 1;
    $("dashLoad").addEventListener("click", loadDashboard);
    yearSel.addEventListener("change", loadDashboard);
    monthSel.addEventListener("change", loadDashboard);
  }

  async function loadDashboard() {
    const year = Number($("dashYear").value);
    const month = Number($("dashMonth").value);
    const btn = $("dashLoad");
    btn.classList.add("is-loading");
    btn.textContent = "กำลังโหลด…";
    try {
      const report = await api("GET", "/dashboard", {
        params: { user_id: state.userId, year, month },
      });
      renderDashboard(report);
      $("dashEmpty").hidden = true;
      $("dashContent").hidden = false;
    } catch (e) {
      toast("โหลดแดชบอร์ดไม่สำเร็จ: " + e.message, true);
    } finally {
      btn.classList.remove("is-loading");
      btn.textContent = "โหลดข้อมูล";
    }
  }

  function renderDashboard(r) {
    const ov = r.monthly_overview || {};
    setStat("statIncome", ov.total_income);
    setStat("statExpense", ov.total_expenses);
    setStat("statNet", ov.net_savings, true);
    $("statRate").textContent = pct(ov.savings_rate);

    renderGauge(r.health_score ?? 0);
    renderExpenseChart(r.expense_breakdown || {});
    renderPortfolio(r.investment_portfolio || {});
    renderGoals(r.goal_progress || {});
    renderTax(r.tax_status || {});
    renderHighlights(r.highlights || []);
  }

  function setStat(id, value, signed = false) {
    const node = $(id);
    node.textContent = thb(value);
    node.classList.remove("positive", "negative");
    if (signed) {
      const num = Number(value || 0);
      if (num > 0) node.classList.add("positive");
      else if (num < 0) node.classList.add("negative");
    }
  }

  function renderGauge(score) {
    const s = Math.max(0, Math.min(100, score));
    const fill = $("gaugeFill");
    const circumference = 251;
    fill.style.strokeDashoffset = String(circumference * (1 - s / 100));
    animateNumber($("gaugeValue"), s);
    const chip = $("healthChip");
    const dot = chip.querySelector(".health-dot");
    dot.className = "health-dot " + (s >= 70 ? "good" : s >= 40 ? "warn" : "bad");
    $("healthValue").textContent = s;
    $("gaugeCaption").textContent =
      s >= 80 ? "สุขภาพการเงินดีเยี่ยม" :
      s >= 60 ? "สุขภาพการเงินดี" :
      s >= 40 ? "ควรปรับปรุง" : "ต้องดูแลด่วน";
  }

  function animateNumber(node, target) {
    const start = 0;
    const duration = 900;
    const t0 = performance.now();
    function tick(now) {
      const p = Math.min(1, (now - t0) / duration);
      node.textContent = Math.round(start + (target - start) * p);
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function renderExpenseChart(breakdown) {
    const cats = breakdown.categories || [];
    const labels = cats.map((c) =>
      THAI_CATEGORY_LABELS[c.category] || c.category || "อื่นๆ");
    const data = cats.map((c) => Number(c.amount || 0));
    destroyChart("expense");
    if (!data.length) {
      $("expenseChart").parentElement.innerHTML = "<p class='notif-empty'>ยังไม่มีข้อมูลรายจ่าย</p>";
      return;
    }
    state.charts.expense = new Chart($("expenseChart"), {
      type: "doughnut",
      data: { labels, datasets: [{
        data,
        backgroundColor: CHART_PALETTE,
        borderWidth: 2, borderColor: "#fdfbfa",
      }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "right",
          labels: { font: { family: CHART_FONT }, color: "#27251e" } } },
      },
    });
  }

  function renderPortfolio(p) {
    $("holdCount").textContent = p.holding_count || 0;
    const tbody = $("portfolioTable").querySelector("tbody");
    tbody.innerHTML = "";
    for (const h of (p.holdings || [])) {
      const gl = Number(h.gain_loss || 0);
      const glClass = gl >= 0 ? "gain" : "loss";
      tbody.append(el("tr", {},
        el("td", {}, h.symbol || "—"),
        el("td", {}, String(h.quantity ?? "—")),
        el("td", {}, thb(h.average_cost)),
        el("td", {}, thb(h.current_value ?? h.total_value)),
        el("td", { class: glClass }, (gl >= 0 ? "+" : "") + thb(gl)),
      ));
    }
    destroyChart("portfolio");
    if ((p.holdings || []).length) {
      state.charts.portfolio = new Chart($("portfolioChart"), {
        type: "bar",
        data: {
          labels: p.holdings.map((h) => h.symbol),
          datasets: [{
            label: "มูลค่า",
            data: p.holdings.map((h) => Number(h.current_value ?? h.total_value ?? 0)),
            backgroundColor: "#134611", borderRadius: 6,
          }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            x: { ticks: { font: { family: CHART_FONT }, color: "#72706b" } },
            y: { ticks: { font: { family: CHART_FONT }, color: "#72706b" } },
          },
        },
      });
    }
  }

  function renderGoals(g) {
    const list = $("goalsList");
    list.innerHTML = "";
    if (!(g.goals || []).length) {
      list.append(el("li", { class: "notif-empty" }, "ยังไม่มีเป้าหมายการเงิน"));
      return;
    }
    for (const goal of g.goals) {
      const pctVal = Math.min(100, Number(goal.percentage || 0));
      list.append(el("li", { class: "goal-item" },
        el("div", { class: "goal-head" },
          el("span", { class: "goal-name" }, goal.name || "เป้าหมาย"),
          el("span", { class: "goal-pct" }, pct(pctVal))),
        el("div", { class: "goal-bar" },
          el("div", { class: "goal-fill", style: `width:${pctVal}%` })),
      ));
    }
  }

  function renderTax(t) {
    const grid = $("taxGrid");
    grid.innerHTML = "";
    const rows = [
      ["สถานะ", t.status === "filed" ? "ยื่นแล้ว" : "ยังไม่ยื่น"],
      ["ปีภาษี", String(t.tax_year || "—")],
      ["รายได้ทั้งปี", thb(t.gross_income)],
      ["ค่าลดหย่อนรวม", thb(t.total_deductions)],
      ["ภาษีที่ต้องจ่าย", thb(t.total_tax)],
      ["อัตราภาษีเฉลี่ย", pct(t.effective_rate)],
    ];
    for (const [k, v] of rows) {
      grid.append(el("div", {}, el("dt", {}, k), el("dd", {}, v)));
    }
  }

  function renderHighlights(highlights) {
    const list = $("highlightsList");
    list.innerHTML = "";
    if (!highlights.length) {
      list.append(el("li", { class: "notif-empty" }, "ไม่มีข้อสังเกตพิเศษ"));
      return;
    }
    for (const h of highlights) {
      list.append(el("li", { class: `highlight-item ${h.highlight_type || "info"}` },
        el("div", {},
          el("div", { class: "highlight-title" }, h.title || ""),
          el("div", { class: "highlight-desc" }, h.description || ""),
        ),
      ));
    }
  }

  function destroyChart(name) {
    if (state.charts[name]) { state.charts[name].destroy(); delete state.charts[name]; }
  }

  // ─── Upload ───
  const RECEIPT_CATEGORIES = {
    food: "อาหาร", transport: "การเดินทาง", housing: "ที่พักอาศัย",
    entertainment: "บันเทิง", utilities: "สาธารณูปโภค", health: "สุขภาพ",
    education: "การศึกษา", shopping: "ช้อปปิ้ง", investment: "ลงทุน", other: "อื่นๆ",
  };
  let uploadMode = "scan"; // "scan" | "bank"

  function setupUpload() {
    const dz = $("dropzone");
    const input = $("fileInput");
    setupDropzone(dz, input);
    setupUploadTabs();
    $("confirmBtn").addEventListener("click", confirmRecheck);
    $("cancelRecheckBtn").addEventListener("click", clearRecheck);
  }

  function setupDropzone(dz, input) {
    dz.addEventListener("click", () => input.click());
    dz.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
    });
    ["dragenter", "dragover"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
    dz.addEventListener("drop", (e) => {
 const f = e.dataTransfer.files[0];
 if (f) uploadFile(f);
    });
    input.addEventListener("change", () => {
 if (input.files[0]) uploadFile(input.files[0]);
    });
  }

  function setupUploadTabs() {
 document.querySelectorAll(".upload-tab").forEach((tab) =>
 tab.addEventListener("click", () => switchUploadMode(tab.dataset.mode)));
  }

  function switchUploadMode(mode) {
 uploadMode = mode;
 document.querySelectorAll(".upload-tab").forEach((t) => {
 const active = t.dataset.mode === mode;
 t.classList.toggle("is-active", active);
 t.setAttribute("aria-selected", active ? "true" : "false");
 });
 $("fileInput").accept = mode === "scan" ? "image/*,.pdf" : ".csv,.xlsx";
 $("dropzoneHint").textContent =
 mode === "scan" ? "รองรับรูปภาพ (JPG/PNG) และ PDF" : "รองรับ .csv และ .xlsx";
 clearRecheck();
 $("uploadResult").hidden = true;
  }

  function isScanFile(file) {
 return (file.type || "").startsWith("image/") || (file.type || "") === "application/pdf"
 || /\.(png|jpe?g|webp|heic|heif|pdf)$/i.test(file.name || "");
  }

  async function uploadFile(file) {
 const dz = $("dropzone");
 dz.classList.add("is-loading");
 try {
 if (uploadMode === "scan" && isScanFile(file)) await scanReceipt(file);
 else if (uploadMode === "bank" && !isScanFile(file)) await importBankStatement(file);
 else toast("ประเภทไฟล์ไม่ตรงกับโหมดที่เลือก", true);
 } catch (e) {
 toast("นำเข้าไม่สำเร็จ: " + e.message, true);
 } finally {
 dz.classList.remove("is-loading");
 }
  }

  async function importBankStatement(file) {
 const form = new FormData();
 form.append("file", file);
 const res = await api("POST", "/upload/bank-statement", {
 params: { user_id: state.userId }, form,
 });
 const r = $("uploadResult");
 r.hidden = false;
 r.textContent = `นำเข้าสำเร็จ — บันทึก ${res.inserted} จาก ${res.total} รายการ`;
 toast(`นำเข้า ${res.inserted} รายการ`);
  }

  async function scanReceipt(file) {
 const form = new FormData();
 form.append("file", file);
 const ctl = new AbortController();
 const timer = setTimeout(() => ctl.abort(), 100000);
 const hint = $("dropzoneHint");
 const prevHint = hint ? hint.textContent : "";
 if (hint) hint.textContent = "กำลังอ่านเอกสารด้วย AI... อาจใช้เวลาสักครู่";
 try {
 const res = await api("POST", "/upload/receipt", {
 params: { user_id: state.userId }, form, signal: ctl.signal,
 });
 $("uploadResult").hidden = true;
 if (!res.drafts || !res.drafts.length) {
 toast("ไม่สามารถอ่านข้อมูลจากเอกสารได้ ลองถ่ายใหม่อีกครั้ง", true);
 return;
 }
 renderRecheck(res.drafts);
 toast(`AI กรอกให้ ${res.drafts.length} รายการ — ตรวจสอบก่อนบันทึก`);
 } catch (e) {
 if (e.name === "AbortError") toast("อ่านเอกสารนานเกินไป กรุณาลองอีกครั้ง", true);
 else throw e;
 } finally {
 clearTimeout(timer);
 if (hint) hint.textContent = prevHint;
 }
  }

  function renderRecheck(drafts) {
 const body = $("recheckBody");
 body.innerHTML = "";
 drafts.forEach((draft) => body.append(buildRecheckRow(draft)));
 $("recheckArea").hidden = false;
  }

  function buildRecheckRow(draft) {
 const row = el("tr", { class: "recheck-row" });
 row.append(
 buildTypeCell(draft.transaction_type),
 buildAmountCell(draft.amount),
 buildDateCell(draft.transaction_date),
 buildCategoryCell(draft.category),
 buildDescCell(draft.description),
 buildRemoveCell(row),
 );
 return row;
  }

  function buildTypeCell(value) {
 const sel = el("select", { class: "field-input recheck-input" });
 [["expense", "รายจ่าย"], ["income", "รายรับ"]].forEach(([k, label]) => {
 const opt = el("option", { value: k }, label);
 if (k === value) opt.selected = true;
 sel.append(opt);
 });
 return el("td", {}, sel);
  }

  function buildCategoryCell(value) {
 const sel = el("select", { class: "field-input recheck-input" });
 Object.entries(RECEIPT_CATEGORIES).forEach(([k, label]) => {
 const opt = el("option", { value: k }, label);
 if (k === value) opt.selected = true;
 sel.append(opt);
 });
 return el("td", {}, sel);
  }

  function buildAmountCell(value) {
 return el("td", {},
 el("input", { type: "number", step: "0.01", min: "0.01", class: "field-input recheck-input",
 dataset: { field: "amount" }, value: String(value || "") }));
  }

  function buildDateCell(value) {
 return el("td", {},
 el("input", { type: "date", class: "field-input recheck-input",
 dataset: { field: "date" }, value: value || "" }));
  }

  function buildDescCell(value) {
 return el("td", {},
 el("input", { type: "text", class: "field-input recheck-input",
 dataset: { field: "desc" }, value: value || "" }));
  }

  function buildRemoveCell(row) {
 return el("td", { class: "recheck-remove-cell" },
 el("button", { type: "button", class: "recheck-remove", "aria-label": "ลบรายการ",
 onclick: () => row.remove() }, "✕"));
  }

  function collectRecheckRows() {
 return Array.from($("recheckBody").querySelectorAll(".recheck-row")).map(readRecheckRow);
  }

  function readRecheckRow(row) {
 const cells = row.querySelectorAll(".recheck-input");
 const type = cells[0].value;
 const amount = cells[1].value;
 const date = cells[2].value;
 const category = cells[3].value;
 const description = cells[4].value;
 return { transaction_type: type, amount, transaction_date: date, category, description };
  }

  async function confirmRecheck() {
    const rows = collectRecheckRows();
    if (!rows.length) { toast("ไม่มีรายการให้บันทึก", true); return; }
    const btn = $("confirmBtn");
    btn.classList.add("is-loading");
    btn.disabled = true;
    try {
      const res = await api("POST", "/transactions/confirm", {
        json: { user_id: state.userId, transactions: rows },
      });
      toast(`บันทึก ${res.inserted} จาก ${res.total} รายการ`);
      clearRecheck();
      alignDashboardFilterTo(rows);
      switchView("dashboard");
    } catch (e) {
      toast("บันทึกไม่สำเร็จ: " + e.message, true);
    } finally {
      btn.classList.remove("is-loading");
      btn.disabled = false;
    }
  }

  // Point the dashboard year/month filter at the month of the most recent
  // confirmed transaction so freshly-saved OCR entries are visible without
  // the user having to change the filter by hand.
  function alignDashboardFilterTo(rows) {
    const dates = rows.map((r) => r.transaction_date).filter(Boolean).sort();
    const latest = dates[dates.length - 1];
    if (!latest) return;
    const parts = latest.split("-").map(Number);
    if (parts.length < 2 || parts.some((n) => Number.isNaN(n))) return;
    $("dashYear").value = String(parts[0]);
    $("dashMonth").value = String(parts[1]);
  }

  function clearRecheck() {
 $("recheckBody").innerHTML = "";
 $("recheckArea").hidden = true;
  }

  // ─── Assets ───
  function setupAssets() {
    $("assetSearchForm").addEventListener("submit", (e) => {
      e.preventDefault();
      searchAssets();
    });
    $("markNotifBtn").addEventListener("click", markNotificationsRead);
  }

  // Show which asset the upcoming result belongs to.
  function showSelectedAsset(symbol, name) {
    const display = $("selectedAsset");
    display.textContent = [name, symbol].filter(Boolean).join(" · ");
    display.hidden = false;
  }

  async function fetchAssetData(symbol, name, btn) {
    if (!symbol) return;
    showSelectedAsset(symbol, name);
    if (btn) {
      btn.classList.add("is-loading");
      btn.textContent = "กำลังดึง…";
    }
    try {
      const res = await api("POST", "/assets/fetch", {
        json: { user_id: state.userId, symbol, fetch_type: "all" },
      });
      renderAssetFetchResult(res.result || {});
    } catch (e) {
      toast("ดึงข้อมูลไม่สำเร็จ: " + e.message, true);
    } finally {
      if (btn) {
        btn.classList.remove("is-loading");
        btn.textContent = "ดึงข้อมูล";
      }
    }
  }

  // Render POST /assets/fetch result as readable cards — never raw JSON.
  function renderAssetFetchResult(result) {
    const box = $("assetResult");
    box.innerHTML = "";
    box.append(el("span", { class: "asset-result-symbol" }, result.symbol || ""));
    const price = result.price == null ? "" : String(result.price);
    if (price) box.append(el("p", { class: "asset-price-line" }, `ราคาล่าสุด: ${price}`));
    if (result.error) box.append(el("p", { class: "asset-result-error" }, result.error));
    const news = Array.isArray(result.news) ? result.news : [];
    if (news.length) box.append(buildNewsSection(news));
    if (!price && !result.error && !news.length) {
      box.append(el("p", { class: "notif-empty" }, "ไม่พบข้อมูลสำหรับสินทรัพย์นี้"));
    }
    box.hidden = false;
  }

  function buildNewsSection(news) {
    const list = el("ul", { class: "asset-news-list" });
    for (const item of news) list.append(buildNewsItem(item));
    return el("div", { class: "asset-news-section" },
      el("p", { class: "asset-news-heading" }, "ข่าวล่าสุด"),
      list,
    );
  }

  function buildNewsItem(item) {
    const card = el("li", { class: "asset-news-card" },
      el("span", { class: "asset-news-title" },
        stripMarkdownEmphasis(item.title) || "ไม่มีหัวข้อ"),
      el("span", { class: "asset-news-meta" }, newsMetaLine(item)),
    );
    if (item.link) card.append(buildNewsLink(item.link));
    return card;
  }

  function newsMetaLine(item) {
    return [stripMarkdownEmphasis(item.source), formatThaiDateTime(item.published_at)]
      .filter(Boolean).join(" · ");
  }

  function buildNewsLink(link) {
    return el("a", {
      class: "asset-news-link", href: link,
      target: "_blank", rel: "noopener",
    }, linkHostname(link));
  }

  async function loadNotifications() {
    try {
      const notifs = await api("GET", "/assets/notifications", { params: { user_id: state.userId } });
      const list = $("notifList");
      list.innerHTML = "";
      if (!notifs.length) {
        list.append(el("li", { class: "notif-empty" }, "ยังไม่มีการแจ้งเตือน"));
        return;
      }
      for (const n of notifs) {
        list.append(el("li", { class: "notif-item" },
          el("span", { class: "notif-symbol" }, n.symbol || "—"),
          el("span", { class: "notif-text" }, n.message || ""),
          el("span", { class: "notif-time" }, formatThaiDateTime(n.created_at)),
        ));
      }
    } catch (e) { /* silent */ }
  }

  async function markNotificationsRead() {
    try {
      await api("POST", "/assets/notifications/read", { params: { user_id: state.userId } });
      loadNotifications();
      toast("ทำเครื่องหมายอ่านแล้ว");
    } catch (e) { toast("ไม่สำเร็จ", true); }
  }

  // ─── Watchlist (tracked assets) ───
  async function loadWatchlist() {
    try {
      const rows = await api("GET", "/assets/watchlist", { params: { user_id: state.userId } });
      renderWatchlist(Array.isArray(rows) ? rows : []);
    } catch (e) { /* silent — same as notifications */ }
  }

  function renderWatchlist(rows) {
    const list = $("watchlistList");
    list.innerHTML = "";
    if (!rows.length) {
      list.append(el("li", { class: "notif-empty" }, "ยังไม่มีสินทรัพย์ที่ติดตาม"));
      return;
    }
    for (const row of rows) list.append(buildWatchlistRow(row));
  }

  function buildWatchlistRow(row) {
    return el("li", {
      class: "watchlist-item watchlist-item-clickable",
      title: "คลิกเพื่อดึงราคาและข่าวล่าสุด",
      onclick: () => fetchAssetData(row.symbol, row.name),
    },
      el("span", { class: "watchlist-symbol" }, row.symbol || "—"),
      el("span", { class: "watchlist-name" }, row.name || ""),
      el("button", {
        type: "button", class: "watchlist-remove",
        "aria-label": `ตัด ${row.name || row.symbol} ออกจากรายการ`,
        onclick: (e) => { e.stopPropagation(); removeWatchlistAsset(row); },
      }, "ตัดออก"),
    );
  }

  async function removeWatchlistAsset(row) {
    if (!row.id) {
      toast("ไม่พบรหัสสินทรัพย์ในรายการ โปรดโหลดหน้าใหม่อีกครั้ง", true);
      return;
    }
    try {
      await api("DELETE", `/assets/watchlist/${row.id}`, { params: { user_id: state.userId } });
      toast(`ตัด ${row.name || row.symbol} ออกจากรายการแล้ว`);
      loadWatchlist();
    } catch (e) { toast("ตัดออกไม่สำเร็จ: " + e.message, true); }
  }

  async function addToWatchlist(symbol, name) {
    try {
      const res = await api("POST", "/assets/watchlist",
        { json: { user_id: state.userId, symbol, name } });
      toast(res && res.status === "already_exists"
        ? `${name || symbol} อยู่ในรายการติดตามอยู่แล้ว`
        : `ติดตาม ${name} (${symbol}) แล้ว`);
      loadWatchlist();
    } catch (e) {
      // 422 bodies carry an actionable Thai message — strip the status prefix.
      toast(e.message.replace(/^\d+:\s*/, ""), true);
    }
  }

  // ─── Asset search (free text) ───
  async function searchAssets() {
    const query = $("assetSearchInput").value.trim();
    if (!query) return;
    const btn = $("assetSearchBtn");
    btn.classList.add("is-loading");
    btn.textContent = "กำลังค้นหา…";
    try {
      const res = await api("GET", "/assets/search", { params: { query } });
      renderAssetSearchResults(res.results || []);
    } catch (e) {
      toast("ค้นหาไม่สำเร็จ: " + e.message, true);
    } finally {
      btn.classList.remove("is-loading");
      btn.textContent = "ค้นหา";
    }
  }

  function renderAssetSearchResults(results) {
    const box = $("assetSearchResults");
    box.innerHTML = "";
    if (!results.length) {
      box.append(el("p", { class: "asset-search-empty" }, "ไม่พบสินทรัพย์ที่ตรงกัน"));
      box.hidden = false;
      return;
    }
    for (const r of results) box.append(buildAssetSearchCard(r));
    box.hidden = false;
  }

  // One card = info area + ดึงข้อมูล (fetch) + ติดตาม (add to the watchlist).
  function buildAssetSearchCard(result) {
    return el("div", { class: "asset-result-card", dataset: { symbol: result.symbol } },
      el("span", { class: "asset-result-main" },
        el("span", { class: "asset-result-name" }, result.name),
        el("span", { class: "asset-result-meta" },
          `${result.symbol} · ${result.exchange} · ${result.type}`),
      ),
      el("div", { class: "asset-card-actions" },
        el("button", {
          type: "button", class: "asset-fetch-btn",
          onclick: (e) => fetchAssetData(result.symbol, result.name, e.currentTarget),
        }, "ดึงข้อมูล"),
        el("button", {
          type: "button", class: "asset-track-btn",
          onclick: () => addToWatchlist(result.symbol, result.name),
        }, "ติดตาม"),
      ),
    );
  }

  // ─── Risk assessment (SEC suitability onboarding) ───
  const RISK_QUESTIONS = [
    { id: 1, text: "ปัจจุบันท่านอายุเท่าใด", multiSelect: false, scored: true,
      options: [
        { key: "ก", text: "มากกว่า 55 ปี" },
        { key: "ข", text: "45 – 55 ปี" },
        { key: "ค", text: "35 – 44 ปี" },
        { key: "ง", text: "น้อยกว่า 35 ปี" },
      ] },
    { id: 2, text: "ปัจจุบันท่านมีภาระทางการเงินและค่าใช้จ่ายประจำ เช่น ค่าผ่อนบ้าน รถ ค่าใช้จ่ายส่วนตัว และค่าเลี้ยงดูครอบครัว เป็นสัดส่วนเท่าใด", multiSelect: false, scored: true,
      options: [
        { key: "ก", text: "มากกว่าร้อยละ 75 ของรายได้ทั้งหมด" },
        { key: "ข", text: "ระหว่างร้อยละ 50 ถึงร้อยละ 75 ของรายได้ทั้งหมด" },
        { key: "ค", text: "ระหว่างร้อยละ 25 ถึงร้อยละ 50 ของรายได้ทั้งหมด" },
        { key: "ง", text: "น้อยกว่าร้อยละ 25 ของรายได้ทั้งหมด" },
      ] },
    { id: 3, text: "ท่านมีสถานภาพทางการเงินในปัจจุบันอย่างไร", multiSelect: false, scored: true,
      options: [
        { key: "ก", text: "มีทรัพย์สินน้อยกว่าหนี้สิน" },
        { key: "ข", text: "มีทรัพย์สินเท่ากับหนี้สิน" },
        { key: "ค", text: "มีทรัพย์สินมากกว่าหนี้สิน" },
        { key: "ง", text: "มีความมั่นใจว่ามีเงินออมหรือเงินลงทุนเพียงพอสำหรับการใช้ชีวิตหลังเกษียณอายุแล้ว" },
      ] },
    { id: 4, text: "ท่านเคยมีประสบการณ์หรือมีความรู้ในการลงทุนในทรัพย์สินกลุ่มใดต่อไปนี้บ้าง (เลือกได้มากกว่า 1 ข้อ)", multiSelect: true, scored: true,
      options: [
        { key: "ก", text: "เงินฝากธนาคาร" },
        { key: "ข", text: "พันธบัตรรัฐบาล หรือกองทุนรวมพันธบัตรรัฐบาล" },
        { key: "ค", text: "หุ้นกู้ หรือกองทุนรวมตราสารหนี้" },
        { key: "ง", text: "หุ้นสามัญ หรือกองทุนรวมหุ้น หรือสินทรัพย์อื่นที่มีความเสี่ยงสูง" },
      ] },
    { id: 5, text: "ระยะเวลาที่ท่านคาดว่าจะไม่มีความจำเป็นต้องใช้เงินลงทุนนี้", multiSelect: false, scored: true,
      options: [
        { key: "ก", text: "ไม่เกิน 1 ปี" },
        { key: "ข", text: "1 ถึง 3 ปี" },
        { key: "ค", text: "3 ถึง 5 ปี" },
        { key: "ง", text: "มากกว่า 5 ปี" },
      ] },
    { id: 6, text: "วัตถุประสงค์หลักในการลงทุนของท่านคือ", multiSelect: false, scored: true,
      options: [
        { key: "ก", text: "เน้นเงินต้นต้องปลอดภัยและได้รับผลตอบแทนสม่ำเสมอแต่ต่ำ" },
        { key: "ข", text: "เน้นโอกาสได้รับผลตอบแทนที่สม่ำเสมอ แต่อาจเสี่ยงที่จะสูญเสียเงินต้นได้บ้าง" },
        { key: "ค", text: "เน้นโอกาสได้รับผลตอบแทนที่สูงขึ้น แต่อาจเสี่ยงที่จะสูญเสียเงินต้นได้มากขึ้น" },
        { key: "ง", text: "เน้นผลตอบแทนสูงสุดในระยะยาว แต่อาจเสี่ยงที่จะสูญเงินต้นส่วนใหญ่ได้" },
      ] },
    { id: 7, text: "เมื่อพิจารณารูปแสดงตัวอย่างผลตอบแทนของกลุ่มการลงทุนที่อาจเกิดขึ้น ท่านเต็มใจที่จะลงทุนในกลุ่มการลงทุนใดมากที่สุด", multiSelect: false, scored: true,
      options: [
        { key: "ก", text: "กลุ่มการลงทุนที่ 1 มีโอกาสได้รับผลตอบแทน 2.5% โดยไม่ขาดทุนเลย" },
        { key: "ข", text: "กลุ่มการลงทุนที่ 2 มีโอกาสได้รับผลตอบแทนสูงสุด 7% แต่อาจมีผลขาดทุนได้ถึง 1%" },
        { key: "ค", text: "กลุ่มการลงทุนที่ 3 มีโอกาสได้รับผลตอบแทนสูงสุด 15% แต่อาจมีผลขาดทุนได้ถึง 5%" },
        { key: "ง", text: "กลุ่มการลงทุนที่ 4 มีโอกาสได้รับผลตอบแทนสูงสุด 25% แต่อาจมีผลขาดทุนได้ถึง 15%" },
      ] },
    { id: 8, text: "ถ้าท่านเลือกลงทุนในทรัพย์สินที่มีโอกาสได้รับผลตอบแทนมาก แต่มีโอกาสขาดทุนสูงด้วยเช่นกัน ท่านจะรู้สึกอย่างไร", multiSelect: false, scored: true,
      options: [
        { key: "ก", text: "กังวลและตื่นตระหนกกลัวขาดทุน" },
        { key: "ข", text: "ไม่สบายใจแต่พอเข้าใจได้บ้าง" },
        { key: "ค", text: "เข้าใจและรับความผันผวนได้ในระดับหนึ่ง" },
        { key: "ง", text: "ไม่กังวลกับโอกาสขาดทุนสูง และหวังกับผลตอบแทนที่อาจจะได้รับสูงขึ้น" },
      ] },
    { id: 9, text: "ท่านจะรู้สึกกังวลหรือรับไม่ได้ เมื่อมูลค่าเงินลงทุนของท่านมีการปรับตัวลดลงในสัดส่วนเท่าใด", multiSelect: false, scored: true,
      options: [
        { key: "ก", text: "5% หรือน้อยกว่า" },
        { key: "ข", text: "มากกว่า 5% ถึง 10%" },
        { key: "ค", text: "มากกว่า 10% ถึง 20%" },
        { key: "ง", text: "มากกว่า 20% ขึ้นไป" },
      ] },
    { id: 10, text: "หากปีที่แล้วท่านลงทุนไป 100,000 บาท ปีนี้ท่านพบว่ามูลค่าเงินลงทุนลดลงเหลือ 85,000 บาท ท่านจะทำอย่างไร", multiSelect: false, scored: true,
      options: [
        { key: "ก", text: "ตกใจ และต้องการขายการลงทุนที่เหลือทิ้ง" },
        { key: "ข", text: "กังวลใจ และจะปรับเปลี่ยนการลงทุนบางส่วนไปในทรัพย์สินที่เสี่ยงน้อยลง" },
        { key: "ค", text: "อดทนถือต่อไปได้ และรอผลตอบแทนปรับตัวกลับมา" },
        { key: "ง", text: "ยังมั่นใจ เพราะเข้าใจว่าต้องลงทุนระยะยาว และจะเพิ่มเงินลงทุนในแบบเดิมเพื่อเฉลี่ยต้นทุน" },
      ] },
    { id: 11, text: "หากการลงทุนในอนุพันธ์และหุ้นกู้อนุพันธ์ประสบความสำเร็จ ท่านจะได้รับผลตอบแทนในอัตราที่สูงมาก แต่หากการลงทุนล้มเหลว ท่านอาจจะสูญเงินลงทุนทั้งหมด และอาจต้องลงเงินชดเชยเพิ่มบางส่วน ท่านยอมรับได้เพียงใด", multiSelect: false, scored: false,
      options: [
        { key: "ก", text: "ไม่ได้" },
        { key: "ข", text: "ได้บ้าง" },
        { key: "ค", text: "ได้" },
      ] },
    { id: 12, text: "นอกเหนือจากความเสี่ยงในการลงทุนแล้ว ท่านสามารถรับความเสี่ยงด้านอัตราแลกเปลี่ยนได้เพียงใด", multiSelect: false, scored: false,
      options: [
        { key: "ก", text: "ไม่ได้" },
        { key: "ข", text: "ได้บ้าง" },
        { key: "ค", text: "ได้" },
      ] },
  ];
  const RISK_SCORED_COUNT = 10; // Q1-Q10 one per step; Q11+Q12 share the last step
  const riskState = { step: 0, answers: {} };

  function setupRiskAssessment() {
    $("riskBackBtn").addEventListener("click", backRiskStep);
    $("riskNextBtn").addEventListener("click", nextRiskStep);
    $("riskSkipLink").addEventListener("click", closeRiskModal);
    $("riskRetakeBtn").addEventListener("click", openRiskModal);
    $("riskQuestionBox").addEventListener("change", updateRiskNextState);
  }

  async function maybeShowRiskAssessment() {
    try {
      const res = await api("GET", "/risk-assessment/latest", {
        params: { user_id: state.userId },
      });
      state.riskProfile = res.assessment;
      renderRiskProfileCard();
      if (!res.assessment) openRiskModal();
    } catch (e) { /* silent — onboarding check must not block the app */ }
  }

  function openRiskModal() {
    riskState.step = 0;
    riskState.answers = {};
    $("riskModalFoot").hidden = false;
    $("riskSkipLink").hidden = false;
    $("riskModal").hidden = false;
    renderRiskStep();
  }

  function closeRiskModal() {
    $("riskModal").hidden = true;
  }

  function currentRiskQuestions() {
    if (riskState.step < RISK_SCORED_COUNT) return [RISK_QUESTIONS[riskState.step]];
    return [RISK_QUESTIONS[10], RISK_QUESTIONS[11]];
  }

  function renderRiskStep() {
    const questions = currentRiskQuestions();
    $("riskStepLabel").textContent = `คำถาม ${riskState.step + 1}/${RISK_SCORED_COUNT + 1}`;
    const box = $("riskQuestionBox");
    box.innerHTML = "";
    const isPair = questions.length > 1;
    for (const q of questions) box.append(buildRiskQuestion(q, isPair));
    updateRiskNextState();
  }

  function buildRiskQuestion(question, isPair) {
    const wrap = el("div", { class: "risk-question" + (isPair ? " is-pair" : "") });
    const optional = !question.scored ? " (ไม่บังคับ)" : "";
    wrap.append(el("p", { class: "risk-question-text" }, `${question.id}. ${question.text}${optional}`));
    const inputType = question.multiSelect ? "checkbox" : "radio";
    for (const opt of question.options) {
      const input = el("input", { type: inputType, name: `risk-q-${question.id}`, value: opt.key });
      wrap.append(el("label", { class: "risk-option" },
        input,
        el("span", { class: "risk-option-text" }, opt.text),
      ));
    }
    return wrap;
  }

  function riskCheckedValues(questionId) {
    const name = `risk-q-${questionId}`;
    return Array.from($("riskQuestionBox").querySelectorAll(`input[name="${name}"]:checked`))
      .map((input) => input.value);
  }

  function updateRiskNextState() {
    const box = $("riskQuestionBox");
    box.querySelectorAll(".risk-option").forEach((option) => {
      const input = option.querySelector("input");
      option.classList.toggle("is-selected", input.checked);
    });
    const questions = currentRiskQuestions();
    const answered = questions.every((q) => riskCheckedValues(q.id).length > 0);
    const isOptionalStep = riskState.step >= RISK_SCORED_COUNT;
    $("riskNextBtn").disabled = !isOptionalStep && !answered;
    $("riskNextBtn").textContent = isOptionalStep ? "ส่งคำตอบ" : "ถัดไป";
    $("riskBackBtn").disabled = riskState.step === 0;
  }

  function collectRiskStepAnswers() {
    for (const q of currentRiskQuestions()) {
      const checked = riskCheckedValues(q.id);
      if (checked.length) riskState.answers[String(q.id)] = q.multiSelect ? checked : checked[0];
      else delete riskState.answers[String(q.id)];
    }
  }

  function nextRiskStep() {
    collectRiskStepAnswers();
    if (riskState.step >= RISK_SCORED_COUNT) {
      submitRiskAssessment();
      return;
    }
    riskState.step += 1;
    renderRiskStep();
  }

  function backRiskStep() {
    collectRiskStepAnswers();
    if (riskState.step === 0) return;
    riskState.step -= 1;
    renderRiskStep();
  }

  async function submitRiskAssessment() {
    const btn = $("riskNextBtn");
    btn.disabled = true;
    btn.textContent = "กำลังส่ง…";
    try {
      const res = await api("POST", "/risk-assessment/submit", {
        json: { user_id: state.userId, answers: riskState.answers },
      });
      state.riskProfile = {
        risk_category: res.risk_category,
        risk_level: res.risk_level,
        total_score: res.total_score,
      };
      renderRiskProfileCard();
      renderRiskResult(res);
    } catch (e) {
      toast(e.message.replace(/^\d+:\s*/, ""), true);
      updateRiskNextState();
    }
  }

  function renderRiskResult(res) {
    $("riskStepLabel").textContent = "ผลการประเมิน";
    $("riskModalFoot").hidden = true;
    $("riskSkipLink").hidden = true;
    const box = $("riskQuestionBox");
    box.innerHTML = "";
    box.append(
      el("div", { class: "risk-result-badge" }, res.risk_category),
      el("p", { class: "risk-result-level" },
        `ระดับ ${res.risk_level}/5 · คะแนนรวม ${res.total_score}`),
      buildRiskAllocationTable(res.allocation),
      el("p", { class: "risk-result-footnote" }, res.allocation.footnote),
      el("p", { class: "risk-result-disclaimer" },
        "เป็นการประเมินเบื้องต้น ไม่ใช่คำแนะนำการลงทุนอย่างเป็นทางการ"),
      el("button", { class: "btn risk-start-btn", type: "button", onclick: closeRiskModal },
        "เริ่มใช้งาน"),
    );
  }

  function buildRiskAllocationTable(allocation) {
    const head = el("tr", {}, ...allocation.columns.map((c) => el("th", {}, c)));
    const body = el("tr", {}, ...allocation.row.map((v) => el("td", {}, v)));
    return el("div", { class: "table-wrap" },
      el("table", { class: "risk-allocation-table" }, el("thead", {}, head), el("tbody", {}, body)));
  }

  function renderRiskProfileCard() {
    const card = $("riskProfileCard");
    if (!state.riskProfile) {
      card.hidden = true;
      return;
    }
    $("riskProfileLevel").textContent =
      `${state.riskProfile.risk_category} (ระดับ ${state.riskProfile.risk_level})`;
    card.hidden = false;
  }

  // ─── Composer (hero search + thread composer) ───
  function wireTextarea(inputId, sendId, formId) {
    const input = $(inputId);
    const send = $(sendId);
    input.addEventListener("input", () => {
      send.disabled = !input.value.trim() || state.streaming;
      input.style.height = "auto";
      input.style.height = Math.min(140, input.scrollHeight) + "px";
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!send.disabled) $(formId).requestSubmit();
      }
    });
    $(formId).addEventListener("submit", (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      input.style.height = "auto";
      send.disabled = true;
      sendChat(text);
    });
  }

  function setupComposer() {
    wireTextarea("chatInput", "chatSend", "chatForm");
    wireTextarea("threadInput", "threadSend", "threadComposer");
  }

  // ─── Sidebar ───
  function setupSidebar() {
    document.querySelectorAll(".nav-item").forEach((n) =>
      n.addEventListener("click", () => switchView(n.dataset.view)));
    $("newChatBtn").addEventListener("click", newConversation);
    $("resetUserBtn").addEventListener("click", resetUser);
    $("connectLineBtn").addEventListener("click", openLineModal);
    $("closeLineModalBtn").addEventListener("click", closeLineModal);
    $("copyLineCmdBtn").addEventListener("click", copyLineCommand);
    $("unlinkLineBtn").addEventListener("click", unlinkLineFromWeb);
    $("healthChip").addEventListener("click", () => switchView("dashboard"));
    $("menuToggle").addEventListener("click", () => $("sidebar").classList.toggle("open"));
  }

  // ─── Init ───
  async function init() {
    ensureUserId();
    initDashboardControls();
    setupSidebar();
    setupComposer();
    setupCategoryNav();
    setupUpload();
    setupAssets();
    setupRiskAssessment();
    maybeShowRiskAssessment();
    renderSuggestions("all");
    await loadConversations();
    if (state.conversationId) selectConversation(state.conversationId, "");
    else enterHeroMode();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
