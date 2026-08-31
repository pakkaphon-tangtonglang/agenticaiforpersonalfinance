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

  // Render model output as markdown → Excel-style HTML tables, sanitized.
  // Falls back to escaped text + <br> if the libs are not yet loaded.
  function renderMarkdown(text) {
    const str = String(text == null ? "" : text);
    if (typeof marked === "undefined" || typeof DOMPurify === "undefined") {
      return esc(str).replace(/\n/g, "<br>");
    }
    marked.setOptions({ gfm: true, breaks: true });
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
    upload: "นำเข้าเอกสาร", assets: "สินทรัพย์", eval: "ประเมินผล",
  };
  function switchView(name) {
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("is-active"));
    $("view-" + name).classList.add("is-active");
    document.querySelectorAll(".nav-item").forEach((n) =>
      n.classList.toggle("is-active", n.dataset.view === name));
    $("minibarTitle").textContent = VIEW_TITLES[name] || name;
    if (name === "assets") loadNotifications();
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
      msg.append(intentBadge(cfg));
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
      const cfg = INTENT_CONFIG[intent] || INTENT_CONFIG.unknown;
      assistantBubble.closest(".msg").append(intentBadge(cfg));
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
        const cfg = INTENT_CONFIG[res.intent] || INTENT_CONFIG.unknown;
        assistantBubble.closest(".msg").append(intentBadge(cfg));
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
    $("assetForm").addEventListener("submit", (e) => {
      e.preventDefault();
      fetchAsset();
    });
    $("markNotifBtn").addEventListener("click", markNotificationsRead);
  }

  async function fetchAsset() {
    const symbol = $("assetSymbol").value.trim();
    const type = $("assetType").value;
    if (!symbol) return;
    const btn = $("assetForm").querySelector(".btn");
    btn.classList.add("is-loading");
    btn.textContent = "กำลังดึง…";
    try {
      const res = await api("POST", "/assets/fetch", {
        json: { user_id: state.userId, symbol, fetch_type: type },
      });
      const box = $("assetResult");
      box.hidden = false;
      box.textContent = JSON.stringify(res.result, null, 2);
    } catch (e) {
      toast("ดึงข้อมูลไม่สำเร็จ: " + e.message, true);
    } finally {
      btn.classList.remove("is-loading");
      btn.textContent = "ดึงข้อมูล";
    }
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
          el("span", { class: "notif-time" }, n.created_at || ""),
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

  // ─── Evaluation ───
  function setupEval() {
    $("runEvalBtn").addEventListener("click", runEvaluation);
  }

  async function runEvaluation() {
    const btn = $("runEvalBtn");
    btn.classList.add("is-loading");
    btn.textContent = "กำลังประเมิน… อาจใช้เวลาสักครู่";
    try {
      const res = await api("POST", "/evaluation/run", {});
      const box = $("evalResult");
      box.hidden = false;
      box.textContent = JSON.stringify(res.results, null, 2);
      toast("ประเมินเสร็จแล้ว");
    } catch (e) {
      toast("ประเมินไม่สำเร็จ: " + e.message, true);
    } finally {
      btn.classList.remove("is-loading");
      btn.textContent = "รันการประเมิน";
    }
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
    setupEval();
    renderSuggestions("all");
    await loadConversations();
    if (state.conversationId) selectConversation(state.conversationId, "");
    else enterHeroMode();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
