/* Personal Finance AI — Guru tab
   Chat with one guru lens, or ask the whole panel and compare.
   Reuses app.js helpers via window.FinanceApp. No photos of real people:
   cards use initials only, and every answer carries a disclaimer.
*/
(function () {
  "use strict";

  const PANEL_ID = "panel";
  const TITLE_MAX = 50;
  const PANEL_SUGGESTIONS = [
    "ตลาดหุ้นตกหนัก ควรขายหรือซื้อเพิ่ม?",
    "เงินเดือน 30,000 ควรเริ่มลงทุนอย่างไร?",
    "ควรมีเงินสำรองฉุกเฉินเท่าไร?",
  ];

  const app = () => window.FinanceApp;
  const $ = (id) => document.getElementById(id);

  const guru = {
    loaded: false,
    wired: false,
    gurus: [],
    byId: {},
    mode: "single", // "single" | "panel"
    selectedId: "",
    panelIds: new Set(),
    conversationId: "",
    streaming: false,
  };

  // ─── Helpers ───
  function initials(name) {
    const latin = name.match(/[A-Z]/g);
    if (latin && latin.length >= 2) return latin.slice(0, 2).join("");
    return name.replace(/^(ดร\.|โค้ช)/, "").trim().slice(0, 2);
  }

  function guruName(guruId) {
    if (guruId === PANEL_ID) return "ถามทั้งวง";
    return guru.byId[guruId] ? guru.byId[guruId].display_name : guruId;
  }

  function sourceLinks(text) {
    const urls = [...new Set(text.match(/https?:\/\/[^\s|\])>"']+/g) || [])];
    if (!urls.length) return null;
    const el = app().el;
    return el("div", { class: "guru-sources" },
      el("span", { class: "guru-sources-label" }, "แหล่งที่มา:"),
      ...urls.map((url) => el("a", {
        class: "guru-source", href: url, target: "_blank", rel: "noopener noreferrer",
      }, new URL(url).hostname.replace(/^www\./, ""))));
  }

  function scrollThread() {
    const thread = $("guruThread");
    thread.scrollTop = thread.scrollHeight;
  }

  // ─── Messages ───
  function appendMessage(role, content, guruId) {
    const el = app().el;
    const bubble = el("div", { class: "msg-bubble" });
    bubble.innerHTML = app().renderMarkdown(content);
    const label = role === "user" ? "คุณ" : `มุมมองแบบ ${guruName(guruId)}`;
    const msg = el("div", { class: `msg ${role}` }, el("span", { class: "msg-role" }, label), bubble);
    $("guruThread").append(msg);
    scrollThread();
    return bubble;
  }

  function finishAnswer(bubble, text) {
    bubble.classList.remove("msg-cursor");
    const links = sourceLinks(text);
    if (links) bubble.closest(".msg").append(links);
  }

  // ─── Cards / chips ───
  function renderCards() {
    const el = app().el;
    const cards = $("guruCards");
    cards.innerHTML = "";
    for (const g of guru.gurus) {
      const active = guru.mode === "single" ? g.guru_id === guru.selectedId : guru.panelIds.has(g.guru_id);
      cards.append(el("button", {
        type: "button",
        class: "guru-card" + (active ? " is-active" : ""),
        "aria-pressed": String(active),
        onclick: () => onCardClick(g.guru_id),
      },
      el("span", { class: "guru-avatar", "aria-hidden": "true" }, initials(g.display_name)),
      el("span", { class: "guru-card-text" },
        el("span", { class: "guru-card-name" }, g.display_name),
        el("span", { class: "guru-card-tagline" }, g.tagline))));
    }
  }

  function renderChips() {
    const el = app().el;
    const chips = $("guruChips");
    chips.innerHTML = "";
    const selected = guru.byId[guru.selectedId];
    const questions = guru.mode === "panel" ? PANEL_SUGGESTIONS : (selected ? selected.typical_questions : []);
    for (const question of questions) {
      chips.append(el("button", {
        type: "button", class: "guru-chip",
        onclick: () => { if (!guru.streaming) send(question); },
      }, question));
    }
  }

  function onCardClick(guruId) {
    if (guru.streaming) return;
    if (guru.mode === "single") {
      if (guru.selectedId !== guruId) startFresh();
      guru.selectedId = guruId;
    } else if (guru.panelIds.has(guruId) && guru.panelIds.size > 1) {
      guru.panelIds.delete(guruId);
    } else {
      guru.panelIds.add(guruId);
    }
    renderCards();
    renderChips();
  }

  function setMode(mode) {
    if (guru.streaming || guru.mode === mode) return;
    guru.mode = mode;
    document.querySelectorAll(".guru-mode-btn").forEach((b) => {
      const on = b.dataset.mode === mode;
      b.classList.toggle("is-active", on);
      b.setAttribute("aria-selected", String(on));
    });
    startFresh();
    renderCards();
    renderChips();
  }

  function startFresh() {
    guru.conversationId = "";
    $("guruThread").innerHTML = "";
    $("guruHistory").value = "";
  }

  // ─── History ───
  async function loadHistory() {
    const select = $("guruHistory");
    select.innerHTML = "";
    select.append(new Option("ประวัติแชทกูรู…", ""));
    try {
      const convs = await app().api("GET", "/conversations", {
        params: { user_id: app().getUserId(), kind: "guru" },
      });
      for (const conv of convs || []) {
        select.append(new Option(`${guruName(conv.guru_id)} · ${conv.title}`, conv.id));
        select.lastChild.dataset.guruId = conv.guru_id || "";
      }
    } catch (e) { /* history is optional */ }
  }

  async function openConversation(conversationId, guruId) {
    if (!conversationId || guru.streaming) return;
    setMode(guruId === PANEL_ID ? "panel" : "single");
    if (guruId !== PANEL_ID) guru.selectedId = guruId;
    guru.conversationId = conversationId;
    $("guruHistory").value = conversationId;
    renderCards();
    renderChips();
    try {
      const messages = await app().api("GET", `/conversations/${conversationId}/messages`);
      $("guruThread").innerHTML = "";
      for (const m of messages || []) {
        const msgGuru = (m.intent || "").replace(/^guru:/, "");
        const bubble = appendMessage(m.role, m.content, msgGuru === "synthesis" ? "" : msgGuru);
        if (msgGuru === "synthesis") bubble.closest(".msg").querySelector(".msg-role").textContent = "สรุปเปรียบเทียบ";
        if (m.role === "assistant") finishAnswer(bubble, m.content);
      }
    } catch (e) {
      app().toast("โหลดประวัติแชทไม่สำเร็จ", true);
    }
  }

  async function ensureConversation(text, guruId) {
    if (guru.conversationId) return;
    const conv = await app().api("POST", "/conversations", {
      json: {
        user_id: app().getUserId(),
        title: text.slice(0, TITLE_MAX),
        kind: "guru",
        guru_id: guruId,
      },
    });
    guru.conversationId = conv.id;
    loadHistory().then(() => { $("guruHistory").value = guru.conversationId; });
  }

  // ─── Streaming: one guru ───
  function streamSingle(text) {
    const bubble = appendMessage("assistant", "", guru.selectedId);
    bubble.classList.add("msg-cursor");
    const params = new URLSearchParams({
      query: text, user_id: app().getUserId(),
      guru_id: guru.selectedId, conversation_id: guru.conversationId,
    });
    let full = "";
    return new Promise((resolve) => {
      const source = new EventSource(`${app().apiBase}/guru/chat/stream?${params}`);
      source.addEventListener("message", (ev) => {
        full += JSON.parse(ev.data);
        bubble.innerHTML = app().renderMarkdown(full);
        scrollThread();
      });
      source.addEventListener("complete", () => { source.close(); finishAnswer(bubble, full); resolve(); });
      source.onerror = () => {
        source.close();
        finishAnswer(bubble, full);
        if (!full) bubble.textContent = "กูรูตอบไม่สำเร็จ โปรดลองอีกครั้ง";
        resolve();
      };
    });
  }

  // ─── Streaming: panel ───
  function panelCard(guruId) {
    const el = app().el;
    const body = el("div", { class: "msg-bubble msg-cursor" });
    const card = el("article", { class: "guru-panel-card", dataset: { guruId } },
      el("header", { class: "guru-panel-head" },
        el("span", { class: "guru-avatar guru-avatar-sm", "aria-hidden": "true" }, initials(guruName(guruId))),
        el("span", {}, guruName(guruId))),
      body);
    return { card, body };
  }

  function renderSynthesis(synthesis) {
    const rows = synthesis.stances.map((s) =>
      `| ${guruName(s.guru_id)} | ${s.stance} | ${s.key_reason} | ${s.risk} |`);
    const markdown = ["| กูรู | จุดยืน | เหตุผลหลัก | ความเสี่ยง |", "|---|---|---|---|", ...rows,
      "", `**เห็นตรงกัน:** ${synthesis.consensus}`, `**เห็นต่าง:** ${synthesis.differences}`].join("\n");
    const bubble = appendMessage("assistant", markdown, "");
    bubble.closest(".msg").classList.add("guru-synthesis");
    bubble.closest(".msg").querySelector(".msg-role").textContent = "สรุปเปรียบเทียบ";
  }

  function streamPanel(text) {
    const ids = [...guru.panelIds];
    const grid = app().el("div", { class: "guru-panel-grid" });
    const bodies = {};
    for (const id of ids) {
      const { card, body } = panelCard(id);
      bodies[id] = body;
      grid.append(card);
    }
    $("guruThread").append(grid);
    scrollThread();
    const params = new URLSearchParams({
      query: text, user_id: app().getUserId(),
      guru_ids: ids.join(","), conversation_id: guru.conversationId,
    });
    return new Promise((resolve) => {
      const source = new EventSource(`${app().apiBase}/guru/panel/stream?${params}`);
      const settle = (ev, failed) => {
        const data = JSON.parse(ev.data);
        const body = bodies[data.guru_id];
        if (!body) return;
        body.classList.remove("msg-cursor");
        body.innerHTML = failed ? "ตอบไม่สำเร็จ" : app().renderMarkdown(data.content);
        if (!failed) { const links = sourceLinks(data.content); if (links) body.after(links); }
      };
      source.addEventListener("guru_answer", (ev) => settle(ev, false));
      source.addEventListener("guru_error", (ev) => settle(ev, true));
      source.addEventListener("synthesis", (ev) => renderSynthesis(JSON.parse(ev.data)));
      source.addEventListener("synthesis_error", () => app().toast("สรุปเปรียบเทียบไม่สำเร็จ", true));
      const finish = () => {
        source.close();
        Object.values(bodies).forEach((b) => {
          if (b.classList.contains("msg-cursor")) { b.classList.remove("msg-cursor"); b.textContent = "ไม่มีคำตอบ"; }
        });
        resolve();
      };
      source.addEventListener("complete", finish);
      source.onerror = finish;
    });
  }

  // ─── Send ───
  function setBusy(busy) {
    guru.streaming = busy;
    $("guruInput").disabled = busy;
    $("guruSend").disabled = busy || !$("guruInput").value.trim();
  }

  async function send(text) {
    if (!text.trim() || guru.streaming) return;
    if (guru.mode === "single" && !guru.selectedId) { app().toast("เลือกกูรูก่อนนะ", true); return; }
    setBusy(true);
    appendMessage("user", text);
    try {
      await ensureConversation(text, guru.mode === "panel" ? PANEL_ID : guru.selectedId);
      await (guru.mode === "panel" ? streamPanel(text) : streamSingle(text));
    } catch (e) {
      app().toast("ส่งคำถามไม่สำเร็จ", true);
    } finally {
      setBusy(false);
      $("guruInput").focus();
    }
  }

  function wireComposer() {
    const input = $("guruInput");
    const sendBtn = $("guruSend");
    input.addEventListener("input", () => {
      sendBtn.disabled = !input.value.trim() || guru.streaming;
      input.style.height = "auto";
      input.style.height = Math.min(140, input.scrollHeight) + "px";
    });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) $("guruComposer").requestSubmit();
      }
    });
    $("guruComposer").addEventListener("submit", (e) => {
      e.preventDefault();
      const text = input.value.trim();
      input.value = "";
      input.style.height = "auto";
      send(text);
    });
  }

  // ─── Open (called by app.js switchView) ───
  function wireOnce() {
    if (guru.wired) return;
    guru.wired = true;
    wireComposer();
    document.querySelectorAll(".guru-mode-btn").forEach((b) =>
      b.addEventListener("click", () => setMode(b.dataset.mode)));
    $("guruHistory").addEventListener("change", (e) =>
      openConversation(e.target.value, e.target.selectedOptions[0].dataset.guruId));
  }

  async function open() {
    if (guru.loaded) return;
    guru.loaded = true;
    wireOnce();
    try {
      guru.gurus = await app().api("GET", "/gurus");
    } catch (e) {
      guru.loaded = false;
      app().toast("โหลดรายชื่อกูรูไม่สำเร็จ", true);
      return;
    }
    guru.byId = Object.fromEntries(guru.gurus.map((g) => [g.guru_id, g]));
    guru.selectedId = guru.gurus.length ? guru.gurus[0].guru_id : "";
    guru.panelIds = new Set(guru.gurus.map((g) => g.guru_id));
    renderCards();
    renderChips();
    loadHistory();
  }

  window.GuruView = { open };
})();
