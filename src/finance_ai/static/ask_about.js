/* Personal Finance AI — "ask about this" bar
   Select text inside an AI answer (chat or Guru) and a small bar pops in under
   it: อธิบายง่ายๆ / ใช้กับฉันยังไง / ยกตัวอย่างตัวเลข, or a pencil to type an own
   question. The picked text stays highlighted (CSS Custom Highlight API) and
   the bar stays open until the user sends, presses × or Esc, or clicks
   elsewhere. Sending posts a new message that quotes the picked text.
   Pure helpers are exported for node tests (module.exports).
*/
(function (root) {
  "use strict";

  const MIN_CHARS = 2;
  const MAX_CHARS = 500;
  const SETTLE_MS = 150;
  const GAP_PX = 8;
  const HIGHLIGHT_NAME = "ask-pin";
  const PRESETS = [
    { label: "อธิบายง่ายๆ", prompt: "อธิบายตรงนี้แบบง่ายๆ" },
    { label: "ใช้กับฉันยังไง", prompt: "ข้อนี้ใช้กับสถานการณ์การเงินของฉันยังไง", featured: true },
    { label: "ยกตัวอย่างตัวเลข", prompt: "ยกตัวอย่างเป็นตัวเลขให้หน่อย" },
  ];
  const ANSWER_SELECTOR = "#chatThread .msg-bubble, #guruThread .msg-bubble";

  // ─── Pure helpers ───
  function tidyQuote(text) {
    return String(text || "").split("\n")
      .map((line) => line.replace(/[ \t]+/g, " ").trim())
      .filter(Boolean)
      .join("\n");
  }

  function isAskable(text) {
    const length = tidyQuote(text).length;
    return length >= MIN_CHARS && length <= MAX_CHARS;
  }

  // "<question>:\n> line 1\n> line 2" — markdown quote so the chat shows it as a quote.
  function composeQuestion(question, quote) {
    const quoted = tidyQuote(quote).split("\n").map((line) => `> ${line}`).join("\n");
    return `${question.trim()}:\n${quoted}`;
  }

  const helpers = { composeQuestion, isAskable, tidyQuote, PRESETS, MAX_CHARS };
  if (typeof module !== "undefined" && module.exports) { module.exports = helpers; return; }
  root.AskAbout = helpers;

  // ─── Browser wiring ───
  const state = { range: null, quote: "", bar: null, settleTimer: 0, pressedOutside: false, frame: 0 };
  const canHighlight = () => typeof CSS !== "undefined" && "highlights" in CSS && typeof Highlight === "function";

  function h(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (key === "class") node.className = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value);
    }
    node.append(...children);
    return node;
  }

  function icon(id) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "i");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    use.setAttribute("href", `#${id}`);
    svg.append(use);
    return svg;
  }

  function buildBar() {
    const presets = h("div", { class: "ask-presets" },
      ...PRESETS.map((preset) => h("button", {
        type: "button", class: "ask-act" + (preset.featured ? " is-featured" : ""),
        onclick: () => send(preset.prompt),
      }, preset.label)),
      h("span", { class: "ask-sep", "aria-hidden": "true" }),
      h("button", { type: "button", class: "ask-icon", "aria-label": "พิมพ์คำถามเอง", onclick: openTyping }, icon("i-pen")));
    const input = h("input", { class: "ask-input", type: "text", maxlength: "300",
      placeholder: "ถามเกี่ยวกับตรงนี้…", "aria-label": "พิมพ์คำถามเกี่ยวกับข้อความที่เลือก" });
    const form = h("form", { class: "ask-form", onsubmit: onTypedSubmit }, input,
      h("button", { type: "submit", class: "ask-send", "aria-label": "ส่งคำถาม" }, icon("i-up")));
    const bar = h("div", { class: "ask-bar", role: "toolbar", "aria-label": "ถามเกี่ยวกับข้อความที่เลือก" },
      presets, form, h("span", { class: "ask-sep", "aria-hidden": "true" }),
      h("button", { type: "button", class: "ask-icon", "aria-label": "ปิด", onclick: () => dismiss(true) }, icon("i-x")));
    // Keep the page selection when tapping buttons; the input may take focus.
    bar.addEventListener("mousedown", (event) => { if (event.target.tagName !== "INPUT") event.preventDefault(); });
    document.body.append(bar);
    return bar;
  }

  function selectedAnswerRange() {
    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || !selection.rangeCount) return null;
    const range = selection.getRangeAt(0);
    const start = range.startContainer.parentElement;
    const end = range.endContainer.parentElement;
    const answer = start && start.closest(ANSWER_SELECTOR);
    const inUserMessage = start && start.closest(".msg.user");
    if (!answer || inUserMessage || !end || end.closest(ANSWER_SELECTOR) !== answer) return null;
    return isAskable(selection.toString()) ? range : null;
  }

  function pin(range) {
    state.range = range.cloneRange();
    state.quote = tidyQuote(range.toString());
    if (canHighlight()) CSS.highlights.set(HIGHLIGHT_NAME, new Highlight(state.range));
    state.bar.classList.add("is-shown");
    place();
  }

  function dismiss(clearSelection) {
    state.range = null;
    if (canHighlight()) CSS.highlights.delete(HIGHLIGHT_NAME);
    state.bar.classList.remove("is-shown", "is-typing", "is-away");
    state.bar.querySelector(".ask-input").value = "";
    if (clearSelection) window.getSelection().removeAllRanges();
  }

  // Centre under the picked text's last line; flip above it near the bottom edge.
  function place() {
    if (!state.range) return;
    if (!state.range.startContainer.isConnected) { dismiss(false); return; }  // chat switched
    const rects = state.range.getClientRects();
    const last = rects[rects.length - 1];
    if (!last) return;
    const bounds = state.range.getBoundingClientRect();
    const half = state.bar.offsetWidth / 2 + GAP_PX;
    const x = Math.min(Math.max(bounds.left + bounds.width / 2, half), window.innerWidth - half);
    const below = last.bottom + GAP_PX;
    const y = below + state.bar.offsetHeight > window.innerHeight ? rects[0].top - GAP_PX - state.bar.offsetHeight : below;
    state.bar.style.setProperty("--ask-x", `${Math.round(x)}px`);
    state.bar.style.setProperty("--ask-y", `${Math.round(y)}px`);
    state.bar.classList.toggle("is-away", !isInView(bounds));
  }

  // Hide (not dismiss) while the picked text is scrolled out of its thread.
  function isInView(bounds) {
    const thread = state.range.startContainer.parentElement.closest(".chat-thread");
    const view = (thread || document.documentElement).getBoundingClientRect();
    return bounds.bottom > Math.max(view.top, 0) && bounds.top < Math.min(view.bottom, window.innerHeight);
  }

  function schedulePlace() {
    cancelAnimationFrame(state.frame);
    state.frame = requestAnimationFrame(place);
  }

  function openTyping() {
    state.bar.classList.add("is-typing");
    schedulePlace();
    setTimeout(() => state.bar.querySelector(".ask-input").focus(), 60);
  }

  function onTypedSubmit(event) {
    event.preventDefault();
    const question = state.bar.querySelector(".ask-input").value.trim();
    if (question) send(question);
  }

  function send(question) {
    const inGuru = Boolean(state.range && state.range.startContainer.parentElement.closest("#guruThread"));
    const busy = inGuru ? root.GuruView && root.GuruView.isBusy() : root.FinanceApp.isStreaming();
    if (busy) { root.FinanceApp.toast("รอคำตอบนี้เสร็จก่อนนะ", true); return; }
    const message = composeQuestion(question, state.quote);
    dismiss(true);
    if (inGuru) root.GuruView.send(message);
    else root.FinanceApp.sendChat(message);
  }

  function onSelectionChange() {
    clearTimeout(state.settleTimer);
    state.settleTimer = setTimeout(() => {
      const range = selectedAnswerRange();
      if (range) pin(range);
    }, SETTLE_MS);
  }

  function onDocumentClick(event) {
    if (!state.range || !state.pressedOutside || state.bar.contains(event.target)) return;
    if (!selectedAnswerRange()) dismiss(false);
  }

  function setup() {
    state.bar = buildBar();
    document.addEventListener("selectionchange", onSelectionChange);
    document.addEventListener("pointerdown", (event) => { state.pressedOutside = !state.bar.contains(event.target); });
    document.addEventListener("click", onDocumentClick);
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && state.range) { event.stopPropagation(); dismiss(true); }
    }, true);
    document.addEventListener("scroll", schedulePlace, { capture: true, passive: true });
    window.addEventListener("resize", schedulePlace);
  }

  document.addEventListener("DOMContentLoaded", setup);
})(typeof window !== "undefined" ? window : this);
