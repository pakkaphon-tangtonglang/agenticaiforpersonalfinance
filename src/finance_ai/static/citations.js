/* Personal Finance AI — citation pills, source chips and popover cards
   Turns "[1]" markers in a rendered answer into small numbered pills, adds a
   compact source row under the answer, and shows one shared popover card
   (site, title, where it was compiled from, snippet, link) on hover, tap or
   keyboard. Sources come from the backend; internal filenames never show.
*/
(function () {
  "use strict";

  const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  const LEGACY_LABELS = [
    /\s*\[แหล่งที่มา:[^\]]*\]/g,
    /\s*\((?:อ้างอิง|ที่มา|แหล่งที่มา)\s*:[^)]*\)/g,
  ];
  const URL_PATTERN = /https?:\/\/[^\s|\])>"'<]+/g;
  const SKIP_TAGS = new Set(["CODE", "PRE", "A", "BUTTON"]);
  const VISIBLE_CHIPS = 3;
  const OPEN_DELAY_MS = 120;
  const CLOSE_DELAY_MS = 180;
  const SHEET_MAX_WIDTH = 480;

  const sourcesByHost = new WeakMap();
  const popover = { el: null, trigger: null, openTimer: 0, closeTimer: 0 };

  // ─── Small DOM helper (app.js may not be ready when this file loads) ───
  function h(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (key === "class") node.className = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (value !== null && value !== undefined && value !== false) node.setAttribute(key, value);
    }
    for (const child of children.flat()) {
      if (child !== null && child !== undefined && child !== false) node.append(child);
    }
    return node;
  }

  // ─── Text clean-up ───
  function stripLegacyLabels(text) {
    return LEGACY_LABELS.reduce((out, pattern) => out.replace(pattern, ""), String(text || ""));
  }

  // For text shown without source cards (e.g. the panel comparison table).
  function stripMarkers(text) {
    return String(text || "").replace(/[ \t]*\[\d+(?:\s*,\s*\d+)*\]/g, "");
  }

  function hostOf(url) {
    try { return new URL(url).hostname.replace(/^www\./, ""); } catch (_) { return ""; }
  }

  // Old answers saved before structured sources: show their URLs as chips.
  function legacySources(text) {
    const urls = [...new Set(String(text || "").match(URL_PATTERN) || [])];
    return urls.map((url, i) => ({
      index: i + 1, title: hostOf(url), url, domain: hostOf(url), snippet: "", kind: "",
    })).filter((source) => source.domain);
  }

  // ─── Pieces of a source ───
  function sourceIcon(source) {
    const letter = (source.publisher || source.domain || source.title || "?").trim().charAt(0);
    const fallback = h("span", { class: "cite-icon cite-icon-letter", "aria-hidden": "true" }, letter);
    if (!source.domain) return fallback;
    const img = h("img", {
      class: "cite-icon", alt: "", width: "16", height: "16", loading: "lazy",
      src: `https://www.google.com/s2/favicons?domain=${encodeURIComponent(source.domain)}&sz=32`,
    });
    img.addEventListener("error", () => img.replaceWith(fallback), { once: true });
    return img;
  }

  function shortLabel(source) {
    return source.publisher || source.domain || source.title;
  }

  function originLine(source) {
    if (source.kind === "original") return source.publisher ? `ต้นฉบับ · ${source.publisher}` : "ต้นฉบับ";
    const from = [source.publisher, source.source_title].filter(Boolean).join(" – ");
    return from ? `อ้างอิงข้อมูลจาก ${from}` : "";
  }

  // ─── Inline [n] pills ───
  function pill(source) {
    return h("button", {
      type: "button", class: "cite-pill", "data-cite": String(source.index),
      "aria-haspopup": "dialog", "aria-expanded": "false",
      "aria-label": `แหล่งที่มา ${source.index}: ${source.title}`,
    }, String(source.index));
  }

  function markerFragment(numbers, byIndex) {
    const fragment = document.createDocumentFragment();
    for (const number of numbers.split(",").map((n) => Number(n.trim()))) {
      if (byIndex.has(number)) fragment.append(pill(byIndex.get(number)));
    }
    return fragment;
  }

  function textNodesWithMarkers(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        for (let p = node.parentElement; p && p !== root; p = p.parentElement) {
          if (SKIP_TAGS.has(p.tagName)) return NodeFilter.FILTER_REJECT;
        }
        MARKER.lastIndex = 0;
        return MARKER.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
      },
    });
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    return nodes;
  }

  function replaceMarkers(root, byIndex) {
    for (const node of textNodesWithMarkers(root)) {
      const fragment = document.createDocumentFragment();
      let last = 0;
      for (const match of node.nodeValue.matchAll(MARKER)) {
        fragment.append(node.nodeValue.slice(last, match.index).replace(/[ \t]+$/, ""));
        fragment.append(markerFragment(match[1], byIndex));
        last = match.index + match[0].length;
      }
      fragment.append(node.nodeValue.slice(last));
      node.replaceWith(fragment);
    }
  }

  // ─── Source chip row ───
  function chip(source) {
    return h("button", {
      type: "button", class: "cite-chip", "data-cite": String(source.index),
      "aria-haspopup": "dialog", "aria-expanded": "false",
      "aria-label": `แหล่งที่มา ${source.index}: ${source.title}`,
    }, sourceIcon(source), h("span", { class: "cite-chip-label" }, shortLabel(source)));
  }

  function chipRow(sources) {
    const row = h("div", { class: "cite-row" }, h("span", { class: "cite-row-label" }, "แหล่งที่มา"));
    sources.forEach((source, i) => {
      const node = chip(source);
      if (i >= VISIBLE_CHIPS) node.hidden = true;
      row.append(node);
    });
    if (sources.length > VISIBLE_CHIPS) {
      const more = h("button", { type: "button", class: "cite-more", "aria-label": "แสดงแหล่งที่มาทั้งหมด" },
        `+${sources.length - VISIBLE_CHIPS}`);
      more.addEventListener("click", () => {
        row.querySelectorAll(".cite-chip[hidden]").forEach((node) => { node.hidden = false; });
        more.remove();
      });
      row.append(more);
    }
    return row;
  }

  // ─── Public: decorate a finished answer ───
  // host: the element that should own the chip row (bubble's .msg or a card).
  function decorate(bubble, sources, host) {
    const owner = host || bubble.closest(".msg") || bubble.parentElement;
    if (!owner) return;
    owner.querySelectorAll(":scope > .cite-row").forEach((row) => row.remove());
    const list = (sources || []).filter((source) => source && source.index && source.title);
    const byIndex = new Map(list.map((source) => [Number(source.index), source]));
    replaceMarkers(bubble, byIndex);
    if (!list.length) return;
    sourcesByHost.set(owner, byIndex);
    owner.setAttribute("data-cite-host", "");
    bubble.after(chipRow(list));
  }

  // ─── Popover ───
  function ensurePopover() {
    if (popover.el) return popover.el;
    popover.el = h("div", { class: "cite-popover", id: "citePopover", role: "dialog", "aria-label": "แหล่งที่มา" });
    popover.el.hidden = true;
    popover.el.addEventListener("mouseenter", () => clearTimeout(popover.closeTimer));
    popover.el.addEventListener("mouseleave", scheduleClose);
    document.body.append(popover.el);
    return popover.el;
  }

  function cardContent(source) {
    const site = source.domain || "คลังความรู้ของระบบ";
    const title = source.url
      ? h("a", { class: "cite-card-title", href: source.url, target: "_blank", rel: "noopener noreferrer" }, source.title)
      : h("span", { class: "cite-card-title" }, source.title);
    return [
      h("div", { class: "cite-card-site" }, sourceIcon(source), h("span", {}, site),
        h("span", { class: "cite-card-num" }, `[${source.index}]`)),
      title,
      originLine(source) ? h("div", { class: "cite-card-origin" }, originLine(source)) : null,
      source.snippet ? h("p", { class: "cite-card-snippet" }, source.snippet) : null,
      source.url ? h("a", { class: "cite-card-link", href: source.url, target: "_blank", rel: "noopener noreferrer" },
        "เปิดแหล่งที่มา ↗") : null,
    ];
  }

  function placePopover(el, trigger) {
    const sheet = window.innerWidth <= SHEET_MAX_WIDTH;
    el.classList.toggle("is-sheet", sheet);
    if (sheet) { el.style.left = ""; el.style.top = ""; return; }
    const rect = trigger.getBoundingClientRect();
    const width = el.offsetWidth;
    const height = el.offsetHeight;
    const below = rect.bottom + 8;
    const top = below + height > window.innerHeight - 8 ? Math.max(8, rect.top - height - 8) : below;
    const left = Math.min(Math.max(8, rect.left - 12), window.innerWidth - width - 8);
    el.style.left = `${left}px`;
    el.style.top = `${top}px`;
  }

  function sourceFor(trigger) {
    const host = trigger.closest("[data-cite-host]");
    const byIndex = host ? sourcesByHost.get(host) : null;
    return byIndex ? byIndex.get(Number(trigger.dataset.cite)) : null;
  }

  function openFor(trigger) {
    const source = sourceFor(trigger);
    if (!source) return;
    clearTimeout(popover.closeTimer);
    if (popover.trigger && popover.trigger !== trigger) popover.trigger.setAttribute("aria-expanded", "false");
    const el = ensurePopover();
    el.replaceChildren(...cardContent(source).filter(Boolean));
    el.hidden = false;
    placePopover(el, trigger);
    popover.trigger = trigger;
    trigger.setAttribute("aria-expanded", "true");
    trigger.setAttribute("aria-controls", el.id);
  }

  function close(returnFocus) {
    clearTimeout(popover.openTimer);
    clearTimeout(popover.closeTimer);
    if (!popover.el || popover.el.hidden) return;
    popover.el.hidden = true;
    if (popover.trigger) {
      popover.trigger.setAttribute("aria-expanded", "false");
      if (returnFocus) popover.trigger.focus();
    }
    popover.trigger = null;
  }

  function scheduleClose() {
    clearTimeout(popover.closeTimer);
    popover.closeTimer = setTimeout(() => close(false), CLOSE_DELAY_MS);
  }

  // ─── Event wiring (delegated, once) ───
  const triggerOf = (target) => (target instanceof Element ? target.closest(".cite-pill, .cite-chip") : null);

  function onClick(event) {
    const trigger = triggerOf(event.target);
    if (trigger) {
      event.preventDefault();
      if (popover.trigger === trigger && !popover.el.hidden) close(false);
      else openFor(trigger);
      return;
    }
    if (popover.el && !popover.el.contains(event.target)) close(false);
  }

  function onPointerOver(event) {
    if (event.pointerType && event.pointerType !== "mouse") return;
    const trigger = triggerOf(event.target);
    if (!trigger) return;
    clearTimeout(popover.closeTimer);
    clearTimeout(popover.openTimer);
    popover.openTimer = setTimeout(() => openFor(trigger), OPEN_DELAY_MS);
  }

  function onPointerOut(event) {
    if (event.pointerType && event.pointerType !== "mouse") return;
    if (!triggerOf(event.target)) return;
    clearTimeout(popover.openTimer);
    scheduleClose();
  }

  function onKeyDown(event) {
    if (event.key === "Escape" && popover.el && !popover.el.hidden) close(true);
  }

  function onFocusOut(event) {
    if (!popover.el || popover.el.hidden) return;
    const next = event.relatedTarget;
    if (next && (popover.el.contains(next) || next === popover.trigger)) return;
    if (next) close(false);
  }

  function onViewportChange(event) {
    if (popover.el && event && popover.el.contains(event.target)) return;
    close(false);
  }

  document.addEventListener("click", onClick);
  document.addEventListener("pointerover", onPointerOver);
  document.addEventListener("pointerout", onPointerOut);
  document.addEventListener("keydown", onKeyDown);
  document.addEventListener("focusout", onFocusOut);
  document.addEventListener("scroll", onViewportChange, true);
  window.addEventListener("resize", () => onViewportChange(null));

  window.Citations = { decorate, stripLegacyLabels, stripMarkers, legacySources };
})();
