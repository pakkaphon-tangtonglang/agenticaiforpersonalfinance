/* ============================================================
   Clarify card — asks one question at a time when an asset search
   is unsure (nothing found, or a mix of stocks/funds/fund houses).
   Modelled on the "Approval card" (human-in-the-loop) reference:
   the question stack slides vertically and the card's height animates
   to fit, the step counter rolls like an odometer, single-choice
   answers auto-advance, and the footer has a quiet Skip and a dark
   Continue (⏎). A typed "something else" answer searches again.

   ClarifyCard.mount(host, clarify, { onDone(filters), onSearch(text), onClose() })
   clarify = { reason, questions: [{ id, question, options: [{ label, key, value }] }],
               custom_placeholder }
   ============================================================ */
(function () {
  "use strict";

  const AUTO_ADVANCE_MS = 480;
  const ROLL_MS = 400;
  const el = (...args) => window.FinanceApp.el(...args);
  const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const ICON = {
    close: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6L6 18M6 6l12 12"/></svg>',
    up: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 15l-6-6-6 6"/></svg>',
    down: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>',
  };

  function iconButton(cls, label, icon) {
    const button = el("button", { type: "button", class: cls, "aria-label": label });
    button.innerHTML = icon;
    return button;
  }

  // ─── Odometer: each changed digit rolls up (or down when going back) ───
  function rollingCounter() {
    const node = el("span", { class: "clarify-count num", "aria-live": "off" });
    let shown = "";
    function set(text) {
      const previous = shown;
      shown = text;
      if (!previous || reduceMotion()) { node.textContent = text; return; }
      const down = parseInt(text, 10) < parseInt(previous, 10);
      node.replaceChildren(...[...text].map((char, i) => rollDigit(previous[i] ?? "", char, down)));
    }
    return { node, set };
  }

  function rollDigit(from, to, down) {
    if (from === to) return document.createTextNode(to);
    const wheel = el("span", { class: "clarify-roll" + (down ? " is-down" : "") },
      el("span", {}, down ? to : from), el("span", {}, down ? from : to));
    const slot = el("span", { class: "clarify-digit" }, wheel);
    requestAnimationFrame(() => requestAnimationFrame(() => wheel.classList.add("is-rolled")));
    setTimeout(() => slot.replaceWith(document.createTextNode(to)), ROLL_MS);
    return slot;
  }

  // ─── One question: radio options + "something else" input ───
  function questionNode(question, placeholder, handlers) {
    const options = question.options.map((option, index) => {
      const button = el("button", { type: "button", class: "clarify-option", "aria-pressed": "false" },
        el("span", { class: "clarify-radio", "aria-hidden": "true" }), el("span", { class: "clarify-label" }, option.label));
      button.addEventListener("click", () => handlers.pick(index));
      return button;
    });
    const input = el("input", { type: "text", class: "clarify-input", placeholder, "aria-label": "คำตอบอื่น" });
    input.addEventListener("input", () => handlers.typed(input.value));
    input.addEventListener("keydown", (e) => { if (e.key === "Enter" && input.value.trim()) { e.preventDefault(); handlers.enter(); } });
    const node = el("div", { class: "clarify-question" },
      el("div", { class: "clarify-title" }, question.question),
      el("div", { class: "clarify-options" }, ...options, el("label", { class: "clarify-option is-custom" }, input)));
    return { node, options, input };
  }

  function mount(host, clarify, handlers) {
    const questions = clarify.questions || [];
    if (!questions.length) return null;
    const answers = questions.map(() => null);
    let current = 0;
    let custom = "";
    let timer = null;
    let measured = false;

    const counter = rollingCounter();
    const track = el("div", { class: "clarify-track" });
    const viewport = el("div", { class: "clarify-viewport", "aria-live": "polite" }, track);
    const prev = iconButton("clarify-step", "คำถามก่อนหน้า", ICON.up);
    const next = iconButton("clarify-step", "คำถามถัดไป", ICON.down);
    const skip = el("button", { type: "button", class: "clarify-skip" }, "ข้าม");
    const go = el("button", { type: "button", class: "clarify-go" });
    const dismiss = iconButton("clarify-dismiss", "ปิด", ICON.close);
    const card = el("div", { class: "clarify-card", role: "group", "aria-label": "ช่วยให้ผลค้นหาตรงขึ้น" },
      dismiss,
      el("div", { class: "clarify-pad" }, viewport),
      el("div", { class: "clarify-footer" },
        el("div", { class: "clarify-steps" }, prev, counter.node, next),
        el("div", { class: "clarify-actions" }, skip, go)));

    const views = questions.map((question, qIndex) => questionNode(question, clarify.custom_placeholder || "อย่างอื่น…", {
      pick: (index) => pick(qIndex, index),
      typed: (text) => { custom = text; answers[qIndex] = null; render(); },
      enter: () => finish(),
    }));
    views.forEach((view) => track.append(view.node));

    function pick(qIndex, index) {
      if (qIndex !== current) return;
      answers[qIndex] = index;
      custom = "";
      views[qIndex].input.value = "";
      render();
      clearTimeout(timer);
      timer = setTimeout(advance, AUTO_ADVANCE_MS);
    }

    function goTo(index) {
      clearTimeout(timer);
      current = Math.min(Math.max(index, 0), questions.length - 1);
      render();
    }

    function advance() {
      if (current === questions.length - 1) finish();
      else goTo(current + 1);
    }

    function filters() {
      const chosen = {};
      answers.forEach((answer, qIndex) => {
        if (answer === null) return;
        const option = questions[qIndex].options[answer];
        chosen[option.key] = { value: option.value, label: option.label };
      });
      return chosen;
    }

    function finish() {
      clearTimeout(timer);
      card.remove();
      if (custom.trim()) handlers.onSearch(custom.trim(), filters());
      else handlers.onDone(filters());
    }

    function render() {
      views.forEach((view, qIndex) => {
        const active = qIndex === current;
        view.node.classList.toggle("is-active", active);
        view.node.toggleAttribute("inert", !active);
        view.node.setAttribute("aria-hidden", String(!active));
        view.options.forEach((button, index) => button.setAttribute("aria-pressed", String(answers[qIndex] === index)));
      });
      const last = current === questions.length - 1;
      const answered = answers[current] !== null || Boolean(custom.trim());
      prev.disabled = current === 0;
      next.disabled = last;
      go.disabled = !answered;
      go.innerHTML = "";
      go.append(document.createTextNode(custom.trim() ? "ค้นหา" : last ? "ดูผลลัพธ์" : "ต่อไป"), el("span", { class: "clarify-enter", "aria-hidden": "true" }, "⏎"));
      skip.textContent = last ? "ข้าม" : "ข้ามข้อนี้";
      counter.set(`${current + 1} / ${questions.length}`);
      sync();
    }

    // The viewport height follows the active question; the track slides to it.
    function sync() {
      const active = views[current].node;
      const animate = measured && !reduceMotion();
      card.classList.toggle("is-animated", animate);
      viewport.style.height = `${active.offsetHeight}px`;
      track.style.transform = `translate3d(0, ${-active.offsetTop}px, 0)`;
      measured = true;
    }

    prev.addEventListener("click", () => goTo(current - 1));
    next.addEventListener("click", () => goTo(current + 1));
    skip.addEventListener("click", () => (current === questions.length - 1 ? finish() : goTo(current + 1)));
    go.addEventListener("click", advance);
    dismiss.addEventListener("click", () => { clearTimeout(timer); card.remove(); handlers.onClose && handlers.onClose(); });
    card.addEventListener("keydown", (e) => {
      if (e.key === "Escape") dismiss.click();
      if (e.key === "Enter" && e.target === card && !go.disabled) advance();
    });

    host.replaceChildren(card);
    render();
    requestAnimationFrame(sync);
    window.addEventListener("resize", sync, { passive: true });
    return { close: () => card.remove() };
  }

  window.ClarifyCard = { mount };
})();
