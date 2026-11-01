/* ============================================================
   Tool chips — the agent's tool calls as compact rows under a
   collapsible header. Each row: icon (chevron on hover), Thai label,
   and a chip with the main argument; click to see the arguments and
   the first lines of what the tool returned.
   Rows arrive live from the `tool` SSE event (see stream_utils.py).
   ============================================================ */
(function () {
  "use strict";

  const ICONS = {
    think: '<svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.4 7.2L22 12l-7.6 2.8L12 22l-2.4-7.2L2 12l7.6-2.8z"/></svg>',
    write: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 3a2.8 2.8 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5z"/></svg>',
    run: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 17l6-5-6-5M12 19h8"/></svg>',
    read: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>',
  };
  const CHEVRON = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>';

  // Tool name → [Thai label, icon]. read = looks data up, run = calculates, write = changes data.
  const TOOLS = {
    add_expense: ["บันทึกรายจ่าย", "write"],
    add_income: ["บันทึกรายรับ", "write"],
    create_financial_goal: ["สร้างเป้าหมาย", "write"],
    update_goal_progress: ["อัปเดตเป้าหมาย", "write"],
    update_savings_goal_cross: ["อัปเดตเงินออม", "write"],
    manage_watchlist: ["จัดการรายการจับตา", "write"],
    add_portfolio_holding: ["เพิ่มรายการถือครอง", "write"],
    calculate_thai_tax: ["คำนวณภาษี", "run"],
    calculate_saving_plan: ["คำนวณแผนออม", "run"],
    plan_pay_yourself_first: ["วางแผนออมก่อนใช้", "run"],
    generate_financial_recommendations: ["วิเคราะห์คำแนะนำ", "run"],
    generate_financial_report_tool: ["สร้างรายงาน", "run"],
    get_financial_health_score: ["คำนวณคะแนนสุขภาพการเงิน", "run"],
    detect_psychological_cues: ["อ่านพฤติกรรมการเงิน", "run"],
    get_guru_scorecard: ["ให้คะแนนตามแนวกูรู", "run"],
    search_finance_knowledge: ["ค้นคลังความรู้", "read"],
    search_guru_knowledge: ["ค้นคลังความรู้กูรู", "read"],
    search_finance_news: ["ค้นข่าว", "read"],
    get_expense_summary_cross: ["ดูสรุปรายจ่าย", "read"],
    get_income_summary_cross: ["ดูสรุปรายรับ", "read"],
    get_goals_summary_cross: ["ดูเป้าหมาย", "read"],
    view_financial_goals: ["ดูเป้าหมาย", "read"],
    get_financial_summary: ["ดูภาพรวมการเงิน", "read"],
    get_portfolio_holdings: ["ดูพอร์ต", "read"],
    get_stock_price: ["ดูราคาหุ้น", "read"],
    resolve_asset_symbol: ["หาชื่อย่อหลักทรัพย์", "read"],
    query_expenses_by_category: ["ค้นรายจ่ายตามหมวด", "read"],
    summarize_monthly_expenses: ["สรุปรายจ่ายรายเดือน", "read"],
  };
  // The argument most worth showing in the chip, in order of preference.
  const CHIP_KEYS = ["query", "symbol", "symbols", "category", "name", "description", "note", "action"];
  const ASCII_ONLY = /^[\x20-\x7E]+$/;

  const el = (...args) => window.FinanceApp.el(...args);

  function toolLook(name) {
    const [label, icon] = TOOLS[name] || [name, "run"];
    return { label, icon };
  }

  function chipText(trace) {
    const key = CHIP_KEYS.find((k) => trace.args[k]) || Object.keys(trace.args)[0];
    if (key) return trace.args[key];
    return trace.preview[0] || trace.name;
  }

  function detailLines(trace) {
    const args = Object.entries(trace.args).map(([k, v]) => ({ text: `${k}: ${v}`, mono: true }));
    const shown = new Set(args.map((line) => line.text));
    const tone = trace.failed ? "fail" : "";
    // Tools often echo their inputs; skip result lines already shown as arguments.
    const results = trace.preview.filter((text) => !shown.has(text)).map((text) => ({ text, tone }));
    return [...args, ...results];
  }

  function iconSlot(icon) {
    const slot = el("span", { class: "tc-icon", "aria-hidden": "true" });
    slot.innerHTML = ICONS[icon] + CHEVRON;
    return slot;
  }

  function detailBlock(lines) {
    return el("div", { class: "tc-detail", inert: "" },
      el("div", { class: "tc-detail-inner" },
        el("div", { class: "tc-detail-lines" }, ...lines.map((line) =>
          el("span", { class: `tc-line${line.mono ? " is-mono" : ""}${line.tone ? " is-" + line.tone : ""}` }, line.text)))));
  }

  function setRowOpen(row, open) {
    row.classList.toggle("is-open", open);
    row.querySelector(".tc-row-btn").setAttribute("aria-expanded", String(open));
    row.querySelector(".tc-detail").toggleAttribute("inert", !open);
  }

  // One row: { icon, label, chip, detail: [{text, mono?, tone?}] }.
  function buildRow(step) {
    const chip = el("span", { class: "tc-chip" + (ASCII_ONLY.test(step.chip) ? " is-mono" : "") }, step.chip);
    const button = el("button", { type: "button", class: "tc-row-btn", "aria-expanded": "false" },
      iconSlot(step.icon), el("span", { class: "tc-label" }, step.label), chip);
    const row = el("div", { class: "tc-row" + (step.failed ? " is-failed" : "") }, button);
    if (!step.detail.length) {
      row.classList.add("is-static");
      button.removeAttribute("aria-expanded");
      return row;
    }
    row.append(detailBlock(step.detail));
    button.addEventListener("click", () => setRowOpen(row, !row.classList.contains("is-open")));
    return row;
  }

  function stepFromTrace(trace) {
    const look = toolLook(trace.name);
    return { ...look, chip: chipText(trace), detail: detailLines(trace), failed: trace.failed };
  }

  function header() {
    const button = el("button", { type: "button", class: "tc-header", "aria-expanded": "true" });
    button.innerHTML = CHEVRON;
    button.append(el("span", { class: "tc-header-text num" }));
    return button;
  }

  function setOpen(root, open) {
    root.classList.toggle("is-collapsed", !open);
    root.querySelector(".tc-header").setAttribute("aria-expanded", String(open));
    root.querySelector(".tc-body").toggleAttribute("inert", !open);
  }

  /* create() → { node, add(trace), collapse(), finish({seconds, agentLabel}), count }
     The node stays hidden until the first tool call arrives. */
  function create() {
    const rows = el("div", { class: "tc-rows" });
    const root = el("div", { class: "tool-chips", hidden: "" },
      header(), el("div", { class: "tc-body" }, el("div", { class: "tc-clip" }, rows)));
    const text = root.querySelector(".tc-header-text");
    let count = 0;
    root.querySelector(".tc-header").addEventListener("click", () =>
      setOpen(root, root.classList.contains("is-collapsed")));
    return {
      node: root,
      get count() { return count; },
      add(trace) {
        count += 1;
        root.hidden = false;
        text.textContent = `ใช้เครื่องมือ ${count} ครั้ง`;
        rows.append(buildRow(stepFromTrace(trace)));
      },
      collapse() { setOpen(root, false); },
      finish({ seconds, agentLabel }) {
        text.textContent = `✦ ใช้เวลา ${Math.max(1, Math.round(seconds))} วินาที · ใช้เครื่องมือ ${count} ครั้ง`;
        if (agentLabel) rows.prepend(buildRow({ icon: "think", label: agentLabel, chip: "ตอบคำถามนี้", detail: [] }));
        root.classList.add("is-settled");
      },
    };
  }

  window.ToolChips = { create, toolLook };
})();
