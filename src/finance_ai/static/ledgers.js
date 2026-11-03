/* Personal Finance AI — dashboard goals, monthly entries and debts
   Lists the user's financial goals, monthly recurring income/expenses and open
   debts, and lets the user add, edit and delete them (the chat can do the
   same). Reuses app.js helpers via window.FinanceApp; categories come from
   history.js.
*/
(function () {
  "use strict";

  const app = () => window.FinanceApp;
  const $ = (id) => document.getElementById(id);
  const editing = { goal: "", debt: "", recurring: "" };
  let openDebts = [];
  let wired = false;

  // ─── Helpers ───
  function formatPercent(value) {
    return `${Number(value || 0).toLocaleString("th-TH", { maximumFractionDigits: 1 })}%`;
  }

  function thaiDate(iso) {
    return new Date(iso + "T00:00:00").toLocaleDateString("th-TH", { day: "numeric", month: "short", year: "numeric" });
  }

  function progressBar(percentage, label) {
    const width = Math.min(100, Math.max(0, Number(percentage || 0)));
    return app().el("div", { class: "goal-bar", role: "progressbar", "aria-label": label,
      "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(Math.round(width)) },
      app().el("div", { class: "goal-fill", style: `width:${width}%` }));
  }

  function rowActions(kind, item, name) {
    const el = app().el;
    return el("span", { class: "history-actions" },
      el("button", { type: "button", class: "history-action", "aria-label": `แก้ไข ${name}`,
        title: "แก้ไข", onclick: () => openForm(kind, item) }, "✎"),
      el("button", { type: "button", class: "history-action is-danger", "aria-label": `ลบ ${name}`,
        title: "ลบ", onclick: () => remove(kind, item, name) }, "🗑"));
  }

  function emptyRow(text) {
    return app().el("li", { class: "notif-empty" }, text);
  }

  // ─── Goals ───
  function goalRow(goal) {
    const el = app().el;
    const due = goal.target_date ? ` · ภายใน ${thaiDate(goal.target_date)}` : "";
    const done = goal.is_completed ? " · ✓ ถึงเป้าแล้ว" : "";
    return el("li", { class: `goal-item${goal.is_completed ? " is-done" : ""}` },
      el("div", { class: "ledger-head" },
        el("span", { class: "goal-name", title: goal.name }, goal.name),
        el("span", { class: "goal-pct" }, formatPercent(goal.percentage)),
        rowActions("goal", goal, goal.name)),
      progressBar(goal.percentage, goal.name),
      el("p", { class: "ledger-sub" },
        `${goal.goal_type_label} · ${app().thb(goal.current_amount)} จาก ${app().thb(goal.target_amount)}${due}${done}`));
  }

  function renderGoals(goals) {
    $("goalCount").textContent = String(goals.length);
    $("goalsList").replaceChildren(...(goals.length
      ? goals.map(goalRow)
      : [emptyRow('ยังไม่มีเป้าหมาย — กด "+ เพิ่มเป้าหมาย" หรือบอกในแชท เช่น "อยากเก็บเงินเที่ยวญี่ปุ่น 60,000"')]));
  }

  // ─── Debts ───
  function debtTerms(debt) {
    const parts = [`${debt.debt_type_label}`, `จ่ายแล้ว ${formatPercent(debt.paid_percentage)} จาก ${app().thb(debt.original_amount)}`];
    if (debt.interest_rate_percent != null) parts.push(`ดอกเบี้ย ${formatPercent(debt.interest_rate_percent)}/ปี`);
    if (debt.monthly_payment != null) parts.push(`ผ่อน ${app().thb(debt.monthly_payment)}/เดือน`);
    return parts.join(" · ");
  }

  function debtRow(debt) {
    const el = app().el;
    return el("li", { class: "goal-item debt-item" },
      el("div", { class: "ledger-head" },
        el("span", { class: "goal-name", title: debt.creditor }, debt.creditor),
        el("span", { class: "ledger-amount" }, `เหลือ ${app().thb(debt.remaining_balance)}`),
        rowActions("debt", debt, debt.creditor)),
      progressBar(debt.paid_percentage, `จ่ายหนี้ ${debt.creditor} แล้ว`),
      el("p", { class: "ledger-sub" }, debtTerms(debt)));
  }

  function renderDebts(listing) {
    const debts = listing.debts;
    $("debtCount").textContent = String(debts.length);
    const monthly = Number(listing.total_monthly_payment) > 0 ? ` · ผ่อนรวม ${app().thb(listing.total_monthly_payment)}/เดือน` : "";
    $("debtTotals").textContent = debts.length ? `หนี้คงเหลือรวม ${app().thb(listing.total_remaining)}${monthly}` : "";
    $("debtsList").replaceChildren(...(debts.length
      ? debts.map(debtRow)
      : [emptyRow('ไม่มีหนี้ค้าง — กด "+ เพิ่มหนี้" หรือบอกในแชท เช่น "ติดหนี้เพื่อน 2,000"')]));
  }

  // ─── Monthly entries ───
  function recurringRow(item) {
    const el = app().el;
    const name = item.description || item.category_label;
    const sign = item.transaction_type === "income" ? "+" : "−";
    const debt = item.debt_creditor ? ` · หักหนี้ ${item.debt_creditor}` : "";
    return el("li", { class: `goal-item recurring-item is-${item.transaction_type}` },
      el("div", { class: "ledger-head" },
        el("span", { class: "goal-name", title: name }, name),
        el("span", { class: "ledger-amount" }, `${sign}${app().thb(item.amount)}`),
        rowActions("recurring", item, name)),
      el("p", { class: "ledger-sub" }, `${item.category_label} · ทุกวันที่ ${item.day_of_month}${debt}`));
  }

  function renderRecurring(items) {
    $("recurringCount").textContent = String(items.length);
    $("recurringList").replaceChildren(...(items.length
      ? items.map(recurringRow)
      : [emptyRow('ยังไม่มี — เช่น เงินเดือน ค่าเช่า ค่าผ่อน กด "+ เพิ่มรายการประจำ" หรือบอกในแชท')]));
  }

  function recurringType() {
    return document.querySelector("input[name='recurringType']:checked").value;
  }

  function fillRecurringChoices(type, category, debtId) {
    const labels = window.HistoryView.categories[type];
    $("recurringCategory").replaceChildren(...Object.entries(labels).map(([key, label]) =>
      new Option(label, key, false, key === category)));
    $("recurringDebt").replaceChildren(new Option("ไม่ผูกกับหนี้", ""),
      ...openDebts.map((d) => new Option(`${d.creditor} (เหลือ ${app().thb(d.remaining_balance)})`, d.debt_id, false, d.debt_id === debtId)));
    $("recurringDebtField").hidden = type !== "expense" || !openDebts.length;
  }

  async function load() {
    wireOnce();
    const params = { user_id: app().getUserId() };
    try {
      const [goals, debts, recurring] = await Promise.all([
        app().api("GET", "/goals", { params }),
        app().api("GET", "/debts", { params }),
        app().api("GET", "/recurring", { params }),
      ]);
      openDebts = debts.debts;
      renderGoals(goals);
      renderDebts(debts);
      renderRecurring(recurring);
    } catch (e) {
      app().toast("โหลดเป้าหมาย/รายการประจำ/หนี้สินไม่สำเร็จ", true);
    }
  }

  // ─── Add / edit forms ───
  const FORMS = {
    goal: {
      title: ["เพิ่มเป้าหมาย", "แก้ไขเป้าหมาย"],
      fill(goal) {
        $("goalName").value = goal ? goal.name : "";
        $("goalType").value = goal ? goal.goal_type : "savings";
        $("goalTarget").value = goal ? goal.target_amount : "";
        $("goalCurrent").value = goal ? goal.current_amount : "0";
        $("goalDate").value = goal && goal.target_date ? goal.target_date : "";
      },
      body: () => ({
        name: $("goalName").value.trim(), goal_type: $("goalType").value,
        target_amount: $("goalTarget").value, current_amount: $("goalCurrent").value || "0",
        target_date: $("goalDate").value || null,
      }),
      problem(body) {
        if (!body.name) return "กรุณาใส่ชื่อเป้าหมาย";
        if (!(Number(body.target_amount) > 0)) return "เป้าหมายต้องมากกว่า 0 บาท";
        if (Number(body.current_amount) < 0) return "ยอดที่เก็บได้ติดลบไม่ได้";
        return "";
      },
      idOf: (goal) => goal.goal_id,
      path: "/goals",
    },
    debt: {
      title: ["เพิ่มหนี้", "แก้ไขหนี้"],
      fill(debt) {
        $("debtCreditor").value = debt ? debt.creditor : "";
        $("debtType").value = debt ? debt.debt_type : "personal";
        $("debtOriginal").value = debt ? debt.original_amount : "";
        $("debtRemaining").value = debt ? debt.remaining_balance : "";
        $("debtRate").value = debt && debt.interest_rate_percent != null ? debt.interest_rate_percent : "";
        $("debtMonthly").value = debt && debt.monthly_payment != null ? debt.monthly_payment : "";
      },
      body() {
        const original = $("debtOriginal").value;
        return {
          creditor: $("debtCreditor").value.trim(), debt_type: $("debtType").value,
          original_amount: original, remaining_balance: $("debtRemaining").value || original,
          interest_rate_percent: $("debtRate").value || null, monthly_payment: $("debtMonthly").value || null,
        };
      },
      problem(body) {
        if (!body.creditor) return "กรุณาใส่ชื่อเจ้าหนี้";
        if (!(Number(body.original_amount) > 0)) return "ยอดที่ยืม/กู้ต้องมากกว่า 0 บาท";
        if (Number(body.remaining_balance) < 0) return "ยอดที่เหลือติดลบไม่ได้";
        if (Number(body.remaining_balance) > Number(body.original_amount)) return "ยอดที่เหลือมากกว่ายอดที่ยืมมาไม่ได้";
        return "";
      },
      idOf: (debt) => debt.debt_id,
      path: "/debts",
    },
    recurring: {
      title: ["เพิ่มรายการประจำ", "แก้ไขรายการประจำ"],
      fill(item) {
        const type = item ? item.transaction_type : "expense";
        document.querySelector(`input[name='recurringType'][value='${type}']`).checked = true;
        fillRecurringChoices(type, item ? item.category : "", item ? item.debt_id : "");
        $("recurringAmount").value = item ? item.amount : "";
        $("recurringDay").value = String(item ? item.day_of_month : new Date().getDate());
        $("recurringDescription").value = item ? item.description : "";
      },
      body() {
        const type = recurringType();
        return {
          transaction_type: type, category: $("recurringCategory").value,
          amount: $("recurringAmount").value, day_of_month: Number($("recurringDay").value),
          description: $("recurringDescription").value.trim(),
          debt_id: type === "expense" ? ($("recurringDebt").value || null) : null,
        };
      },
      problem(body) {
        if (!(Number(body.amount) > 0)) return "จำนวนเงินต้องมากกว่า 0";
        return "";
      },
      idOf: (item) => item.entry_id,
      path: "/recurring",
      deleteWord: "หยุด",
    },
  };

  function openForm(kind, item) {
    const form = FORMS[kind];
    editing[kind] = item ? form.idOf(item) : "";
    form.fill(item);
    $(`${kind}ModalTitle`).textContent = form.title[item ? 1 : 0];
    $(`${kind}FormError`).hidden = true;
    $(`${kind}Modal`).hidden = false;
    $(`${kind}Form`).querySelector("input:not([type=radio])").focus();
  }

  function closeForm(kind) {
    $(`${kind}Modal`).hidden = true;
    $(`${kind}Add`).focus();
  }

  function showFormError(kind, message) {
    $(`${kind}FormError`).textContent = message;
    $(`${kind}FormError`).hidden = false;
  }

  async function saveForm(kind, event) {
    event.preventDefault();
    const form = FORMS[kind];
    const body = form.body();
    const problem = form.problem(body);
    if (problem) { showFormError(kind, problem); return; }
    const id = editing[kind];
    let saved;
    try {
      saved = await app().api(id ? "PATCH" : "POST", id ? `${form.path}/${id}` : form.path,
        { json: { ...body, user_id: app().getUserId() } });
    } catch (e) {
      showFormError(kind, "บันทึกไม่สำเร็จ: " + e.message.replace(/^\d+:\s*/, ""));
      return;
    }
    closeForm(kind);
    const posted = saved && saved.posted_now > 0 ? " และบันทึกของเดือนนี้แล้ว" : "";
    app().toast((id ? "แก้ไขแล้ว" : "เพิ่มแล้ว") + posted);
    refreshAfterChange(Boolean(posted));
  }

  // Only these cards reload (3 requests); the whole dashboard reloads only when a
  // save recorded a transaction, so quick edits stay under the server's rate limit.
  function refreshAfterChange(transactionsChanged) {
    if (transactionsChanged) app().loadDashboard();
    else load();
  }

  // ─── Delete ───
  async function remove(kind, item, name) {
    const word = FORMS[kind].deleteWord || "ลบ";
    const note = kind === "recurring" ? " (รายการที่บันทึกไปแล้วยังอยู่)" : "";
    if (!window.confirm(`${word} "${name}" ใช่ไหม?${note}`)) return;
    try {
      await app().api("DELETE", `${FORMS[kind].path}/${FORMS[kind].idOf(item)}`,
        { params: { user_id: app().getUserId() } });
    } catch (e) {
      app().toast("ลบไม่สำเร็จ", true);
      return;
    }
    app().toast(`${word} "${name}" แล้ว`);
    refreshAfterChange(false);
  }

  // ─── Wiring ───
  function wireOnce() {
    if (wired) return;
    wired = true;
    for (const kind of Object.keys(FORMS)) {
      $(`${kind}Add`).addEventListener("click", () => openForm(kind, null));
      $(`${kind}Cancel`).addEventListener("click", () => closeForm(kind));
      $(`${kind}Form`).addEventListener("submit", (e) => saveForm(kind, e));
      $(`${kind}Modal`).addEventListener("keydown", (e) => { if (e.key === "Escape") closeForm(kind); });
    }
    document.querySelectorAll("input[name='recurringType']").forEach((radio) =>
      radio.addEventListener("change", () => fillRecurringChoices(recurringType(), "", "")));
  }

  window.LedgerCards = { load };
})();
