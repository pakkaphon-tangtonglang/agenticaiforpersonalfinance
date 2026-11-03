/* Personal Finance AI — dashboard income/expense history
   Lists the selected month's entries and lets the user add, edit and delete
   them. Deleting hides the row at once and shows an undo toast for 10 s; the
   DELETE request is sent only if the user does not undo. Reuses app.js
   helpers via window.FinanceApp.
*/
(function () {
  "use strict";

  const UNDO_SECONDS = 10;
  // Keys mirror tools/expense_constants.py (EXPENSE_CATEGORIES / INCOME_TYPES).
  const CATEGORIES = {
    expense: {
      food: "อาหาร", transport: "การเดินทาง", housing: "ที่พักอาศัย", entertainment: "บันเทิง",
      utilities: "สาธารณูปโภค", health: "สุขภาพ", education: "การศึกษา", shopping: "ช้อปปิ้ง",
      investment: "ลงทุน", debt: "ชำระหนี้", other: "อื่นๆ",
    },
    income: {
      salary: "เงินเดือน", freelance: "ฟรีแลนซ์", bonus: "โบนัส", investment: "รายได้จากการลงทุน",
      rental: "ค่าเช่า", savings: "เงินออม/เงินสะสม", other: "อื่นๆ",
    },
  };

  const app = () => window.FinanceApp;
  const $ = (id) => document.getElementById(id);
  const history = { kind: "all", year: 0, month: 0, entries: [], editingId: "", wired: false };
  const pending = { id: "", row: null, timer: 0, ticker: 0 };

  // ─── Helpers ───
  function todayIso() {
    const now = new Date();
    return new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  }

  function shortDate(iso) {
    return new Date(iso + "T00:00:00").toLocaleDateString("th-TH", { day: "numeric", month: "short" });
  }

  function signedAmount(entry) {
    const value = app().thb(entry.amount);
    return entry.transaction_type === "income" ? "+" + value : "−" + value;
  }

  // ─── List ───
  // Each row sits in a wrapper that folds closed (grid-template-rows 1fr → 0fr)
  // when the filter hides it or it is deleted, so the list never jumps.
  function entryRow(entry) {
    const el = app().el;
    const item = el("li", { class: "history-item", dataset: { id: entry.id, kind: entry.transaction_type } },
      el("div", { class: "history-clip" }, entryContent(entry)));
    return item;
  }

  function entryContent(entry) {
    const el = app().el;
    return el("div", { class: `history-row is-${entry.transaction_type}` },
      el("span", { class: "history-date" }, shortDate(entry.transaction_date)),
      el("span", { class: "history-main" },
        el("span", { class: "history-category" }, entry.category_label),
        el("span", { class: "history-desc", title: entry.description }, entry.description)),
      el("span", { class: "history-amount num" }, signedAmount(entry)),
      el("span", { class: "history-actions" },
        el("button", { type: "button", class: "history-action", "aria-label": "แก้ไขรายการ",
          title: "แก้ไข", onclick: () => openForm(entry) }, "✎"),
        el("button", { type: "button", class: "history-action is-danger", "aria-label": "ลบรายการ",
          title: "ลบ", onclick: () => removeWithUndo(entry) }, "🗑")));
  }

  function renderList(page) {
    history.entries = page.entries;
    $("historyList").replaceChildren(...page.entries.map(entryRow));
    $("historyTotals").textContent =
      `รายรับ ${app().thb(page.total_income)} · รายจ่าย ${app().thb(page.total_expense)}`;
    applyFilter();
  }

  function setFolded(item, reason, folded) {
    item.classList.toggle(reason, folded);
    item.inert = item.classList.contains("is-filtered") || item.classList.contains("is-removed");
  }

  // Show the selected kind, update the chip counts and the empty note.
  function applyFilter() {
    const items = [...document.querySelectorAll("#historyList .history-item")];
    for (const item of items) {
      setFolded(item, "is-filtered", history.kind !== "all" && item.dataset.kind !== history.kind);
    }
    const live = items.filter((item) => !item.classList.contains("is-removed"));
    const counts = { all: live.length, income: 0, expense: 0 };
    live.forEach((item) => { counts[item.dataset.kind] += 1; });
    document.querySelectorAll(".history-filter-count").forEach((badge) => {
      badge.textContent = String(counts[badge.closest(".history-filter").dataset.kind] || 0);
    });
    const shown = counts[history.kind] || 0;
    $("historyCount").textContent = String(shown);
    $("historyEmpty").hidden = shown > 0;
  }

  async function load(year, month) {
    history.year = year;
    history.month = month;
    try {
      const page = await app().api("GET", "/transactions", {
        params: { user_id: app().getUserId(), year, month, type: "all" },
      });
      renderList(page);
    } catch (e) {
      app().toast("โหลดประวัติรายการไม่สำเร็จ", true);
    }
  }

  function setKind(kind) {
    history.kind = kind;
    document.querySelectorAll(".history-filter").forEach((b) => {
      const on = b.dataset.kind === kind;
      b.classList.toggle("is-active", on);
      b.setAttribute("aria-pressed", String(on));
    });
    applyFilter();
  }

  // ─── Add / edit form ───
  function selectedType() {
    return document.querySelector("input[name='historyType']:checked").value;
  }

  function fillCategories(type, selected) {
    const select = $("historyCategory");
    select.replaceChildren(...Object.entries(CATEGORIES[type]).map(([key, label]) =>
      new Option(label, key, false, key === selected)));
  }

  function openForm(entry) {
    history.editingId = entry ? entry.id : "";
    const type = entry ? entry.transaction_type : "expense";
    document.querySelector(`input[name='historyType'][value='${type}']`).checked = true;
    fillCategories(type, entry ? entry.category : "");
    $("historyDate").max = todayIso();
    $("historyDate").value = entry ? entry.transaction_date : todayIso();
    $("historyAmount").value = entry ? entry.amount : "";
    $("historyDescription").value = entry ? entry.description : "";
    $("historyModalTitle").textContent = entry ? "แก้ไขรายการ" : "เพิ่มรายการ";
    $("historyFormError").hidden = true;
    $("historyModal").hidden = false;
    $("historyAmount").focus();
  }

  function closeForm() {
    $("historyModal").hidden = true;
    $("historyAdd").focus();
  }

  function formBody() {
    return {
      user_id: app().getUserId(),
      transaction_type: selectedType(),
      transaction_date: $("historyDate").value,
      amount: $("historyAmount").value,
      category: $("historyCategory").value,
      description: $("historyDescription").value.trim(),
    };
  }

  function formError(body) {
    if (!body.transaction_date) return "กรุณาเลือกวันที่";
    if (body.transaction_date > todayIso()) return "เลือกวันที่ในอนาคตไม่ได้ (บันทึกเฉพาะรายการที่เกิดขึ้นแล้ว)";
    if (!(Number(body.amount) > 0)) return "จำนวนเงินต้องมากกว่า 0";
    return "";
  }

  async function saveForm(event) {
    event.preventDefault();
    const body = formBody();
    const problem = formError(body);
    if (problem) { showFormError(problem); return; }
    const path = history.editingId ? `/transactions/${history.editingId}` : "/transactions";
    try {
      await app().api(history.editingId ? "PATCH" : "POST", path, { json: body });
    } catch (e) {
      showFormError("บันทึกไม่สำเร็จ: " + e.message);
      return;
    }
    closeForm();
    app().toast(history.editingId ? "แก้ไขรายการแล้ว" : "เพิ่มรายการแล้ว");
    app().loadDashboard();
  }

  function showFormError(message) {
    $("historyFormError").textContent = message;
    $("historyFormError").hidden = false;
  }

  // ─── Delete with undo (10 s) ───
  function sendDelete(id) {
    const url = new URL(`/transactions/${id}`, app().apiBase);
    url.searchParams.set("user_id", app().getUserId());
    return fetch(url, { method: "DELETE", keepalive: true });
  }

  function hideUndoToast() {
    clearInterval(pending.ticker);
    clearTimeout(pending.timer);
    $("undoToast").hidden = true;
  }

  async function commitPendingDelete() {
    if (!pending.id) return;
    const id = pending.id;
    pending.id = "";
    hideUndoToast();
    try {
      const response = await sendDelete(id);
      if (!response.ok && response.status !== 404) throw new Error(String(response.status));
    } catch (e) {
      app().toast("ลบรายการไม่สำเร็จ", true);
    }
    app().loadDashboard();
  }

  function startCountdown() {
    let left = UNDO_SECONDS;
    $("undoToastCount").textContent = String(left);
    pending.ticker = setInterval(() => {
      left -= 1;
      $("undoToastCount").textContent = String(Math.max(left, 0));
    }, 1000);
    pending.timer = setTimeout(commitPendingDelete, UNDO_SECONDS * 1000);
  }

  function removeWithUndo(entry) {
    if (pending.id) commitPendingDelete();
    const row = document.querySelector(`.history-item[data-id='${entry.id}']`);
    if (row) setFolded(row, "is-removed", true);
    applyFilter();
    pending.id = entry.id;
    pending.row = row;
    $("undoToastText").textContent = `ลบ "${entry.description || entry.category_label}" แล้ว`;
    $("undoToast").hidden = false;
    startCountdown();
  }

  function undoDelete() {
    if (pending.row) setFolded(pending.row, "is-removed", false);
    applyFilter();
    pending.id = "";
    pending.row = null;
    hideUndoToast();
    $("historyAdd").focus();
  }

  // ─── Wiring ───
  function wireOnce() {
    if (history.wired) return;
    history.wired = true;
    $("historyAdd").addEventListener("click", () => openForm(null));
    $("historyCancel").addEventListener("click", closeForm);
    $("historyForm").addEventListener("submit", saveForm);
    $("undoToastBtn").addEventListener("click", undoDelete);
    document.querySelectorAll("input[name='historyType']").forEach((radio) =>
      radio.addEventListener("change", () => fillCategories(selectedType(), "")));
    document.querySelectorAll(".history-filter").forEach((b) =>
      b.addEventListener("click", () => setKind(b.dataset.kind)));
    $("historyModal").addEventListener("keydown", (e) => { if (e.key === "Escape") closeForm(); });
    window.addEventListener("pagehide", commitPendingDelete);
  }

  window.HistoryView = {
    load(year, month) { wireOnce(); return load(year, month); },
  };
})();
