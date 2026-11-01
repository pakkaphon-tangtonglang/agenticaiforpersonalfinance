/* ============================================================
   Watchlist news alerts
   - timeControl(row): the 🔔 daily-time picker on each watchlist row
     (PUT /assets/watchlist/{id}/alert).
   - A poller checks /assets/notifications every minute while the app is
     open and shows new alerts as browser notifications (or a toast when
     the browser does not allow notifications).
   The server fires alerts at the chosen time; users who linked LINE
   (เชื่อมต่อ <user-id>) also get them in LINE.
   ============================================================ */
(function () {
  "use strict";

  const POLL_MS = 60 * 1000;
  const SEEN_KEY = "pfai_seen_alerts";
  const SEEN_LIMIT = 200;
  const BODY_LIMIT = 180;
  const app = () => window.FinanceApp;
  const supportsNotifications = () => "Notification" in window;

  // ─── Daily time picker ───
  function timeControl(row) {
    const el = app().el;
    const label = row.name || row.symbol;
    const input = el("input", {
      type: "time", class: "watchlist-alert-time", step: "60",
      "aria-label": `เวลาแจ้งข่าว ${label} ทุกวัน`, title: "เวลาแจ้งข่าวทุกวัน",
    });
    input.value = row.alert_time || "";
    const clear = el("button", { type: "button", class: "watchlist-alert-clear", "aria-label": `ปิดแจ้งข่าว ${label}` }, "×");
    const wrap = el("div", { class: "watchlist-alert" + (row.alert_time ? " is-on" : "") }, bellIcon(), input, clear);
    clear.hidden = !row.alert_time;
    input.addEventListener("change", () => saveAlert(row, input.value || null, wrap));
    clear.addEventListener("click", () => { input.value = ""; saveAlert(row, null, wrap); });
    return wrap;
  }

  function bellIcon() {
    const span = document.createElement("span");
    span.className = "watchlist-alert-bell";
    span.setAttribute("aria-hidden", "true");
    span.innerHTML = '<svg class="i"><use href="#i-bell"/></svg>';
    return span;
  }

  async function saveAlert(row, alertTime, wrap) {
    const label = row.name || row.symbol;
    try {
      const res = await app().api("PUT", `/assets/watchlist/${encodeURIComponent(row.id || row.symbol)}/alert`, {
        json: { user_id: app().getUserId(), alert_time: alertTime },
      });
      row.alert_time = res.alert_time;
      wrap.classList.toggle("is-on", Boolean(res.alert_time));
      wrap.querySelector(".watchlist-alert-clear").hidden = !res.alert_time;
      app().toast(res.alert_time ? `จะแจ้งข่าว ${label} ทุกวันเวลา ${res.alert_time} น.` : `ปิดแจ้งข่าว ${label} แล้ว`);
      if (res.alert_time) askPermission();
    } catch (e) {
      app().toast("ตั้งเวลาไม่สำเร็จ ใช้รูปแบบ 24 ชม. เช่น 08:30", true);
    }
  }

  // ─── Browser notification permission ───
  function askPermission() {
    if (!supportsNotifications() || Notification.permission !== "default") return;
    Notification.requestPermission().then(updatePermissionButton);
  }

  function updatePermissionButton() {
    const button = document.getElementById("alertPermissionBtn");
    if (button) button.hidden = !supportsNotifications() || Notification.permission !== "default";
  }

  // ─── Poller ───
  function readSeen() {
    try {
      const raw = localStorage.getItem(SEEN_KEY);
      return raw === null ? null : new Set(JSON.parse(raw));
    } catch (_) { return null; }
  }

  function saveSeen(seen) {
    try { localStorage.setItem(SEEN_KEY, JSON.stringify([...seen].slice(-SEEN_LIMIT))); }
    catch (_) { /* private mode: we may notify twice, never miss */ }
  }

  function plainText(markdown) {
    return String(markdown || "").replace(/[*_#`>\[\]]/g, "").replace(/\s+/g, " ").trim().slice(0, BODY_LIMIT);
  }

  function showAlert(item) {
    const title = `🔔 ข่าว ${app().displaySymbol(item.symbol)}`;
    if (supportsNotifications() && Notification.permission === "granted") {
      const note = new Notification(title, { body: plainText(item.message), tag: item.id, icon: "favicon.png" });
      note.onclick = () => { window.focus(); note.close(); };
      return;
    }
    app().toast(`${title} — ดูได้ในหน้าสินทรัพย์`);
  }

  // First run on this browser: remember what exists without notifying,
  // so old alerts do not pop up all at once.
  async function poll() {
    const userId = app() && app().getUserId();
    if (!userId) return;
    let items;
    try { items = await app().api("GET", "/assets/notifications", { params: { user_id: userId } }); }
    catch (_) { return; }
    const known = readSeen();
    const seen = known || new Set();
    const fresh = (items || []).filter((item) => !seen.has(item.id));
    fresh.forEach((item) => seen.add(item.id));
    saveSeen(seen);
    if (!known || !fresh.length) return;
    fresh.forEach(showAlert);
    if (app().loadNotifications) app().loadNotifications();
  }

  function start() {
    const button = document.getElementById("alertPermissionBtn");
    if (button) button.addEventListener("click", askPermission);
    updatePermissionButton();
    poll();
    setInterval(poll, POLL_MS);
    document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
  }

  window.WatchlistAlerts = { timeControl };
  start();
})();
