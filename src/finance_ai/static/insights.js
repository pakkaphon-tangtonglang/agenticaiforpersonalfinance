/* Personal Finance AI — dashboard insight cards (GET /dashboard/trends)
   A small carousel: spending over 6 months, the category that is above the
   user's usual level, and holdings' 30-day price/NAV lines against SET50.
   Cards without data are left out. ‹ › swap cards with a blurred crossfade;
   each card ends with a question that opens the chat.
   buildPages() is pure and exported for node tests (module.exports).
*/
(function (root) {
  "use strict";

  const THAI_MONTHS = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."];
  const MAX_SPIKE_BARS = 3;
  const SWAP_MS = 160;

  // ─── Pure helpers ───
  const toNumber = (value) => Number(value || 0);
  const money = (value) => `${Math.round(value).toLocaleString("th-TH")} ฿`;
  const signedPercent = (value) => `${value > 0 ? "+" : ""}${value.toFixed(2)}%`;
  const shortDay = (iso) => {
    const [, month, day] = iso.split("-").map(Number);
    return `${day} ${THAI_MONTHS[month - 1]}`;
  };

  function spendingPage(trends) {
    const months = trends.spending || [];
    const current = months.length ? toNumber(months[months.length - 1].total) : 0;
    const average = toNumber(trends.average_spending);
    if (months.filter((m) => toNumber(m.total) > 0).length < 2 || average <= 0) return null;
    const change = Math.round(((current - average) / average) * 100);
    return {
      key: "spending",
      prose: [["เดือนนี้ใช้ไปแล้ว "], [money(current), "strong"], [` ${change >= 0 ? "สูง" : "ต่ำ"}กว่าค่าเฉลี่ย `],
        [`${change >= 0 ? "+" : ""}${change}%`, change > 0 ? "bad" : "good"]],
      stats: [{ label: "เดือนนี้", value: money(current), color: "--exp" }, { label: "ค่าเฉลี่ย", value: money(average), color: "--faint" }],
      chart: { type: "line", unit: "money", labels: months.map((m) => THAI_MONTHS[m.month - 1]),
        series: [{ label: "รายจ่าย", values: months.map((m) => toNumber(m.total)), color: "--exp", fill: true },
          { label: "ค่าเฉลี่ย", values: months.map(() => average), color: "--faint", dashed: true }] },
      question: change > 0 ? "ควรลดรายจ่ายตรงไหนก่อน?" : "เดือนนี้ใช้จ่ายเป็นยังไงบ้าง?",
    };
  }

  function spikePage(trends) {
    const spikes = [...(trends.spikes || [])].sort((a, b) => b.increase_pct - a.increase_pct).slice(0, MAX_SPIKE_BARS);
    if (!spikes.length) return null;
    const top = spikes[0];
    return {
      key: "spike",
      prose: [["หมวด "], [top.label, "strong"], [" เดือนนี้สูงกว่าปกติ "], [`+${top.increase_pct}%`, "bad"],
        [` (เทียบ ${top.months_compared} เดือนก่อน)`]],
      stats: [{ label: `${top.label} เดือนนี้`, value: money(toNumber(top.current)), color: "--exp" },
        { label: "ปกติ", value: money(toNumber(top.average)), color: "--faint" }],
      chart: { type: "bar", unit: "money", labels: spikes.map((s) => s.label),
        series: [{ label: "ปกติ", values: spikes.map((s) => toNumber(s.average)), color: "--line-strong" },
          { label: "เดือนนี้", values: spikes.map((s) => toNumber(s.current)), color: "--exp" }] },
      question: `ลดค่า${top.label}ยังไงดี?`,
    };
  }

  // Best and worst holdings (one when only one), plus the SET50 line, as % since day 1.
  function assetPage(trends) {
    const assets = [...(trends.assets || [])].sort((a, b) => toNumber(b.change_pct) - toNumber(a.change_pct));
    if (!assets.length) return null;
    const picked = assets.length > 1 ? [assets[0], assets[assets.length - 1]] : [assets[0]];
    const lines = trends.benchmark ? [...picked, trends.benchmark] : picked;
    const colors = ["--accent", "--plan"];
    const colorOf = (line, i) => (line.kind === "benchmark" ? "--faint" : colors[i]);
    const best = picked[0];
    const worst = picked[picked.length - 1];
    const prose = [["30 วันที่ผ่านมา "], [best.symbol, "strong"], [" "], toneOf(best)];
    if (worst !== best) prose.push([" ขณะที่ "], [worst.symbol, "strong"], [" "], toneOf(worst));
    return {
      key: "assets",
      prose,
      stats: lines.map((line, i) => ({ label: line.kind === "benchmark" ? line.name : line.symbol,
        value: signedPercent(toNumber(line.change_pct)), color: colorOf(line, i), tone: toNumber(line.change_pct) >= 0 ? "good" : "bad" })),
      chart: { type: "line", unit: "percent", labels: longestDays(lines),
        series: lines.map((line, i) => ({ label: line.kind === "benchmark" ? line.name : line.symbol,
          values: percentSeries(line.points), raw: line.points.map((p) => toNumber(p.value)),
          color: colorOf(line, i), dashed: line.kind === "benchmark" })) },
      note: "เทียบเป็น % จากวันแรก · NAV กองทุนช้ากว่า 1 วัน · เส้นประ = ตลาด SET50",
      question: "ควรปรับพอร์ตไหม?",
    };
  }

  function toneOf(line) {
    const change = toNumber(line.change_pct);
    return [signedPercent(change), change >= 0 ? "good" : "bad"];
  }

  function percentSeries(points) {
    const first = toNumber(points[0] && points[0].value);
    return points.map((p) => (first > 0 ? Number((((toNumber(p.value) - first) / first) * 100).toFixed(2)) : 0));
  }

  function longestDays(lines) {
    const longest = lines.reduce((a, b) => (b.points.length > a.points.length ? b : a));
    return longest.points.map((p) => shortDay(p.day));
  }

  function buildPages(trends) {
    if (!trends) return [];
    return [spendingPage(trends), spikePage(trends), assetPage(trends)].filter(Boolean);
  }

  const helpers = { buildPages };
  if (typeof module !== "undefined" && module.exports) { module.exports = helpers; return; }

  // ─── Browser: carousel ───
  const view = { pages: [], index: 0, chart: null, requestId: 0 };
  const $ = (id) => document.getElementById(id);
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const app = () => root.FinanceApp;

  async function load(year, month) {
    const requestId = ++view.requestId;
    let trends = null;
    try {
      trends = await app().api("GET", "/dashboard/trends", { params: { user_id: app().getUserId(), year, month } });
    } catch (e) { /* the cards are optional */ }
    if (requestId !== view.requestId) return;  // a newer month was picked meanwhile
    view.pages = buildPages(trends);
    view.index = 0;
    $("insightCard").hidden = !view.pages.length;
    if (view.pages.length) render();
  }

  function proseNodes(parts) {
    return parts.map(([text, tone]) => {
      if (!tone) return document.createTextNode(text);
      return app().el("span", { class: `insight-${tone}` }, text);
    });
  }

  function render() {
    const el = app().el;
    const page = view.pages[view.index];
    $("insightCount").textContent = String(view.pages.length);
    $("insightPager").hidden = view.pages.length < 2;
    $("insightBody").replaceChildren(
      el("p", { class: "insight-prose" }, ...proseNodes(page.prose)),
      el("div", { class: "insight-panel" },
        el("div", { class: "insight-stats" }, ...page.stats.map(statNode)),
        el("div", { class: "insight-chart" }, el("canvas", { id: "insightCanvas", role: "img", "aria-label": page.prose.map((p) => p[0]).join("") })),
        page.note ? el("p", { class: "insight-note" }, page.note) : null),
      el("button", { type: "button", class: "insight-ask", onclick: () => app().askInChat(page.question) }, page.question));
    $("insightDots").replaceChildren(...view.pages.map((_, i) => el("i", { class: i === view.index ? "is-on" : "" })));
    drawChart(page.chart);
  }

  function statNode(stat) {
    const el = app().el;
    return el("div", { class: "insight-stat" },
      el("span", { class: "insight-stat-label" }, el("i", { style: `background:var(${stat.color})` }), stat.label),
      el("span", { class: "insight-stat-value" + (stat.tone ? ` insight-${stat.tone}` : "") }, stat.value));
  }

  function drawChart(spec) {
    if (view.chart) view.chart.destroy();
    if (typeof Chart === "undefined") return;
    view.chart = new Chart($("insightCanvas"), chartConfig(spec));
  }

  function datasetOf(spec, series) {
    const color = cssVar(series.color);
    return {
      label: series.label, data: series.values, raw: series.raw,
      borderColor: color, backgroundColor: spec.type === "bar" ? color : tint(color),
      fill: Boolean(series.fill), tension: 0.4, borderWidth: 2.25, pointRadius: 0, pointHoverRadius: 4,
      borderDash: series.dashed ? [4, 4] : [], borderRadius: 6, maxBarThickness: 22,
    };
  }

  // Canvas fills need a plain colour: add ~14% alpha to #rrggbb tokens.
  function tint(color) {
    return /^#[0-9a-f]{6}$/i.test(color) ? `${color}24` : color;
  }

  function tooltipLabel(spec) {
    return (ctx) => {
      if (spec.unit !== "percent") return ` ${ctx.dataset.label}: ${money(ctx.parsed.y)}`;
      const raw = ctx.dataset.raw ? ctx.dataset.raw[ctx.dataIndex] : null;
      return ` ${ctx.dataset.label}: ${signedPercent(ctx.parsed.y)}${raw == null ? "" : ` (${raw.toLocaleString("th-TH")})`}`;
    };
  }

  function chartConfig(spec) {
    const muted = cssVar("--muted");
    const font = { family: "'Inter','Sarabun',sans-serif", size: 11 };
    return {
      type: spec.type,
      data: { labels: spec.labels, datasets: spec.series.map((series) => datasetOf(spec, series)) },
      options: {
        responsive: true, maintainAspectRatio: false,
        animation: reduceMotion() ? false : { duration: 600, easing: "easeOutQuart" },
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: false }, tooltip: {
          backgroundColor: cssVar("--sheet"), titleColor: cssVar("--ink"), bodyColor: cssVar("--ink"),
          borderColor: cssVar("--line-strong"), borderWidth: 1, padding: 8, usePointStyle: true,
          titleFont: font, bodyFont: font, callbacks: { label: tooltipLabel(spec) } } },
        scales: {
          x: { grid: { display: false }, border: { display: false }, ticks: { color: muted, font, maxTicksLimit: 6 } },
          y: { grid: { color: cssVar("--line") }, border: { display: false }, ticks: { color: muted, font, maxTicksLimit: 4,
            callback: (value) => (spec.unit === "percent" ? `${value}%` : `${Math.round(value / 1000)}k`) } },
        },
      },
    };
  }

  function move(step) {
    const count = view.pages.length;
    const next = () => { view.index = (view.index + step + count) % count; render(); };
    if (reduceMotion()) { next(); return; }
    const body = $("insightBody");
    body.classList.add("is-swapping");
    setTimeout(() => { next(); body.classList.remove("is-swapping"); }, SWAP_MS);
  }

  function setup() {
    $("insightPrev").addEventListener("click", () => move(-1));
    $("insightNext").addEventListener("click", () => move(1));
  }

  root.InsightCards = {
    load,
    redraw() { if (view.pages.length && !$("insightCard").hidden) render(); },
  };
  document.addEventListener("DOMContentLoaded", setup);
})(typeof window !== "undefined" ? window : this);
