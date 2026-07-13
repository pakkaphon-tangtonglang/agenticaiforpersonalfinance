"""Generate test dataset PDF with questions and expected answers."""

from pathlib import Path

import json

import yaml
from fpdf import FPDF

# Leelawadee supports both Thai and Latin characters
LEELA_PATH = Path("C:/Windows/Fonts/leelawad.ttf")
LEELA_BOLD_PATH = Path("C:/Windows/Fonts/leelawdb.ttf")
OUTPUT_PATH = Path("data/evaluation/test_dataset.pdf")

INTENT_LABELS = {
    "tax": "ภาษี",
    "expense": "ค่าใช้จ่าย",
    "investment": "การลงทุน",
    "asset_monitoring": "ติดตามสินทรัพย์",
    "planning": "วางแผนการเงิน",
    "recommendation": "คำแนะนำ",
    "report": "รายงาน",
    "general": "ทั่วไป",
    "unknown": "นอกขอบเขต",
}

DIFFICULTY_LABELS = {
    "easy": "ง่าย",
    "medium": "ปานกลาง",
    "hard": "ยาก",
}

CATEGORY_LABELS = {
    "fabricated_number": "ตัวเลขที่แต่งขึ้น",
    "fabricated_law": "กฎหมายที่แต่งขึ้น",
    "incorrect_calculation": "การคำนวณผิดพลาด",
    "missing_citation": "ขาดการอ้างอิง",
}

DEDUCTION_LABELS = {
    "personal_allowance": "ลดหย่อนส่วนตัว",
    "social_security": "ประกันสังคม",
    "life_insurance": "ประกันชีวิต",
    "health_insurance": "ประกันสุขภาพ",
    "rmf": "RMF",
    "ssf": "SSF",
    "child_allowance": "ค่าลดหย่อนบุตร",
    "spouse_allowance": "ค่าลดหย่อนคู่สมรส",
    "parent_allowance": "ค่าลดหย่อนบิดามารดา",
    "mortgage_interest": "ดอกเบี้ยบ้าน",
    "provident_fund": "กองทุนสำรองเลี้ยงชีพ",
    "donations": "เงินบริจาค",
}


def _make_pdf() -> FPDF:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("Leela", "", str(LEELA_PATH))
    pdf.add_font("Leela", "B", str(LEELA_BOLD_PATH))
    return pdf


def _f(pdf: FPDF, size: int, bold: bool = False) -> None:
    """Set Leelawadee font (supports Thai + Latin)."""
    pdf.set_font("Leela", style="B" if bold else "", size=size)


def _add_cover(pdf: FPDF) -> None:
    pdf.add_page()
    pdf.ln(40)
    _f(pdf, 22, bold=True)
    pdf.cell(0, 12, "ชุดข้อมูลทดสอบระบบ", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    _f(pdf, 16, bold=True)
    pdf.cell(0, 10, "Agentic AI for Personal Finance", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    _f(pdf, 12)
    pdf.cell(0, 8, "ประกอบด้วยชุดทดสอบ 3 หมวด:", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    _f(pdf, 11)
    pdf.cell(0, 8, "หมวดที่ 1 — การคำนวณภาษี (Tax Accuracy)  20 กรณี", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "หมวดที่ 2 — การจำแนกเจตนา (Intent Routing)  27 กรณี", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "หมวดที่ 3 — ป้องกัน Hallucination  10 กรณี", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    _f(pdf, 10)
    pdf.cell(0, 7, "รวมทั้งหมด 57 กรณีทดสอบ", align="C", new_x="LMARGIN", new_y="NEXT")


def _section_header(pdf: FPDF, title: str, subtitle: str) -> None:
    pdf.add_page()
    pdf.set_fill_color(33, 97, 140)
    pdf.rect(0, 0, 210, 35, "F")
    pdf.set_text_color(255, 255, 255)
    _f(pdf, 16, bold=True)
    pdf.set_y(10)
    pdf.cell(0, 10, title, align="C", new_x="LMARGIN", new_y="NEXT")
    _f(pdf, 11)
    pdf.cell(0, 8, subtitle, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.set_y(45)


def _draw_row(pdf: FPDF, label: str, value: str, fill: bool = False) -> None:
    if fill:
        pdf.set_fill_color(245, 247, 250)
    else:
        pdf.set_fill_color(255, 255, 255)
    _f(pdf, 9, bold=True)
    pdf.cell(45, 7, label, border=1, fill=True, new_x="RIGHT", new_y="TOP")
    _f(pdf, 9)
    pdf.multi_cell(0, 7, value, border=1, fill=True, new_x="LMARGIN", new_y="NEXT")


def _add_tax_section(pdf: FPDF, cases: list) -> None:
    _section_header(pdf, "หมวดที่ 1: ชุดทดสอบการคำนวณภาษี", f"จำนวน {len(cases)} กรณีทดสอบ")

    for case in cases:
        if pdf.get_y() > 240:
            pdf.add_page()

        pdf.set_fill_color(52, 152, 219)
        pdf.set_text_color(255, 255, 255)
        _f(pdf, 10, bold=True)
        label = f"  {case['case_id']}  |  {case.get('description', '')}"
        pdf.cell(0, 8, label, fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        fill = False
        _draw_row(pdf, "คำถาม", case["query"], fill)
        fill = not fill

        income = f"{int(case['gross_income']):,} บาท/ปี"
        _draw_row(pdf, "รายได้รวม", income, fill)
        fill = not fill

        deductions = case.get("deductions_by_type", {})
        if deductions:
            ded_parts = [
                f"{DEDUCTION_LABELS.get(k, k)}: {int(v):,} บาท"
                for k, v in deductions.items()
            ]
            ded_text = "  |  ".join(ded_parts)
        else:
            ded_text = "ไม่มีค่าลดหย่อน"
        _draw_row(pdf, "ค่าลดหย่อน", ded_text, fill)

        pdf.set_fill_color(232, 245, 233)
        _f(pdf, 9, bold=True)
        tax_thb = f"{int(case['expected_total_tax']):,} บาท"
        rate_pct = f"{float(case['expected_effective_rate']) * 100:.2f}%"
        tol_thb = f"+/-{int(case['tolerance_thb']):,} บาท"
        pdf.cell(45, 7, "เฉลย: ภาษีที่ต้องชำระ", border=1, fill=True, new_x="RIGHT", new_y="TOP")
        _f(pdf, 9)
        answer_text = f"{tax_thb}   (อัตราภาษีแท้จริง {rate_pct})   ยอมรับได้ {tol_thb}"
        pdf.cell(0, 7, answer_text, border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.ln(3)


def _add_routing_section(pdf: FPDF, cases: list) -> None:
    _section_header(pdf, "หมวดที่ 2: ชุดทดสอบการจำแนกเจตนา", f"จำนวน {len(cases)} กรณีทดสอบ")
    _draw_table_header_routing(pdf)

    for i, case in enumerate(cases):
        if pdf.get_y() > 265:
            pdf.add_page()
            _draw_table_header_routing(pdf)

        fill = i % 2 == 0
        pdf.set_fill_color(*(245, 247, 250) if fill else (255, 255, 255))

        _f(pdf, 8, bold=True)
        pdf.cell(28, 7, case["case_id"], border=1, fill=True, new_x="RIGHT", new_y="TOP")
        _f(pdf, 8)
        pdf.cell(90, 7, case["query"], border=1, fill=True, new_x="RIGHT", new_y="TOP")

        intent = case.get("expected_intent", "")
        intent_th = INTENT_LABELS.get(intent, intent)
        label = f"{intent_th} ({intent})"
        pdf.cell(42, 7, label, border=1, fill=True, new_x="RIGHT", new_y="TOP")

        diff = case.get("difficulty", "easy")
        diff_label = DIFFICULTY_LABELS.get(diff, diff)
        pdf.cell(30, 7, diff_label, border=1, fill=True, new_x="LMARGIN", new_y="NEXT")


def _draw_table_header_routing(pdf: FPDF) -> None:
    pdf.set_fill_color(33, 97, 140)
    pdf.set_text_color(255, 255, 255)
    _f(pdf, 9, bold=True)
    pdf.cell(28, 8, "รหัสทดสอบ", border=1, fill=True, new_x="RIGHT", new_y="TOP")
    pdf.cell(90, 8, "คำถาม", border=1, fill=True, new_x="RIGHT", new_y="TOP")
    pdf.cell(42, 8, "เฉลย: Agent ที่ถูกต้อง", border=1, fill=True, new_x="RIGHT", new_y="TOP")
    pdf.cell(30, 8, "ระดับ", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def _add_hallucination_section(pdf: FPDF, cases: list) -> None:
    _section_header(pdf, "หมวดที่ 3: ชุดทดสอบป้องกัน Hallucination", f"จำนวน {len(cases)} กรณีทดสอบ")

    for case in cases:
        if pdf.get_y() > 240:
            pdf.add_page()

        pdf.set_fill_color(155, 89, 182)
        pdf.set_text_color(255, 255, 255)
        _f(pdf, 10, bold=True)
        category = case.get("category", "")
        category_th = CATEGORY_LABELS.get(category, category)
        pdf.cell(0, 8, f"  {case['case_id']}  |  ประเภท: {category_th}",
                 fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        fill = False
        _draw_row(pdf, "คำถาม", case["query"], fill)
        fill = not fill

        facts = "\n".join(f"- {f}" for f in case.get("known_facts", []))
        _draw_row(pdf, "เฉลย: ข้อเท็จจริง", facts, fill)
        fill = not fill

        forbidden = case.get("forbidden_patterns", [])
        if forbidden:
            forb_text = "  |  ".join(forbidden)
            pdf.set_fill_color(255, 235, 238)
            _f(pdf, 9, bold=True)
            pdf.cell(45, 7, "ห้ามตอบว่า", border=1, fill=True, new_x="RIGHT", new_y="TOP")
            _f(pdf, 9)
            pdf.multi_cell(0, 7, forb_text, border=1, fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.ln(3)


def _load_eval_results() -> list[dict]:
    """Load and aggregate evaluation results per model from JSON files."""
    results_dir = Path("data/evaluation/results")
    aggregated: dict[str, dict] = {}

    for f in sorted(results_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        provider = d.get("llm_provider", "") or ""
        model = d.get("llm_model", "") or ""
        if not provider or not model:
            continue
        key = f"{provider}/{model}"
        if key not in aggregated:
            aggregated[key] = {"provider": provider, "model": model, "runs": []}
        aggregated[key]["runs"].append(d)

    rows = []
    for key, entry in aggregated.items():
        runs = entry["runs"]

        def avg(getter):
            vals = [getter(r) for r in runs if getter(r) is not None]
            return sum(vals) / len(vals) if vals else None

        routing_acc = avg(lambda r: float((r.get("routing") or {}).get("accuracy", 0)) if r.get("routing") else None)
        tax_pass = avg(lambda r: float((r.get("tax_accuracy") or {}).get("accuracy_rate", 0)) if r.get("tax_accuracy") else None)
        tax_mae = avg(lambda r: float((r.get("tax_accuracy") or {}).get("mean_absolute_error_thb", 0)) if r.get("tax_accuracy") else None)
        tax_latency = avg(lambda r: float((r.get("tax_accuracy") or {}).get("mean_latency_seconds", 0)) if r.get("tax_accuracy") else None)
        quality_overall = avg(lambda r: float((r.get("quality") or {}).get("mean_overall", 0)) if r.get("quality") else None)
        quality_thai = avg(lambda r: float((r.get("quality") or {}).get("mean_thai_language_quality", 0)) if r.get("quality") else None)

        rows.append({
            "model": entry["model"],
            "routing_acc": routing_acc,
            "tax_pass": tax_pass,
            "tax_mae": tax_mae,
            "tax_latency": tax_latency,
            "quality_overall": quality_overall,
            "quality_thai": quality_thai,
            "runs": len(runs),
        })

    return rows


def _best(rows: list[dict], key: str, higher_is_better: bool = True) -> str:
    """Return the model name with the best value for a metric."""
    valid = [(r["model"], r[key]) for r in rows if r[key] is not None]
    if not valid:
        return ""
    return max(valid, key=lambda x: x[1] if higher_is_better else -x[1])[0]


def _fmt(value, fmt: str = ".2f", suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:{fmt}}{suffix}"


def _add_llm_selection_section(pdf: FPDF, rows: list[dict]) -> None:
    _section_header(
        pdf,
        "ภาคที่ 4: ผลการเปรียบเทียบและคัดเลือก LLM",
        f"เปรียบเทียบ {len(rows)} โมเดล จากผลการทดสอบจริง",
    )

    # --- intro ---
    _f(pdf, 10)
    pdf.multi_cell(
        0, 7,
        "การคัดเลือกโมเดลภาษา (LLM) ดำเนินการโดยรันชุดทดสอบเดียวกันกับทุกโมเดล "
        "แล้วเปรียบเทียบตามเกณฑ์ 5 มิติ ได้แก่ "
        "ความแม่นยำในการจำแนกเจตนา (Routing Accuracy), "
        "อัตราการผ่านการคำนวณภาษี (Tax Pass Rate), "
        "ค่าคลาดเคลื่อนภาษีเฉลี่ย (Tax MAE), "
        "เวลาตอบสนองเฉลี่ย (Latency) และ "
        "คะแนนคุณภาพคำตอบโดยรวม (Quality Score 1-5)",
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.ln(4)

    # --- comparison table ---
    _f(pdf, 10, bold=True)
    pdf.cell(0, 8, "ตารางสรุปผลการทดสอบเปรียบเทียบ", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    col_w = [52, 22, 20, 24, 22, 22, 22]
    headers = ["โมเดล", "Routing\nAcc (%)", "Tax\nPass (%)", "Tax MAE\n(THB)", "Latency\n(s)", "Quality\nOverall", "Thai\nQuality"]

    # Header row
    pdf.set_fill_color(33, 97, 140)
    pdf.set_text_color(255, 255, 255)
    _f(pdf, 8, bold=True)
    for w, h in zip(col_w, headers):
        # Use first line only for header (no newlines in cell)
        label = h.replace("\n", " ")
        pdf.cell(w, 8, label, border=1, fill=True, new_x="RIGHT", new_y="TOP")
    pdf.set_x(pdf.l_margin)
    pdf.ln(8)
    pdf.set_text_color(0, 0, 0)

    best_routing = _best(rows, "routing_acc")
    best_tax_pass = _best(rows, "tax_pass")
    best_tax_mae = _best(rows, "tax_mae", higher_is_better=False)
    best_latency = _best(rows, "tax_latency", higher_is_better=False)
    best_quality = _best(rows, "quality_overall")

    for i, row in enumerate(rows):
        fill = i % 2 == 0
        pdf.set_fill_color(*(245, 247, 250) if fill else (255, 255, 255))

        model_short = row["model"].split("/")[-1]
        values = [
            model_short,
            _fmt(row["routing_acc"] * 100 if row["routing_acc"] is not None else None, ".1f", "%"),
            _fmt(row["tax_pass"] * 100 if row["tax_pass"] is not None else None, ".1f", "%"),
            _fmt(row["tax_mae"], ".0f", " THB"),
            _fmt(row["tax_latency"], ".1f", "s"),
            _fmt(row["quality_overall"], ".2f"),
            _fmt(row["quality_thai"], ".2f"),
        ]
        bests = [None, best_routing, best_tax_pass, best_tax_mae, best_latency, best_quality, None]

        for j, (w, val) in enumerate(zip(col_w, values)):
            is_best = bests[j] is not None and row["model"] == bests[j]
            if is_best:
                pdf.set_fill_color(209, 236, 241)
            elif fill:
                pdf.set_fill_color(245, 247, 250)
            else:
                pdf.set_fill_color(255, 255, 255)
            _f(pdf, 8, bold=is_best)
            pdf.cell(w, 7, val, border=1, fill=True, new_x="RIGHT", new_y="TOP")

        pdf.set_x(pdf.l_margin)
        pdf.ln(7)

    pdf.ln(3)
    _f(pdf, 8)
    pdf.cell(0, 6, "* เซลล์สีฟ้า = ดีที่สุดในแต่ละมิติ", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # --- per-intent routing breakdown ---
    _f(pdf, 10, bold=True)
    pdf.cell(0, 8, "ความแม่นยำการจำแนกเจตนา แยกตาม Agent (Routing per Intent)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    intents = list(INTENT_LABELS.keys())
    result_files = sorted(Path("data/evaluation/results").glob("*.json"))
    model_routing: dict[str, dict] = {}
    for f in result_files:
        d = json.loads(f.read_text(encoding="utf-8"))
        model = d.get("llm_model", "") or ""
        if not model or not d.get("routing"):
            continue
        per_intent = d["routing"].get("per_intent_accuracy", {})
        if model not in model_routing:
            model_routing[model] = {}
        for intent, acc in per_intent.items():
            if intent not in model_routing[model]:
                model_routing[model][intent] = []
            model_routing[model][intent].append(float(acc))

    model_names = list(model_routing.keys())
    intent_col_w = 38
    model_col_w = int((190 - intent_col_w) / max(len(model_names), 1))

    pdf.set_fill_color(33, 97, 140)
    pdf.set_text_color(255, 255, 255)
    _f(pdf, 8, bold=True)
    pdf.cell(intent_col_w, 8, "Intent / Agent", border=1, fill=True, new_x="RIGHT", new_y="TOP")
    for m in model_names:
        pdf.cell(model_col_w, 8, m.split("/")[-1], border=1, fill=True, new_x="RIGHT", new_y="TOP")
    pdf.set_x(pdf.l_margin)
    pdf.ln(8)
    pdf.set_text_color(0, 0, 0)

    intent_order = ["tax", "expense", "investment", "planning", "recommendation", "report", "general", "unknown"]
    for idx, intent in enumerate(intent_order):
        intent_th = INTENT_LABELS.get(intent, intent)
        fill = idx % 2 == 0
        pdf.set_fill_color(*(245, 247, 250) if fill else (255, 255, 255))
        _f(pdf, 8)
        label = f"{intent_th} ({intent})"
        pdf.cell(intent_col_w, 7, label, border=1, fill=True, new_x="RIGHT", new_y="TOP")
        for m in model_names:
            vals = model_routing.get(m, {}).get(intent, [])
            avg_val = sum(vals) / len(vals) if vals else None
            cell_text = _fmt(avg_val * 100 if avg_val is not None else None, ".0f", "%")
            if avg_val is not None and avg_val == 1.0:
                pdf.set_fill_color(209, 236, 241)
            elif fill:
                pdf.set_fill_color(245, 247, 250)
            else:
                pdf.set_fill_color(255, 255, 255)
            _f(pdf, 8, bold=(avg_val is not None and avg_val == 1.0))
            pdf.cell(model_col_w, 7, cell_text, border=1, fill=True, new_x="RIGHT", new_y="TOP")
        pdf.set_x(pdf.l_margin)
        pdf.ln(7)

    pdf.ln(6)

    # --- conclusion ---
    pdf.set_fill_color(235, 245, 251)
    pdf.set_draw_color(33, 97, 140)
    _f(pdf, 10, bold=True)
    pdf.cell(0, 8, "สรุปผลและเหตุผลการคัดเลือก", new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(0, 0, 0)

    conclusions = [
        ("Gemini 2.5 Flash",
         "ค่า Tax MAE ต่ำที่สุด (53 THB) และคะแนนคุณภาพภาษาไทยสูงสุด (4.93/5.00) "
         "เหมาะกับงานที่ต้องการความแม่นยำของตัวเลขสูงและการสื่อสารเป็นภาษาไทย เช่น Tax Agent และ Planning Agent"),
        ("GPT-4o-mini",
         "Routing Accuracy สูงสุด (97.14%) และ Tax Pass Rate สูงสุด (95%) "
         "พร้อม Latency เฉลี่ยต่ำที่สุด (8.3s) เหมาะกับ Router Agent ที่ต้องการความเร็วและความถูกต้องในการจำแนกคำถาม"),
        ("DeepSeek v3",
         "Tax Pass Rate ไม่เสถียร (45-55%) ซึ่งต่ำกว่าเกณฑ์ที่ยอมรับได้ "
         "ไม่แนะนำสำหรับงานคำนวณภาษีในระบบนี้"),
    ]

    for model_name, reason in conclusions:
        pdf.set_fill_color(245, 247, 250)
        _f(pdf, 9, bold=True)
        pdf.cell(0, 7, f"  {model_name}", border="LTR", fill=True, new_x="LMARGIN", new_y="NEXT")
        _f(pdf, 9)
        pdf.multi_cell(0, 7, f"  {reason}", border="LBR", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)


def main() -> None:
    tax_data = yaml.safe_load(Path("data/evaluation/tax_accuracy_dataset.yaml").read_text(encoding="utf-8"))
    routing_data = yaml.safe_load(Path("data/evaluation/routing_dataset.yaml").read_text(encoding="utf-8"))
    hal_data = yaml.safe_load(Path("data/evaluation/hallucination_dataset.yaml").read_text(encoding="utf-8"))
    eval_rows = _load_eval_results()

    pdf = _make_pdf()
    _add_cover(pdf)
    _add_tax_section(pdf, tax_data["cases"])
    _add_routing_section(pdf, routing_data["cases"])
    _add_hallucination_section(pdf, hal_data["cases"])
    _add_llm_selection_section(pdf, eval_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUTPUT_PATH))
    size_kb = OUTPUT_PATH.stat().st_size // 1024
    print(f"PDF saved -> {OUTPUT_PATH}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
