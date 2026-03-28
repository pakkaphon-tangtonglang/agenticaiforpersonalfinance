# mypy: ignore-errors
"""Generate PDF evaluation report comparing 3 LLM models."""

import json
from pathlib import Path

from fpdf import FPDF

FONT_PATH = "C:/Windows/Fonts/cordia.ttc"
OUTPUT_PATH = "data/evaluation/results/evaluation_report_3models.pdf"

MODEL_FILES = {
    "Gemini 2.5 Flash": "data/evaluation/results/5a120e98-140b-4d38-9fc0-123394d64469.json",
    "DeepSeek V3": "data/evaluation/results/68c4f247-8129-48cc-8067-abbcfb7ea698.json",
    "GPT-4o-mini": "data/evaluation/results/4833ac74-b1dd-485f-a64b-d9bd73ba3548.json",
}


def load_data(path: str) -> dict:
    return json.load(open(path, encoding="utf-8"))


class EvalPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("Cordia", "", FONT_PATH)
        self.add_font("Cordia", "B", FONT_PATH)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Cordia", "B", 12)
        self.cell(
            0,
            8,
            "Personal Finance AI - LLM Evaluation Report",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Cordia", "", 10)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.set_font("Cordia", "B", 16)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def sub_title(self, title: str):
        self.set_font("Cordia", "B", 14)
        self.set_text_color(41, 128, 185)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(1)

    def body_text(self, text: str):
        self.set_font("Cordia", "", 13)
        self.multi_cell(0, 7, text)
        self.ln(1)

    def add_table(self, headers: list[str], rows: list[list[str]], col_widths: list[int]):
        # Header
        self.set_font("Cordia", "B", 12)
        self.set_fill_color(52, 73, 94)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, h, border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_text_color(0, 0, 0)
        self.set_font("Cordia", "", 12)
        for ri, row in enumerate(rows):
            if ri % 2 == 0:
                self.set_fill_color(236, 240, 241)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                align = "L" if i == 0 else "C"
                self.cell(col_widths[i], 7, val, border=1, fill=True, align=align)
            self.ln()
        self.ln(3)

    def note_box(self, title: str, body: str):
        """Render a highlighted note box for failed cases."""
        self.set_font("Cordia", "B", 11)
        self.set_fill_color(253, 237, 236)
        self.set_text_color(192, 57, 43)
        self.cell(0, 7, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Cordia", "", 11)
        self.set_text_color(80, 80, 80)
        # Show full response
        self.multi_cell(0, 6, body, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)


def fmt_pct(val) -> str:
    return f"{float(val) * 100:.2f}%"


def fmt_dec(val, places=2) -> str:
    return f"{float(val):.{places}f}"


def fmt_sec(val) -> str:
    return f"{float(val):.2f}s"


def build_model_page(pdf: EvalPDF, name: str, d: dict):
    pdf.add_page()
    pdf.section_title(f"Model: {name}")
    pdf.body_text(f"Provider: {d['llm_provider']}    Model: {d['llm_model']}")
    pdf.body_text(f"Timestamp: {d['timestamp']}")

    r = d["routing"]
    pdf.sub_title("1. Routing Accuracy")
    pdf.body_text(f"Accuracy: {fmt_pct(r['accuracy'])} ({r['correct_count']}/{r['total_cases']})")
    pdf.body_text(f"Mean Latency: {fmt_sec(r['mean_latency_seconds'])}")
    # Per-intent table
    if r.get("per_intent_accuracy"):
        headers = ["Intent", "Accuracy"]
        rows = []
        for intent, acc in r["per_intent_accuracy"].items():
            rows.append([intent, fmt_pct(acc)])
        pdf.add_table(headers, rows, [90, 90])

    rag = d["rag_retrieval"]
    pdf.sub_title("2. RAG Retrieval")
    pdf.body_text(
        f"Precision@k: {fmt_dec(rag['mean_precision_at_k'], 4)}    "
        f"MRR: {fmt_dec(rag['mean_reciprocal_rank'], 4)}    "
        f"Keyword Hit Rate: {fmt_dec(rag['mean_keyword_hit_rate'], 4)}"
    )
    pdf.body_text(f"Mean Latency: {fmt_sec(rag['mean_latency_seconds'])}")
    if rag.get("per_domain_precision"):
        headers = ["Domain", "Precision"]
        rows = [[dom, fmt_dec(val, 4)] for dom, val in rag["per_domain_precision"].items()]
        pdf.add_table(headers, rows, [90, 90])

    t = d["tax_accuracy"]
    pdf.sub_title("3. Tax Accuracy")
    pdf.body_text(
        f"Accuracy: {fmt_pct(t['accuracy_rate'])} ({t['within_tolerance_count']}/{t['total_cases']})"
    )
    pdf.body_text(f"MAE: {fmt_dec(t['mean_absolute_error_thb'], 2)} THB")
    pdf.body_text(f"Mean Latency: {fmt_sec(t['mean_latency_seconds'])}")
    # Per-case results
    headers = ["Case", "Expected", "Extracted", "Error", "Pass"]
    rows = []
    for res in t["results"]:
        exp = str(res["expected_total_tax"])
        ext = str(res["extracted_total_tax"]) if res["extracted_total_tax"] is not None else "null"
        err = str(res["absolute_error_thb"]) if res["absolute_error_thb"] is not None else "N/A"
        ok = "Y" if res["is_within_tolerance"] else "N"
        rows.append([res["case_id"], exp, ext, err, ok])
    pdf.add_table(headers, rows, [30, 35, 35, 45, 20])

    # Failed case notes
    failed = [r for r in t["results"] if not r["is_within_tolerance"]]
    if failed:
        pdf.set_font("Cordia", "B", 13)
        pdf.cell(
            0, 8, f"Failed Cases Analysis ({len(failed)} cases)", new_x="LMARGIN", new_y="NEXT"
        )
        pdf.ln(1)
        for res in failed:
            ext = res["extracted_total_tax"] if res["extracted_total_tax"] is not None else "null"
            err = res["absolute_error_thb"] if res["absolute_error_thb"] is not None else "N/A"
            title = (
                f"{res['case_id']}: expected={res['expected_total_tax']}, "
                f"extracted={ext}, error={err}"
            )
            body = f"Query: {res['query']}\nResponse: {res['agent_response']}"
            pdf.note_box(title, body)

    h = d["hallucination"]
    pdf.sub_title("4. Hallucination Compliance")
    pdf.body_text(
        f"Compliance Rate: {fmt_pct(h['compliance_rate'])} ({h['compliant_count']}/{h['total_cases']})"
    )
    pdf.body_text(f"Mean Latency: {fmt_sec(h['mean_latency_seconds'])}")

    q = d["quality"]
    pdf.sub_title("5. Response Quality (Judge: Gemini 2.5 Flash)")
    headers = ["Metric", "Score"]
    rows = [
        ["Overall", fmt_dec(q["mean_overall"])],
        ["Relevance", fmt_dec(q["mean_relevance"])],
        ["Completeness", fmt_dec(q["mean_completeness"])],
        ["Accuracy", fmt_dec(q["mean_accuracy"])],
        ["Thai Language", fmt_dec(q["mean_thai_language_quality"])],
    ]
    pdf.add_table(headers, rows, [90, 90])

    p = d["performance"]
    pdf.sub_title("6. Performance")
    headers = ["Metric", "Value"]
    rows = [
        ["Mean Latency", fmt_sec(p["mean_total_latency"])],
        ["P50", fmt_sec(p["p50_latency"])],
        ["P95", fmt_sec(p["p95_latency"])],
        ["P99", fmt_sec(p["p99_latency"])],
        ["Total Input Tokens", str(p["total_input_tokens"])],
        ["Total Output Tokens", str(p["total_output_tokens"])],
    ]
    pdf.add_table(headers, rows, [90, 90])


def build_summary_page(pdf: EvalPDF, models: dict[str, dict]):
    pdf.add_page()
    pdf.section_title("Summary: 3-Model Comparison")
    pdf.ln(3)

    headers = ["Dimension", "Gemini 2.5 Flash", "DeepSeek V3", "GPT-4o-mini"]
    col_w = [50, 45, 45, 45]

    data = list(models.values())
    names = list(models.keys())

    def get_vals(key_path):
        results = []
        for d in data:
            obj = d
            for k in key_path:
                obj = obj[k]
            results.append(obj)
        return results

    rows = []

    # Routing
    vals = get_vals(["routing", "accuracy"])
    rows.append(["Routing Accuracy", *[fmt_pct(v) for v in vals]])

    # RAG
    vals = get_vals(["rag_retrieval", "mean_precision_at_k"])
    rows.append(["RAG Precision@k", *[fmt_dec(v, 4) for v in vals]])

    vals = get_vals(["rag_retrieval", "mean_reciprocal_rank"])
    rows.append(["RAG MRR", *[fmt_dec(v, 4) for v in vals]])

    vals = get_vals(["rag_retrieval", "mean_keyword_hit_rate"])
    rows.append(["RAG Keyword Hit", *[fmt_dec(v, 4) for v in vals]])

    # Tax
    vals = get_vals(["tax_accuracy", "accuracy_rate"])
    rows.append(["Tax Accuracy", *[fmt_pct(v) for v in vals]])

    vals = get_vals(["tax_accuracy", "mean_absolute_error_thb"])
    rows.append(["Tax MAE (THB)", *[fmt_dec(v, 0) for v in vals]])

    # Hallucination
    vals = get_vals(["hallucination", "compliance_rate"])
    rows.append(["Hallucination Compliance", *[fmt_pct(v) for v in vals]])

    # Quality
    vals = get_vals(["quality", "mean_overall"])
    rows.append(["Quality (Overall/5)", *[fmt_dec(v) for v in vals]])

    vals = get_vals(["quality", "mean_relevance"])
    rows.append(["  Relevance", *[fmt_dec(v) for v in vals]])

    vals = get_vals(["quality", "mean_completeness"])
    rows.append(["  Completeness", *[fmt_dec(v) for v in vals]])

    vals = get_vals(["quality", "mean_accuracy"])
    rows.append(["  Accuracy", *[fmt_dec(v) for v in vals]])

    vals = get_vals(["quality", "mean_thai_language_quality"])
    rows.append(["  Thai Language", *[fmt_dec(v) for v in vals]])

    # Performance
    vals = get_vals(["performance", "mean_total_latency"])
    rows.append(["Mean Latency", *[fmt_sec(v) for v in vals]])

    vals = get_vals(["performance", "p95_latency"])
    rows.append(["P95 Latency", *[fmt_sec(v) for v in vals]])

    pdf.add_table(headers, rows, col_w)

    # Best per dimension
    pdf.ln(3)
    pdf.sub_title("Best Model per Dimension")
    best_rows = [
        ["Routing", "GPT-4o-mini (97.14%)"],
        ["Tax Accuracy", "GPT-4o-mini (95.00%)"],
        ["Hallucination", "GPT-4o-mini (90.00%)"],
        ["Quality", "Gemini 2.5 Flash (4.87/5)"],
        ["Latency", "GPT-4o-mini (7.25s)"],
    ]
    pdf.add_table(["Dimension", "Best Model"], best_rows, [70, 115])


def main():
    pdf = EvalPDF()
    pdf.alias_nb_pages()

    # Title page
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Cordia", "B", 28)
    pdf.cell(0, 15, "Personal Finance AI", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Cordia", "B", 22)
    pdf.cell(0, 12, "LLM Evaluation Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Cordia", "", 16)
    pdf.cell(
        0,
        10,
        "3-Model Comparison: Gemini 2.5 Flash vs DeepSeek V3 vs GPT-4o-mini",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(5)
    pdf.cell(
        0,
        10,
        "6 Dimensions: Routing, RAG, Tax Accuracy, Hallucination, Quality, Performance",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(20)
    pdf.set_font("Cordia", "", 14)
    pdf.cell(0, 8, "Date: 2026-03-22", align="C", new_x="LMARGIN", new_y="NEXT")

    models = {}
    for name, path in MODEL_FILES.items():
        d = load_data(path)
        models[name] = d
        build_model_page(pdf, name, d)

    build_summary_page(pdf, models)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    pdf.output(OUTPUT_PATH)
    print(f"PDF saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
