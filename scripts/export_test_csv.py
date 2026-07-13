"""Export all evaluation datasets to CSV files."""

import csv
import yaml
from pathlib import Path

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

CATEGORY_LABELS = {
    "fabricated_number": "ตัวเลขที่แต่งขึ้น",
    "fabricated_law": "กฎหมายที่แต่งขึ้น",
    "incorrect_calculation": "การคำนวณผิดพลาด",
    "missing_citation": "ขาดการอ้างอิง",
}

DIFFICULTY_LABELS = {
    "easy": "ง่าย",
    "medium": "ปานกลาง",
    "hard": "ยาก",
}

OUT_DIR = Path("data/evaluation")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved {len(rows)} rows -> {path.name}")


def export_tax() -> None:
    data = yaml.safe_load(Path("data/evaluation/tax_accuracy_dataset.yaml").read_text(encoding="utf-8"))
    rows = []
    for c in data["cases"]:
        ded = c.get("deductions_by_type", {})
        ded_parts = [f"{DEDUCTION_LABELS.get(k, k)}: {int(v):,} บาท" for k, v in ded.items()]
        rows.append({
            "case_id": c["case_id"],
            "description": c.get("description", ""),
            "query": c["query"],
            "gross_income_thb": int(c["gross_income"]),
            "deductions": " | ".join(ded_parts) if ded_parts else "ไม่มีค่าลดหย่อน",
            "expected_total_tax_thb": int(c["expected_total_tax"]),
            "expected_effective_rate_pct": f"{float(c['expected_effective_rate']) * 100:.2f}%",
            "tolerance_thb": int(c["tolerance_thb"]),
        })
    _write_csv(OUT_DIR / "tax_accuracy_dataset.csv", rows)


def export_routing() -> None:
    data = yaml.safe_load(Path("data/evaluation/routing_dataset.yaml").read_text(encoding="utf-8"))
    rows = []
    for c in data["cases"]:
        intent = c.get("expected_intent", "")
        rows.append({
            "case_id": c["case_id"],
            "query": c["query"],
            "expected_intent": intent,
            "expected_intent_th": INTENT_LABELS.get(intent, intent),
            "difficulty": c.get("difficulty", ""),
            "difficulty_th": DIFFICULTY_LABELS.get(c.get("difficulty", ""), ""),
        })
    _write_csv(OUT_DIR / "routing_dataset.csv", rows)


def export_hallucination() -> None:
    data = yaml.safe_load(Path("data/evaluation/hallucination_dataset.yaml").read_text(encoding="utf-8"))
    rows = []
    for c in data["cases"]:
        category = c.get("category", "")
        facts = c.get("known_facts", [])
        forbidden = c.get("forbidden_patterns", [])
        rows.append({
            "case_id": c["case_id"],
            "category": category,
            "category_th": CATEGORY_LABELS.get(category, category),
            "query": c["query"],
            "known_facts": " | ".join(facts),
            "forbidden_patterns": " | ".join(forbidden),
        })
    _write_csv(OUT_DIR / "hallucination_dataset.csv", rows)


def main() -> None:
    print("Exporting evaluation datasets to CSV...")
    export_tax()
    export_routing()
    export_hallucination()
    print("Done.")


if __name__ == "__main__":
    main()
