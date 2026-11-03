"""Debt types and their Thai labels."""

DEBT_TYPES: dict[str, str] = {
    "personal": "ยืมคนรู้จัก",
    "credit_card": "บัตรเครดิต",
    "personal_loan": "สินเชื่อส่วนบุคคล",
    "home_loan": "สินเชื่อบ้าน",
    "car_loan": "สินเชื่อรถ",
    "student_loan": "กยศ.",
    "other": "หนี้อื่นๆ",
}
DEFAULT_DEBT_TYPE: str = "personal"
FALLBACK_DEBT_TYPE: str = "other"
