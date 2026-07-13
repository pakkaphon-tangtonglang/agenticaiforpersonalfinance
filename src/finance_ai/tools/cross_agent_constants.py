"""Constants for cross-agent data sharing.

Defines domain mappings and summary types used when agents
query data from other financial domains.
"""

# Domain name -> Thai display label
CROSS_AGENT_DOMAINS: dict[str, str] = {
    "tax": "ภาษี",
    "expense": "ค่าใช้จ่าย",
    "investment": "การลงทุน",
    "planning": "วางแผนการเงิน",
    "income": "รายได้",
}

# Summary type -> Thai display label
SUMMARY_TYPES: dict[str, str] = {
    "expense_summary": "สรุปค่าใช้จ่าย",
    "portfolio_summary": "สรุปพอร์ตการลงทุน",
    "goals_summary": "สรุปเป้าหมายการเงิน",
    "income_summary": "สรุปรายได้",
    "tax_filing_summary": "สรุปการยื่นภาษี",
}
