"""LangGraph tools for debts: owing money, paying it back, and listing what is left."""

from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.session_helper import get_tool_session
from finance_ai.agents.tax_tools import parse_decimal_value
from finance_ai.database.models.debt import Debt
from finance_ai.tools.debt_constants import DEBT_TYPES
from finance_ai.tools.debt_service import (
    apply_debt_payment,
    find_active_debt,
    record_debt,
    set_debt_balance,
    summarize_debts,
)
from finance_ai.tools.expense_service import create_expense_transaction

DEBT_CATEGORY: str = "debt"


def debt_to_dict(debt: Debt) -> dict[str, Any]:
    """Plain values for the LLM and the API (money as strings).

    Example:
        >>> debt_to_dict(debt)["creditor"]
        'เพื่อน'
    """
    return {
        "debt_id": debt.id,
        "creditor": debt.creditor,
        "debt_type": debt.debt_type,
        "debt_type_label": DEBT_TYPES.get(debt.debt_type, debt.debt_type),
        "original_amount": str(debt.original_amount),
        "remaining_balance": str(debt.remaining_balance),
        "interest_rate_percent": _optional_text(debt.interest_rate_percent),
        "monthly_payment": _optional_text(debt.monthly_payment),
    }


def _optional_text(value: Optional[Decimal]) -> Optional[str]:
    """Decimal as text without trailing zeros, or None."""
    return None if value is None else format(value.normalize(), "f")


def _optional_decimal(text: str, field_name: str) -> Optional[Decimal]:
    """Parse an optional number; blank means not given."""
    return parse_decimal_value(text, field_name) if text.strip() else None


@tool
def record_debt_tool(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    creditor: str,
    amount: str,
    debt_type: str = "personal",
    interest_rate_percent: str = "",
    monthly_payment: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """บันทึกยอดหนี้ที่ผู้ใช้ติดอยู่ (ไม่ใช่รายจ่าย ห้ามเรียก add_expense ด้วย).

    ใช้เมื่อผู้ใช้บอกว่า "ติดหนี้", "ยืมเงิน...มา", "เป็นหนี้", "มีหนี้บัตร/สินเชื่อ...อยู่".
    ถ้าเจ้าหนี้รายเดิมมีหนี้ค้างอยู่แล้ว ยอดจะบวกเพิ่มเข้าไป.

    Args:
        creditor: เจ้าหนี้ เช่น "เพื่อน", "แม่", "ธนาคารกสิกร", "KTC".
        amount: ยอดหนี้ (บาท) เช่น "2000".
        debt_type: personal (ยืมคนรู้จัก), credit_card, personal_loan, home_loan,
            car_loan, student_loan (กยศ.), other.
        interest_rate_percent: ดอกเบี้ยต่อปี (%) ถ้าผู้ใช้บอก เช่น "18".
        monthly_payment: ยอดผ่อนต่อเดือน ถ้าผู้ใช้บอก.
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict with status "recorded", the debt and added_to_existing, or an error.
    """
    try:
        parsed_amount = parse_decimal_value(amount, "amount")
        rate = _optional_decimal(interest_rate_percent, "interest_rate_percent")
        installment = _optional_decimal(monthly_payment, "monthly_payment")
        with get_tool_session(db_session_factory) as session:
            result = record_debt(
                session, user_id, creditor, parsed_amount, debt_type,
                interest_rate_percent=rate, monthly_payment=installment,
            )  # fmt: skip
            payload = debt_to_dict(result.debt)
            return {"status": "recorded", **payload, "added_to_existing": result.added_to_existing}
    except ValueError as error:
        return {"status": "error", "message": str(error)}


@tool
def pay_debt(
    amount: str,
    creditor: str = "",
    description: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """บันทึกการจ่ายหนี้: เป็นรายจ่ายหมวด debt และหักยอดหนี้ที่บันทึกไว้ (ใช้แทน add_expense).

    ใช้เมื่อผู้ใช้ "ใช้หนี้", "คืนเงิน...", "จ่ายบัตรเครดิต", "จ่ายค่างวด/ผ่อน...".
    ถ้าไม่มียอดหนี้ของเจ้าหนี้นี้บันทึกไว้ (status paid_no_debt_on_file)
    ให้ถามสั้น ๆ ครั้งเดียวว่ายังเหลือหนี้อีกไหม ถ้าเหลือให้ใช้ record_debt_tool.

    Args:
        amount: จำนวนที่จ่าย (บาท) เช่น "5000".
        creditor: เจ้าหนี้หรือประเภทหนี้ เช่น "เพื่อน", "KTC", "บัตรเครดิต".
        description: คำอธิบายของรายจ่าย เช่น "ใช้หนี้เพื่อน".
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict with status "paid" (remaining_balance, paid_off, overpaid),
        "paid_no_debt_on_file" (next_step ASK_REMAINING_DEBT) or an error.
    """
    try:
        paid = parse_decimal_value(amount, "amount")
        with get_tool_session(db_session_factory) as session:
            note = description or f"ชำระหนี้ {creditor}".strip()
            create_expense_transaction(session, user_id, paid, DEBT_CATEGORY, note, date.today())
            debt = find_active_debt(session, user_id, creditor)
            if debt is None:
                return _no_debt_on_file(paid, creditor)
            payment = apply_debt_payment(session, debt, paid)
            return {"status": "paid", "amount": str(paid), **debt_to_dict(debt),
                    "paid_off": payment.paid_off, "overpaid": str(payment.overpaid)}  # fmt: skip
    except ValueError as error:
        return {"status": "error", "message": str(error)}


def _no_debt_on_file(paid: Decimal, creditor: str) -> dict[str, Any]:
    """Expense saved, but no recorded debt to lower."""
    return {
        "status": "paid_no_debt_on_file",
        "amount": str(paid),
        "creditor": creditor,
        "next_step": "ASK_REMAINING_DEBT",
    }


@tool
def list_debts_tool(
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """ดูหนี้ที่ยังค้างอยู่ทั้งหมด พร้อมยอดรวมและยอดผ่อนต่อเดือนรวม.

    Args:
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict with debts, debt_count, total_remaining and total_monthly_payment.
    """
    with get_tool_session(db_session_factory) as session:
        summary = summarize_debts(session, user_id)
        return {
            "debts": [debt_to_dict(debt) for debt in summary.debts],
            "debt_count": summary.debt_count,
            "total_remaining": str(summary.total_remaining),
            "total_monthly_payment": str(summary.total_monthly_payment),
        }


@tool
def update_debt_balance(
    creditor: str,
    remaining_balance: str,
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """แก้ยอดหนี้คงเหลือ (เช่น จดผิด หรือผู้ใช้บอกว่าเหลือเท่าไร) 0 = ปิดหนี้ (ไม่สร้างรายจ่าย).

    Args:
        creditor: เจ้าหนี้หรือประเภทหนี้ เช่น "เพื่อน".
        remaining_balance: ยอดที่ยังเหลือจริง (บาท).
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict with status "updated" and the debt, "not_found", or an error.
    """
    try:
        balance = parse_decimal_value(remaining_balance, "remaining_balance")
        with get_tool_session(db_session_factory) as session:
            debt = set_debt_balance(session, user_id, creditor, balance)
            if debt is None:
                return {"status": "not_found", "creditor": creditor}
            return {"status": "updated", **debt_to_dict(debt), "paid_off": debt.is_paid_off}
    except ValueError as error:
        return {"status": "error", "message": str(error)}


DEBT_TOOLS = [record_debt_tool, pay_debt, list_debts_tool, update_debt_balance]
