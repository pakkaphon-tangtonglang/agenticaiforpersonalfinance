"""LangGraph tools for monthly recurring income and expenses."""

from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.session_helper import get_tool_session
from finance_ai.agents.tax_tools import parse_decimal_value
from finance_ai.database.models.recurring_entry import RecurringEntry
from finance_ai.tools.recurring_service import (
    bangkok_today,
    cancel_recurring_entry,
    create_recurring_entry,
    list_recurring_entries,
    post_due_entries,
)


def recurring_entry_to_dict(entry: RecurringEntry) -> dict[str, Any]:
    """Plain values for the LLM and the API.

    Args:
        entry: Saved recurring entry.

    Returns:
        Dict with type, category, amount (string), day and description.
    """
    return {
        "entry_id": entry.id,
        "entry_type": entry.transaction_type,
        "category": entry.category,
        "amount": str(entry.amount),
        "day_of_month": entry.day_of_month,
        "description": entry.description,
    }


@tool
def add_recurring_entry(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    entry_type: str,
    amount: str,
    category: str,
    day_of_month: str,
    description: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """บันทึกรายรับหรือรายจ่ายที่เกิดซ้ำทุกเดือน (ระบบจะบันทึกให้เองทุกเดือนในวันนั้น).

    ใช้เมื่อผู้ใช้บอกว่า "ทุกเดือน", "เดือนละ", "เข้าวันที่ X ทุกเดือน", "ผ่อนเดือนละ".
    ต้องมีจำนวนเงินและวันที่ของเดือน ถ้าไม่มีให้ถามก่อน ห้ามเดา.
    เริ่มนับจากเดือนนี้: ถ้าวันนี้ตรงหรือเลยวันนั้นแล้ว จะบันทึกของเดือนนี้ให้ทันที
    (posted_now = 1) ถ้ายังไม่ถึง จะบันทึกเมื่อถึงวันนั้น.

    Args:
        entry_type: "income" (รายรับ) หรือ "expense" (รายจ่าย).
        amount: จำนวนเงินต่อเดือน (บาท) เช่น "8000".
        category: รายรับ: salary, freelance, rental, investment, other;
            รายจ่าย: food, transport, housing, utilities, health, education,
            shopping, entertainment, debt, investment, other.
        day_of_month: วันที่ของเดือน 1-31 เช่น "3".
        description: คำอธิบาย เช่น "ค่าเช่าบ้าน".
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict with status "created", the entry and posted_now (รายการที่บันทึกทันที),
        or an error.
    """
    try:
        parsed_amount = parse_decimal_value(amount, "amount")
        day = int(day_of_month.strip())
        with get_tool_session(db_session_factory) as session:
            entry = create_recurring_entry(
                session, user_id, entry_type, category, parsed_amount, day,
                description, bangkok_today().replace(day=1),
            )  # fmt: skip
            posted_now = post_due_entries(session, bangkok_today(), user_id)
            return {"status": "created", **recurring_entry_to_dict(entry), "posted_now": posted_now}
    except ValueError as error:
        return {"status": "error", "message": str(error)}


@tool
def list_recurring_entries_tool(
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """ดูรายการรับ/จ่ายประจำทุกเดือนที่ตั้งไว้.

    Args:
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict with entries (list) and count.
    """
    with get_tool_session(db_session_factory) as session:
        entries = [recurring_entry_to_dict(e) for e in list_recurring_entries(session, user_id)]
    return {"entries": entries, "count": len(entries)}


@tool
def cancel_recurring_entry_tool(
    search_term: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """ยกเลิกรายการรับ/จ่ายประจำทุกเดือน (รายการที่บันทึกไปแล้วยังอยู่).

    Args:
        search_term: คำจากคำอธิบายหรือหมวด เช่น "เช่า", "ผ่อนรถ" ว่าง = รายการล่าสุด.
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict with status "cancelled" and the entry, or "not_found".
    """
    with get_tool_session(db_session_factory) as session:
        entry = cancel_recurring_entry(session, user_id, search_term)
        if entry is None:
            return {"status": "not_found", "search_term": search_term}
        return {"status": "cancelled", **recurring_entry_to_dict(entry)}


RECURRING_TOOLS = [add_recurring_entry, list_recurring_entries_tool, cancel_recurring_entry_tool]
