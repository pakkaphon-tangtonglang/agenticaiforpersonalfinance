"""LangGraph tools that record tax deductions from other agents' conversations.

When the user mentions a payment that is tax deductible (life insurance,
RMF/SSF, donations...), the Expense, Asset Monitoring and Tax agents call
these tools right after recording the payment, so the deduction is counted
without the user asking. The Tax Agent then calculates from saved records.
"""

from datetime import date
from decimal import Decimal
from typing import Annotated, Any

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from finance_ai.agents.session_helper import get_tool_session
from finance_ai.agents.tax_tools import parse_decimal_value
from finance_ai.tools.deduction_service import (
    DeductionRecordResult,
    calculate_yearly_social_security,
    has_deduction_for_year,
    record_deduction,
    remove_latest_deduction,
    validate_claimable_deduction_type,
)

SOCIAL_SECURITY_TYPE: str = "social_security"


def _resolve_tax_year(tax_year: str) -> int:
    """Tax year as int; empty means the current year."""
    return int(tax_year) if tax_year.strip() else date.today().year


def _error(error: ValueError) -> dict[str, Any]:
    """Error payload the agent can explain to the user."""
    return {"status": "error", "message": str(error)}


def _money(amount: Decimal | None) -> str | None:
    """Money with 2 decimals, or None."""
    return None if amount is None else str(amount.quantize(Decimal("0.01")))


def deduction_result_to_dict(result: DeductionRecordResult) -> dict[str, Any]:
    """Plain values for the LLM (money as strings).

    Args:
        result: Service result.

    Returns:
        Dict with status, totals and remaining room.
    """
    cap = result.percentage_of_income_cap
    return {
        "status": result.status,
        "deduction_type": result.deduction_type,
        "amount": _money(result.amount),
        "tax_year": result.tax_year,
        "year_total": _money(result.year_total),
        "maximum_allowed": _money(result.maximum_allowed),
        "remaining_room": _money(result.remaining_room),
        "exceeds_limit": result.exceeds_limit,
        "percentage_of_income_cap": None if cap is None else str(cap),
    }


@tool
def record_tax_deduction_cross(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    deduction_type: str,
    amount: str,
    description: str = "",
    tax_year: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """บันทึกค่าลดหย่อนภาษีจากรายจ่ายที่ผู้ใช้จ่ายจริง เพื่อให้นำไปคิดภาษีอัตโนมัติ.

    เรียกทันทีโดยไม่ต้องถามผู้ใช้ เมื่อผู้ใช้บอกว่าจ่าย/ซื้อสิ่งที่ลดหย่อนได้แน่ชัด:
    life_insurance (ประกันชีวิต), health_insurance (ประกันสุขภาพตัวเอง),
    rmf, ssf, provident_fund (กองทุนสำรองเลี้ยงชีพ), mortgage_interest (ดอกเบี้ยบ้าน),
    donations (เงินบริจาค), social_security (ประกันสังคม).
    ถ้าไม่แน่ใจว่าเป็นประเภทไหน (เช่น "จ่ายประกัน" เฉยๆ อาจเป็นประกันรถ) ให้ถามก่อน.
    status "duplicate" = เคยบันทึกยอดนี้แล้ว ไม่ได้บันทึกซ้ำ.

    Args:
        deduction_type: ประเภทค่าลดหย่อน เช่น "life_insurance".
        amount: จำนวนเงินที่จ่าย (บาท) เช่น "20000".
        description: คำอธิบาย เช่น "ประกันชีวิต AIA".
        tax_year: ปีภาษี ค.ศ. เช่น "2026" ค่าเริ่มต้น = ปีนี้.
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict with status, year_total, maximum_allowed and remaining_room
        (ยังลดหย่อนได้อีกเท่าไร), or an error.
    """
    try:
        parsed_amount = parse_decimal_value(amount, "amount")
        with get_tool_session(db_session_factory) as session:
            result = record_deduction(
                session, user_id, deduction_type, parsed_amount,
                _resolve_tax_year(tax_year), description,
            )  # fmt: skip
    except ValueError as error:
        return _error(error)
    return deduction_result_to_dict(result)


@tool
def record_social_security_cross(
    monthly_salary: str,
    tax_year: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """บันทึกค่าลดหย่อนประกันสังคมทั้งปีจากเงินเดือน (5% ของเงินเดือนไม่เกิน 15,000 x 12).

    เรียกเฉพาะหลังผู้ใช้ยืนยันแล้วว่าเป็นพนักงานที่มีประกันสังคม (มาตรา 33)
    ห้ามเรียกถ้าผู้ใช้เป็นฟรีแลนซ์ หรือยังไม่ได้ตอบ. บันทึกได้ปีละครั้ง.

    Args:
        monthly_salary: เงินเดือนต่อเดือน (บาท) เช่น "50000".
        tax_year: ปีภาษี ค.ศ. ค่าเริ่มต้น = ปีนี้.
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict like record_tax_deduction_cross, status "already_recorded"
        when this year's social security is saved already, or an error.
    """
    try:
        yearly = calculate_yearly_social_security(
            parse_decimal_value(monthly_salary, "monthly_salary")
        )
        year = _resolve_tax_year(tax_year)
        with get_tool_session(db_session_factory) as session:
            if has_deduction_for_year(session, user_id, SOCIAL_SECURITY_TYPE, year):
                return {"status": "already_recorded", "tax_year": year}
            result = record_deduction(
                session, user_id, SOCIAL_SECURITY_TYPE, yearly, year, "ประกันสังคม (มาตรา 33)"
            )
    except ValueError as error:
        return _error(error)
    return deduction_result_to_dict(result)


@tool
def undo_tax_deduction_cross(
    deduction_type: str,
    tax_year: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """ยกเลิกค่าลดหย่อนที่บันทึกล่าสุดของประเภทนั้น.

    ใช้เมื่อผู้ใช้บอกว่า "ยกเลิกค่าลดหย่อนเมื่อกี้", "ไม่ต้องเอาไปลดหย่อน", "บันทึกผิด".

    Args:
        deduction_type: ประเภทค่าลดหย่อน เช่น "life_insurance".
        tax_year: ปีภาษี ค.ศ. ค่าเริ่มต้น = ปีนี้.
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Dict with status "removed" and the amount, "not_found", or an error.
    """
    try:
        normalized = validate_claimable_deduction_type(deduction_type)
        year = _resolve_tax_year(tax_year)
        with get_tool_session(db_session_factory) as session:
            removed = remove_latest_deduction(session, user_id, normalized, year)
    except ValueError as error:
        return _error(error)
    if removed is None:
        return {"status": "not_found", "deduction_type": normalized, "tax_year": year}
    return {
        "status": "removed",
        "deduction_type": normalized,
        "amount": _money(removed),
        "tax_year": year,
    }


@tool
def calculate_tax_from_records(
    tax_year: str = "",
    user_id: Annotated[str, InjectedState("user_id")] = "",
    db_session_factory: Annotated[Any, InjectedState("db_session_factory")] = None,
) -> dict[str, Any]:
    """คำนวณภาษีจากข้อมูลที่ผู้ใช้บันทึกไว้ (รายได้ ค่าลดหย่อน ภาษีหัก ณ ที่จ่าย).

    ใช้เมื่อผู้ใช้ถาม "ภาษีปีนี้ของฉันเท่าไร" โดยไม่ได้บอกตัวเลขมาเอง.
    รายได้คือผลรวมรายการที่บันทึกแล้วเท่านั้น (income_record_count รายการ)
    ถ้าดูไม่ครบปี ให้บอกผู้ใช้และถามรายได้ทั้งปี แล้วใช้ calculate_thai_tax
    กับค่าลดหย่อนจาก deductions_used แทน.

    Args:
        tax_year: ปีภาษี ค.ศ. ค่าเริ่มต้น = ปีนี้.
        user_id: UUID ของผู้ใช้ (injected จาก graph state).
        db_session_factory: Session factory (injected จาก graph state).

    Returns:
        Tax result plus deductions_used and income_record_count,
        status "no_income" when no income is recorded, or an error.
    """
    from finance_ai.database.crud.income_crud import IncomeCRUD  # noqa: PLC0415
    from finance_ai.tools.tax_service import calculate_tax_from_saved_records  # noqa: PLC0415

    year = _resolve_tax_year(tax_year)
    try:
        with get_tool_session(db_session_factory) as session:
            income_count = len(IncomeCRUD().get_by_user_and_year(session, user_id, year))
            if income_count == 0:
                return {"status": "no_income", "tax_year": year}
            result, deductions = calculate_tax_from_saved_records(session, user_id, year)
    except ValueError as error:
        return _error(error)
    return {
        "status": "calculated",
        "tax_year": year,
        "income_record_count": income_count,
        "deductions_used": {key: str(value) for key, value in deductions.items()},
        **result.model_dump(mode="json"),
    }
