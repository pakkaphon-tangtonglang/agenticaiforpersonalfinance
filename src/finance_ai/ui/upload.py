"""Streamlit view for uploading bank statements.

Handles file upload, parsing, preview, and bulk import of
transactions into the database.
"""

from typing import Any, Callable

import streamlit as st
from sqlalchemy.orm import Session

from finance_ai.tools.bank_statement_parser import (
    ParsedTransaction,
    parse_bank_statement,
    parse_with_llm,
)


def render_upload_view(
    user_id: str,
    db_session_factory: Callable[[], Session],
    chat_model: Any = None,
) -> None:
    """Render the file upload view.

    Args:
        user_id: UUID of the current user.
        db_session_factory: Callable that creates DB sessions.
        chat_model: LangChain chat model for LLM fallback parsing.
    """
    st.subheader("นำเข้า Bank Statement")
    st.caption("รองรับไฟล์ CSV และ Excel (.xlsx)")

    uploaded_file = st.file_uploader(
        "เลือกไฟล์ Bank Statement",
        type=["csv", "xlsx"],
        key="bank_statement_uploader",
    )

    if uploaded_file is None:
        _show_upload_guide()
        return

    transactions = _parse_uploaded_file(uploaded_file, chat_model)
    if not transactions:
        st.warning("ไม่พบรายการที่สามารถนำเข้าได้ กรุณาตรวจสอบรูปแบบไฟล์")
        return

    _show_preview(transactions)
    _handle_import(transactions, user_id, db_session_factory)


def _show_upload_guide() -> None:
    """Show upload format instructions."""
    st.info(
        "**รูปแบบไฟล์ที่รองรับ:**\n\n"
        "- CSV/Excel ที่มีคอลัมน์: วันที่, รายละเอียด, จำนวนเงิน\n"
        "- รองรับวันที่แบบ พ.ศ. (เช่น 12/03/2569)\n"
        "- จำนวนเงินติดลบ = รายรับ"
    )


def _parse_uploaded_file(
    uploaded_file: Any,
    chat_model: Any = None,
) -> list[ParsedTransaction]:
    """Parse uploaded file into transactions.

    Tries rule-based parsing first. Falls back to LLM
    parsing when rule-based returns no results.

    Args:
        uploaded_file: Streamlit UploadedFile object.
        chat_model: LangChain chat model for LLM fallback.

    Returns:
        List of parsed transactions, empty on error.
    """
    try:
        file_type = _detect_file_type(uploaded_file.name)
        content = uploaded_file.read()
        transactions = parse_bank_statement(content, file_type)
        if transactions:
            return transactions
        return _try_llm_fallback(content, chat_model)
    except Exception as exc:  # noqa: BLE001
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {exc}")
        return []


def _try_llm_fallback(
    content: str | bytes,
    chat_model: Any,
) -> list[ParsedTransaction]:
    """Attempt LLM-based parsing as fallback.

    Args:
        content: Raw file content.
        chat_model: LangChain chat model instance.

    Returns:
        List of parsed transactions, empty on error.
    """
    if chat_model is None:
        return []
    text = content if isinstance(content, str) else content.decode("utf-8-sig", errors="replace")
    st.info("กำลังใช้ AI วิเคราะห์รูปแบบไฟล์...")
    return parse_with_llm(text, chat_model)


def _detect_file_type(filename: str) -> str:
    """Detect file type from filename extension.

    Args:
        filename: Name of the uploaded file.

    Returns:
        'csv' or 'xlsx'.
    """
    if filename.lower().endswith(".xlsx"):
        return "xlsx"
    return "csv"


def _show_preview(transactions: list[ParsedTransaction]) -> None:
    """Show preview table of parsed transactions.

    Args:
        transactions: List of parsed transactions.
    """
    st.success(f"พบ {len(transactions)} รายการ")

    preview_data = [
        {
            "วันที่": str(t.transaction_date),
            "รายละเอียด": t.description,
            "จำนวนเงิน": f"{float(t.amount):,.2f}",
            "หมวดหมู่": t.category,
            "ประเภท": "รายรับ" if t.transaction_type == "income" else "รายจ่าย",
        }
        for t in transactions
    ]
    st.dataframe(preview_data, use_container_width=True)


def _handle_import(
    transactions: list[ParsedTransaction],
    user_id: str,
    db_session_factory: Callable[[], Session],
) -> None:
    """Handle the import button and bulk insert.

    Args:
        transactions: Parsed transactions to import.
        user_id: UUID of the user.
        db_session_factory: Session factory callable.
    """
    if st.button(
        f"นำเข้า {len(transactions)} รายการ",
        type="primary",
        use_container_width=True,
    ):
        count = _bulk_insert(transactions, user_id, db_session_factory)
        st.success(f"นำเข้าสำเร็จ {count} รายการ!")
        st.balloons()
        st.rerun()


def _bulk_insert(
    transactions: list[ParsedTransaction],
    user_id: str,
    db_session_factory: Callable[[], Session],
) -> int:
    """Bulk insert parsed transactions into the database.

    Args:
        transactions: List of transactions to insert.
        user_id: UUID of the user.
        db_session_factory: Session factory callable.

    Returns:
        Number of transactions inserted.
    """
    from finance_ai.database.models.income import Income  # noqa: PLC0415
    from finance_ai.database.models.transaction import (  # noqa: PLC0415
        Transaction,
    )

    session = db_session_factory()
    count = 0

    try:
        for txn in transactions:
            inserted = _insert_single_transaction(
                session,
                txn,
                user_id,
                Transaction,
                Income,
            )
            if inserted:
                count += 1
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return count


def _insert_single_transaction(
    session: Session,
    txn: ParsedTransaction,
    user_id: str,
    transaction_cls: Any,
    income_cls: Any,
) -> bool:
    """Insert a single transaction into the correct table if not duplicate.

    Skips records that already exist (same user, date, description, amount).
    Income goes to both transactions and incomes tables.

    Args:
        session: Database session.
        txn: Parsed transaction to insert.
        user_id: UUID of the user.
        transaction_cls: Transaction model class.
        income_cls: Income model class.

    Returns:
        True if inserted, False if skipped as duplicate.
    """
    from sqlalchemy import select  # noqa: PLC0415

    existing: Any = session.execute(
        select(transaction_cls).where(
            transaction_cls.user_id == user_id,
            transaction_cls.transaction_date == txn.transaction_date,
            transaction_cls.description == txn.description,
            transaction_cls.amount == txn.amount,
        )
    ).first()
    if existing:
        return False

    record = transaction_cls(
        user_id=user_id,
        transaction_type=txn.transaction_type,
        category=txn.category,
        amount=txn.amount,
        description=txn.description,
        transaction_date=txn.transaction_date,
    )
    session.add(record)

    if txn.transaction_type == "income":
        duplicate_income: Any = session.execute(
            select(income_cls).where(
                income_cls.user_id == user_id,
                income_cls.tax_year == txn.transaction_date.year,
                income_cls.description == txn.description,
                income_cls.amount == txn.amount,
            )
        ).first()
        if not duplicate_income:
            session.add(
                income_cls(
                    user_id=user_id,
                    income_type="other",
                    description=txn.description,
                    amount=txn.amount,
                    tax_year=txn.transaction_date.year,
                    pay_period="monthly",
                )
            )
    return True
