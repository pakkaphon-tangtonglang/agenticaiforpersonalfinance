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
)


def render_upload_view(
    user_id: str,
    db_session_factory: Callable[[], Session],
) -> None:
    """Render the file upload view.

    Args:
        user_id: UUID of the current user.
        db_session_factory: Callable that creates DB sessions.
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

    transactions = _parse_uploaded_file(uploaded_file)
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
) -> list[ParsedTransaction]:
    """Parse uploaded file into transactions.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        List of parsed transactions, empty on error.
    """
    try:
        file_type = _detect_file_type(uploaded_file.name)
        content = uploaded_file.read()
        return parse_bank_statement(content, file_type)
    except Exception as exc:  # noqa: BLE001
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {exc}")
        return []


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
    from finance_ai.database.crud.transaction_crud import (  # noqa: PLC0415
        TransactionCRUD,
    )

    crud = TransactionCRUD()
    session = db_session_factory()
    count = 0

    try:
        for txn in transactions:
            crud.create(
                session,
                user_id=user_id,
                transaction_type=txn.transaction_type,
                category=txn.category,
                amount=txn.amount,
                description=txn.description,
                date=txn.transaction_date,
            )
            count += 1
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

    return count
