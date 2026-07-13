"""Tests for finance_ai.tools.bank_statement_parser."""

from datetime import date
from decimal import Decimal

import pytest

from finance_ai.tools.bank_statement_parser import (
    ParsedTransaction,
    classify_expense_category,
    normalize_thai_date,
    parse_bank_statement,
    parse_generic_csv,
)


class TestNormalizeThaiDate:
    """Tests for normalize_thai_date."""

    def test_buddhist_era_slash(self) -> None:
        """Buddhist era date DD/MM/BBBB converts correctly."""
        result = normalize_thai_date("12/03/2569")
        assert result == date(2026, 3, 12)

    def test_christian_era_slash(self) -> None:
        """Christian era date DD/MM/YYYY works."""
        result = normalize_thai_date("15/06/2026")
        assert result == date(2026, 6, 15)

    def test_iso_format_dash(self) -> None:
        """ISO format YYYY-MM-DD works."""
        result = normalize_thai_date("2026-03-12")
        assert result == date(2026, 3, 12)

    def test_buddhist_era_dash(self) -> None:
        """Buddhist era with dash DD-MM-BBBB converts."""
        result = normalize_thai_date("01-01-2569")
        assert result == date(2026, 1, 1)

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError):
            normalize_thai_date("not a date")

    def test_strips_whitespace(self) -> None:
        """Whitespace around date is stripped."""
        result = normalize_thai_date("  12/03/2569  ")
        assert result == date(2026, 3, 12)


class TestClassifyExpenseCategory:
    """Tests for classify_expense_category."""

    def test_food_thai(self) -> None:
        """Thai food keyword is classified as food."""
        assert classify_expense_category("ร้านอาหาร สุกี้") == "food"

    def test_transport_english(self) -> None:
        """English transport keyword works."""
        assert classify_expense_category("Grab ride") == "transport"

    def test_utilities(self) -> None:
        """Utility keyword is classified correctly."""
        assert classify_expense_category("ค่าไฟฟ้า เดือน มี.ค.") == "utilities"

    def test_unknown_returns_other(self) -> None:
        """Unknown description returns 'other'."""
        assert classify_expense_category("random text xyz") == "other"

    def test_case_insensitive(self) -> None:
        """Matching is case insensitive."""
        assert classify_expense_category("NETFLIX subscription") == "entertainment"


class TestParsedTransaction:
    """Tests for ParsedTransaction model."""

    def test_defaults(self) -> None:
        """Default values are correct."""
        txn = ParsedTransaction(
            transaction_date=date(2026, 3, 1),
            description="test",
            amount=Decimal("100"),
        )
        assert txn.category == "other"
        assert txn.transaction_type == "expense"


class TestParseGenericCsv:
    """Tests for parse_generic_csv."""

    def test_basic_csv(self) -> None:
        """Basic CSV with standard headers parses correctly."""
        csv_content = "date,description,amount\n" "12/03/2569,ค่าอาหาร,350\n" "13/03/2569,BTS,44\n"
        result = parse_generic_csv(csv_content)
        assert len(result) == 2
        assert result[0].transaction_date == date(2026, 3, 12)
        assert result[0].amount == Decimal("350")
        assert result[0].category == "food"

    def test_thai_headers(self) -> None:
        """Thai column headers are auto-detected."""
        csv_content = "วันที่,รายละเอียด,จำนวนเงิน\n" "01/01/2569,ค่าน้ำ,150\n"
        result = parse_generic_csv(csv_content)
        assert len(result) == 1
        assert result[0].amount == Decimal("150")

    def test_negative_amount_is_income(self) -> None:
        """Negative amount is classified as income."""
        csv_content = "date,description,amount\n01/01/2026,เงินเดือน,-50000\n"
        result = parse_generic_csv(csv_content)
        assert result[0].transaction_type == "income"
        assert result[0].amount == Decimal("50000")

    def test_comma_in_amount(self) -> None:
        """Commas in amount are handled."""
        csv_content = 'date,description,amount\n01/01/2026,test,"1,234.56"\n'
        result = parse_generic_csv(csv_content)
        assert result[0].amount == Decimal("1234.56")

    def test_empty_csv(self) -> None:
        """Empty CSV returns empty list."""
        result = parse_generic_csv("date,description,amount\n")
        assert result == []

    def test_invalid_rows_skipped(self) -> None:
        """Rows that can't be parsed are skipped."""
        csv_content = "date,description,amount\n" "not-date,test,abc\n" "01/01/2026,valid,100\n"
        result = parse_generic_csv(csv_content)
        assert len(result) == 1


class TestParseBankStatement:
    """Tests for parse_bank_statement dispatcher."""

    def test_csv_dispatch(self) -> None:
        """CSV file type dispatches to parse_generic_csv."""
        csv_content = "date,description,amount\n01/01/2026,test,100\n"
        result = parse_bank_statement(csv_content, "csv")
        assert len(result) == 1

    def test_csv_bytes_decoded(self) -> None:
        """Bytes content is decoded for CSV."""
        csv_bytes = b"date,description,amount\n01/01/2026,test,100\n"
        result = parse_bank_statement(csv_bytes, "csv")
        assert len(result) == 1

    def test_unsupported_type_raises(self) -> None:
        """Unsupported file type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            parse_bank_statement("data", "pdf")

    def test_excel_non_bytes_raises(self) -> None:
        """Excel with string content raises ValueError."""
        with pytest.raises(ValueError, match="Excel content must be bytes"):
            parse_bank_statement("not bytes", "xlsx")
