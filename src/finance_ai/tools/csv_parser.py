"""CSV parser for portfolio import.

Parses CSV content into validated holding dictionaries suitable
for creating InvestmentHolding records via the service layer.
"""

import csv
import io
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from finance_ai.tools.investment_constants import CSV_REQUIRED_COLUMNS


def validate_csv_columns(header_row: list[str]) -> None:
    """Validate that the CSV header contains all required columns.

    Args:
        header_row: List of column names from the CSV header.

    Raises:
        ValueError: If any required column is missing.

    Example:
        >>> validate_csv_columns(["symbol", "asset_type", "name", "quantity",
        ...                       "price_per_unit", "purchase_date"])
    """
    normalized = [col.strip().lower() for col in header_row]
    missing = [col for col in CSV_REQUIRED_COLUMNS if col not in normalized]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Required: {list(CSV_REQUIRED_COLUMNS)}"
        )


def parse_csv_row(
    row: dict[str, str],
    row_number: int,
) -> dict[str, str | Decimal | date]:
    """Parse and validate a single CSV row into a holding dict.

    Args:
        row: Dict from csv.DictReader with string values.
        row_number: Row number for error messages (1-indexed).

    Returns:
        Dict with parsed and validated values.

    Raises:
        ValueError: If any field has an invalid value.

    Example:
        >>> parse_csv_row({"symbol": "PTT.BK", ...}, 1)
    """
    symbol = row.get("symbol", "").strip()
    if not symbol:
        raise ValueError(f"Row {row_number}: symbol cannot be empty.")
    asset_type = row.get("asset_type", "").strip().lower()
    if not asset_type:
        raise ValueError(f"Row {row_number}: asset_type cannot be empty.")
    name = row.get("name", "").strip()
    quantity = _parse_decimal_field(row, "quantity", row_number)
    price_per_unit = _parse_decimal_field(row, "price_per_unit", row_number)
    purchase_date = _parse_date_field(row, "purchase_date", row_number)
    return {
        "symbol": symbol,
        "asset_type": asset_type,
        "name": name,
        "quantity": quantity,
        "price_per_unit": price_per_unit,
        "purchase_date": purchase_date,
    }


def _parse_decimal_field(
    row: dict[str, str],
    field: str,
    row_number: int,
) -> Decimal:
    """Parse a string field to Decimal.

    Args:
        row: CSV row dict.
        field: Field name to parse.
        row_number: Row number for error messages.

    Returns:
        Parsed Decimal value.

    Raises:
        ValueError: If the field is empty or not a valid number.
    """
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"Row {row_number}: {field} cannot be empty.")
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(
            f"Row {row_number}: Cannot parse {field} as number. " f"Received: '{value}'"
        ) from exc


def _parse_date_field(
    row: dict[str, str],
    field: str,
    row_number: int,
) -> date:
    """Parse a string field to date (YYYY-MM-DD).

    Args:
        row: CSV row dict.
        field: Field name to parse.
        row_number: Row number for error messages.

    Returns:
        Parsed date object.

    Raises:
        ValueError: If the field is empty or not a valid date.
    """
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"Row {row_number}: {field} cannot be empty.")
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"Row {row_number}: Cannot parse {field} as date (YYYY-MM-DD). " f"Received: '{value}'"
        ) from exc


def parse_portfolio_csv(
    csv_content: str,
) -> list[dict[str, str | Decimal | date]]:
    """Parse a complete CSV string into a list of holding dicts.

    Args:
        csv_content: Raw CSV string with header row.

    Returns:
        List of validated holding dicts ready for service layer.

    Raises:
        ValueError: If CSV is empty, missing columns, or has invalid rows.

    Example:
        >>> holdings = parse_portfolio_csv(
        ...     "symbol,asset_type,name,quantity,price_per_unit,purchase_date\\n"
        ...     "PTT.BK,stock,PTT,100,35.50,2025-01-15"
        ... )
    """
    reader = csv.DictReader(io.StringIO(csv_content.strip()))
    if reader.fieldnames is None:
        raise ValueError("CSV content is empty or has no header row.")
    validate_csv_columns(list(reader.fieldnames))
    rows = list(reader)
    if not rows:
        raise ValueError("CSV has a header but no data rows.")
    return [parse_csv_row(row, i + 1) for i, row in enumerate(rows)]
