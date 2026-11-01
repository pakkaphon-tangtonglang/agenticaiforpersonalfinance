"""Clean up values the way people type them before they are validated.

Pure functions only: they fix formatting that has one obvious meaning
(Thai digits, "$AAPL", "35.-", Buddhist-era years, "8:30 pm"). Anything
with more than one meaning is left alone, so the validator rejects it and
the agent asks the user.
"""

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

# Units and separators people type around numbers: "1,000", "35 บาท", "฿35.50"
_NUMBER_NOISE = re.compile(r"[,\s]|฿|บาท|THB|หุ้น|หน่วย", re.IGNORECASE)
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_SYMBOL_PREFIXES = re.compile(r"^(?:\$|SET:|SET\s+|BKK:)", re.IGNORECASE)
_BUDDHIST_ERA_OFFSET = 543
_FIRST_BUDDHIST_YEAR = 2400  # years above this are Buddhist era (พ.ศ.)
_TWELVE_HOUR = re.compile(r"^\s*(\d{1,2})(?:\s*[:.]\s*(\d{2}))?\s*(am|pm)\s*$", re.IGNORECASE)
_NOON = 12
_ISO_DATE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_DAY_FIRST_DATE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$")


def normalize_digits(text: str) -> str:
    """Replace Thai digits with Arabic digits.

    Example:
        >>> normalize_digits("๑๐๐ หุ้น")
        '100 หุ้น'
    """
    return text.translate(_THAI_DIGITS)


def clean_symbol_text(text: str) -> str:
    """Strip ticker decorations: "$AAPL", "SET:PTT", " ptt.bk " → "AAPL", "PTT", "ptt.bk".

    Case is kept so a later search can still match names.

    Example:
        >>> clean_symbol_text(" SET:PTT ")
        'PTT'
    """
    stripped = normalize_digits(text).strip()
    return _SYMBOL_PREFIXES.sub("", stripped).strip()


def clean_number_text(text: str) -> str:
    """Turn Thai digits to Arabic and drop the Thai "no satang" mark ("35.-" → "35").

    Example:
        >>> clean_number_text("๓๕.-")
        '35'
    """
    return normalize_digits(text).strip().removesuffix(".-")


def parse_decimal_text(value: Any, field_name: str) -> Decimal:
    """Parse a typed number ("1,000", "35 บาท", "฿35.50", "๓๕.-") into a finite Decimal.

    Args:
        value: The value to parse (str, int, float, Decimal).
        field_name: Name of the field for error messages.

    Returns:
        Parsed Decimal value.

    Raises:
        ValueError: If the value is not a finite number ("nan" and "inf" are rejected).

    Example:
        >>> parse_decimal_text("1,200,000 บาท", "gross_income")
        Decimal('1200000')
    """
    cleaned = _NUMBER_NOISE.sub("", clean_number_text(str(value)))
    try:
        parsed = Decimal(cleaned)
    except (InvalidOperation, ValueError) as exc:
        raise _number_error(field_name, value) from exc
    if not parsed.is_finite():
        raise _number_error(field_name, value)
    return parsed


def _number_error(field_name: str, value: Any) -> ValueError:
    """Error for a value that is not a usable number."""
    return ValueError(f"Cannot parse {field_name} as Decimal. Received: {value}")


def twelve_hour_to_24(text: str) -> str:
    """Convert "8:30 pm" / "12 am" to "20:30" / "00:00"; other text is returned unchanged.

    Example:
        >>> twelve_hour_to_24("8.30 PM")
        '20:30'
    """
    match = _TWELVE_HOUR.match(normalize_digits(text))
    if match is None or not 1 <= int(match.group(1)) <= _NOON:
        return normalize_digits(text)  # "13 pm" stays invalid and is rejected later
    hour, minute = int(match.group(1)) % _NOON, int(match.group(2) or 0)
    if match.group(3).lower() == "pm":
        hour += _NOON
    return f"{hour:02d}:{minute:02d}"


def parse_flexible_date(text: str) -> date:
    """Parse YYYY-MM-DD, DD/MM/YYYY or DD-MM-YYYY, in either AD or Buddhist era.

    The Buddhist-era year is converted before the date is built, so leap
    days work ("29/02/2567" is 29 Feb 2024).

    Args:
        text: Date as typed (Thai digits allowed).

    Returns:
        The date in the Gregorian calendar.

    Raises:
        ValueError: When the text is not a real date in a supported format.

    Example:
        >>> parse_flexible_date("1/10/2569")
        datetime.date(2026, 10, 1)
    """
    year, month, day = _date_parts(normalize_digits(text).strip())
    if year > _FIRST_BUDDHIST_YEAR:
        year -= _BUDDHIST_ERA_OFFSET
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise _date_error(text) from exc


def _date_parts(cleaned: str) -> tuple[int, int, int]:
    """Split a typed date into (year, month, day)."""
    iso = _ISO_DATE.match(cleaned)
    if iso:
        return int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
    day_first = _DAY_FIRST_DATE.match(cleaned)
    if day_first:
        return int(day_first.group(3)), int(day_first.group(2)), int(day_first.group(1))
    raise _date_error(cleaned)


def _date_error(text: str) -> ValueError:
    """Error for a date that cannot be read."""
    return ValueError(f"Unrecognized date. Received: {text!r}. Use YYYY-MM-DD, e.g. 2026-10-01")


# A reply that is only an hour ("8", "๘", "8 ครับ", "ตอน 8") with no time word.
_POLITE_ENDINGS = re.compile(r"(?:ครับ|คับ|ค่ะ|คะ|ค่า|นะ|จ้า|จ้ะ|จ่ะ)+$")
_BARE_HOUR = re.compile(r"^(?:ตอน\s*)?(\d{1,2})$")


def ambiguous_hour(text: str) -> int | None:
    """The hour when a reply is just "8"-style (1-12, could be morning or evening).

    Times with a word or minutes are clear: "8 โมง", "2 ทุ่ม", "20", "8:30", "8 pm".

    Args:
        text: The user's message as typed.

    Returns:
        The hour (1-12) when ambiguous, else None.

    Example:
        >>> ambiguous_hour("8 ครับ")
        8
        >>> ambiguous_hour("2 ทุ่ม") is None
        True
    """
    cleaned = normalize_digits(text).strip().rstrip(".!?").strip()
    cleaned = _POLITE_ENDINGS.sub("", cleaned).strip()
    match = _BARE_HOUR.match(cleaned)
    if match is None:
        return None
    hour = int(match.group(1))
    return hour if 1 <= hour <= _NOON else None
