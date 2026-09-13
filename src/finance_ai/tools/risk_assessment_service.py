"""Scoring service for the SEC suitability (risk assessment) questionnaire.

Pure functions only: validate answers, compute the total score, map the score
to a risk level, and fetch the example asset-allocation row for a level.
"""

from typing import Any

from finance_ai.tools.risk_assessment_constants import (
    ALLOCATION_COLUMNS,
    ASSET_ALLOCATION_BY_LEVEL,
    RISK_QUESTIONS,
    RISK_LEVELS,
    SCORED_QUESTION_IDS,
)


def _get_question(question_id: int) -> dict[str, Any] | None:
    """Return the question dict for a question id, or None if unknown.

    Args:
        question_id: Question number (1-12).

    Returns:
        The question dict or None when the id is unknown.
    """
    return next((question for question in RISK_QUESTIONS if question["id"] == question_id), None)


def _parse_question_key(key: Any) -> int:
    """Convert an answers-dict key to a known question id.

    Args:
        key: Dict key from the answers mapping (expected string digits).

    Returns:
        The recognized question id.

    Raises:
        ValueError: If the key does not match any known question id.
    """
    try:
        question_id = int(str(key))
    except (TypeError, ValueError):
        question_id = -1
    if _get_question(question_id) is None:
        raise ValueError(f"ไม่รู้จักคำถามข้อ {key} (Unknown question id: {key})")
    return question_id


def _validate_single_choice_answer(question: dict[str, Any], answer: Any) -> None:
    """Validate one single-choice answer against its question's option keys.

    Args:
        question: The question dict being answered.
        answer: The submitted choice key.

    Raises:
        ValueError: If the choice key is not one of the question's options.
    """
    valid_keys = {option[0] for option in question["options"]}
    if not isinstance(answer, str) or answer not in valid_keys:
        question_id = question["id"]
        raise ValueError(
            f"คำถามข้อ {question_id}: ตัวเลือก '{answer}' ไม่ถูกต้อง "
            f"(Invalid choice for question {question_id}: allowed {sorted(valid_keys)})"
        )


def _validate_multi_select_answer(question: dict[str, Any], answer: Any) -> None:
    """Validate a multi-select answer (question 4): non-empty list of valid keys.

    Args:
        question: The multi-select question dict.
        answer: The submitted list of choice keys.

    Raises:
        ValueError: If the answer is not a non-empty list or contains invalid keys.
    """
    if not isinstance(answer, list) or not answer:
        raise ValueError(
            "คำถามข้อ 4 ต้องเป็นรายการตัวเลือกที่ไม่ว่างเปล่า "
            "(Question 4 requires a non-empty list of choice keys)"
        )
    valid_keys = {option[0] for option in question["options"]}
    for choice_key in answer:
        if not isinstance(choice_key, str) or choice_key not in valid_keys:
            raise ValueError(
                f"คำถามข้อ 4: ตัวเลือก '{choice_key}' ไม่ถูกต้อง "
                f"(Invalid choice for question 4: allowed {sorted(valid_keys)})"
            )


def _require_scored_answers(answers: dict[str, Any]) -> None:
    """Ensure every scored question (1-10) has an answer.

    Args:
        answers: The submitted answers.

    Raises:
        ValueError: Naming the first missing scored question.
    """
    for question_id in SCORED_QUESTION_IDS:
        if str(question_id) not in answers:
            raise ValueError(
                f"คำถามข้อ {question_id} หายไป (Missing answer for question {question_id})"
            )


def validate_answers(answers: dict[str, Any]) -> None:
    """Validate a full answers mapping for the questionnaire.

    Q1-10 are required with single choice keys (ก/ข/ค/ง); Q4 must be a
    non-empty list of choice keys; Q11/12 are optional (ก/ข/ค when present).

    Args:
        answers: Mapping of question id (as string) to the chosen key(s).

    Raises:
        ValueError: With a Thai+English message naming the offending question.

    Example:
        >>> validate_answers({"1": "ง", "2": "ง", "3": "ง", "4": ["ก", "ง"],
        ...                   "5": "ง", "6": "ง", "7": "ง", "8": "ง", "9": "ง", "10": "ง"})
    """
    for answer_key, answer in answers.items():
        question = _get_question(_parse_question_key(answer_key))
        if question is None:  # pragma: no cover - _parse_question_key guarantees a hit
            continue
        if question["multi_select"]:
            _validate_multi_select_answer(question, answer)
        else:
            _validate_single_choice_answer(question, answer)
    _require_scored_answers(answers)


def calculate_total_score(answers: dict[str, Any]) -> int:
    """Sum the points of scored questions Q1-10.

    Question 4 (multi-select) contributes the MAX of its selected points;
    supplementary questions 11-12 are ignored.

    Args:
        answers: Mapping of question id (as string) to the chosen key(s).

    Returns:
        Total score between 10 and 40.

    Raises:
        ValueError: If the answers are invalid (missing/unknown/ill-formed).

    Example:
        >>> calculate_total_score({"1": "ง", ..., "4": ["ก", "ง"], ...})
        40
    """
    validate_answers(answers)
    total = 0
    for question in RISK_QUESTIONS:
        if not question["scored"]:
            continue
        points_by_key = {option[0]: option[2] for option in question["options"]}
        answer = answers[str(question["id"])]
        if question["multi_select"]:
            total += max(points_by_key[choice_key] for choice_key in answer)
        else:
            total += points_by_key[answer]
    return total


def determine_risk_level(total_score: int) -> tuple[int, str]:
    """Map a total score to (risk level, Thai investor category).

    Args:
        total_score: Sum of scored question points (10-40).

    Returns:
        Tuple of (level 1-5, Thai category name).

    Raises:
        ValueError: If the score is outside the 10-40 range.

    Example:
        >>> determine_risk_level(37)
        (5, 'เสี่ยงสูงมาก')
    """
    for min_score, max_score, level, category in RISK_LEVELS:
        if min_score <= total_score <= max_score:
            return level, category
    raise ValueError(
        f"คะแนน {total_score} อยู่นอกช่วงที่ยอมรับ 10-40 "
        f"(Score {total_score} outside allowed range 10-40)"
    )


def get_allocation(risk_level: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the example asset allocation row for a risk level.

    Args:
        risk_level: Risk level 1-5 from determine_risk_level.

    Returns:
        Tuple of (column labels, percentage strings) for display.

    Raises:
        ValueError: If the risk level is not 1-5.

    Example:
        >>> get_allocation(1)[1]
        ('>60%', '<20%', '<10%', '<5%', '<5%')
    """
    if risk_level not in ASSET_ALLOCATION_BY_LEVEL:
        raise ValueError(
            f"ระดับความเสี่ยง {risk_level} ไม่ถูกต้อง ต้องเป็น 1-5 "
            f"(Invalid risk level {risk_level}, must be 1-5)"
        )
    return ALLOCATION_COLUMNS, ASSET_ALLOCATION_BY_LEVEL[risk_level]
