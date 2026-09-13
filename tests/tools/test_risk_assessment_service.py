"""Tests for the risk assessment scoring service (pure functions)."""

from typing import Any

import pytest

from finance_ai.tools.risk_assessment_service import (
    calculate_total_score,
    determine_risk_level,
    get_allocation,
    validate_answers,
)

SCORED_IDS = range(1, 11)


def answers_all(choice: str) -> dict[str, Any]:
    """Build valid answers for Q1-10 with the same choice (Q4 becomes a list)."""
    return {
        str(question_id): [choice] if question_id == 4 else choice for question_id in SCORED_IDS
    }


def answers_all_but_q4(other_choice: str, q4_choices: list[str]) -> dict[str, Any]:
    """Build answers where Q4 is a multi-select list and the rest use one choice."""
    answers: dict[str, Any] = answers_all(other_choice)
    answers["4"] = q4_choices
    return answers


class TestCalculateTotalScore:
    """Tests for calculate_total_score."""

    def test_all_lowest_choices_score_ten(self) -> None:
        """All-ก answers must score 10 (9 questions x1 + Q4 max=1)."""
        assert calculate_total_score(answers_all("ก")) == 10

    def test_all_highest_choices_score_forty(self) -> None:
        """All-ง answers must score 40 (9 questions x4 + Q4 max=4)."""
        assert calculate_total_score(answers_all("ง")) == 40

    def test_q4_multi_select_uses_max_not_sum(self) -> None:
        """Q4 ['ก','ง'] must count 4 (max), not 5 (sum)."""
        answers = answers_all_but_q4("ก", ["ก", "ง"])
        assert calculate_total_score(answers) == 13

    def test_single_choice_q4_still_counts(self) -> None:
        """Q4 with a single selected key scores that key's points."""
        answers = answers_all_but_q4("ก", ["ง"])
        assert calculate_total_score(answers) == 13

    def test_supplementary_questions_never_affect_score(self) -> None:
        """Q11/Q12 answers must be ignored by scoring."""
        answers = answers_all("ก")
        answers["11"] = "ค"
        answers["12"] = "ข"
        assert calculate_total_score(answers) == 10

    def test_missing_question_raises(self) -> None:
        """A missing scored question raises ValueError naming the question."""
        answers = answers_all("ก")
        del answers["3"]
        with pytest.raises(ValueError, match="3"):
            calculate_total_score(answers)


class TestValidateAnswers:
    """Tests for validate_answers."""

    def test_valid_minimal_answers_pass(self) -> None:
        """Q1-10 only, single choices each, must validate."""
        validate_answers(answers_all("ก"))  # no exception = valid

    def test_valid_with_supplementary_questions(self) -> None:
        """Q11/12 present with valid keys must validate."""
        answers = answers_all("ข")
        answers["11"] = "ก"
        answers["12"] = "ค"
        validate_answers(answers)  # no exception = valid

    def test_missing_question_message_names_the_question(self) -> None:
        """The error message must name the missing question number."""
        answers = answers_all("ก")
        del answers["7"]
        with pytest.raises(ValueError, match="7"):
            validate_answers(answers)

    def test_invalid_single_choice_key_raises(self) -> None:
        """An invalid choice key for a scored question raises ValueError."""
        answers = answers_all("ก")
        answers["5"] = "ฮ"
        with pytest.raises(ValueError, match="5"):
            validate_answers(answers)

    def test_unknown_question_key_raises(self) -> None:
        """An unknown question id like '99' raises ValueError."""
        answers = answers_all("ก")
        answers["99"] = "ก"
        with pytest.raises(ValueError, match="99"):
            validate_answers(answers)

    def test_q4_non_list_raises(self) -> None:
        """Q4 given as a string (not a list) raises ValueError."""
        answers = answers_all("ก")
        answers["4"] = "ง"
        with pytest.raises(ValueError, match="4"):
            validate_answers(answers)

    def test_q4_empty_list_raises(self) -> None:
        """Q4 given as an empty list raises ValueError."""
        answers = answers_all("ก")
        answers["4"] = []
        with pytest.raises(ValueError, match="4"):
            validate_answers(answers)

    def test_q4_invalid_member_raises(self) -> None:
        """Q4 list containing an invalid key raises ValueError."""
        answers = answers_all("ก")
        answers["4"] = ["ก", "ฮ"]
        with pytest.raises(ValueError, match="4"):
            validate_answers(answers)

    def test_invalid_supplementary_choice_raises(self) -> None:
        """Q11/12 present but with a scored-only key like 'ง' raises ValueError."""
        answers = answers_all("ก")
        answers["12"] = "ง"
        with pytest.raises(ValueError, match="12"):
            validate_answers(answers)


class TestDetermineRiskLevel:
    """Tests for the score-to-level mapping."""

    @pytest.mark.parametrize(
        ("total_score", "expected_level", "expected_category"),
        [
            (10, 1, "เสี่ยงต่ำ"),
            (14, 1, "เสี่ยงต่ำ"),
            (15, 2, "เสี่ยงปานกลางค่อนข้างต่ำ"),
            (21, 2, "เสี่ยงปานกลางค่อนข้างต่ำ"),
            (22, 3, "เสี่ยงปานกลางค่อนข้างสูง"),
            (29, 3, "เสี่ยงปานกลางค่อนข้างสูง"),
            (30, 4, "เสี่ยงสูง"),
            (36, 4, "เสี่ยงสูง"),
            (37, 5, "เสี่ยงสูงมาก"),
            (40, 5, "เสี่ยงสูงมาก"),
        ],
    )
    def test_level_boundaries(
        self, total_score: int, expected_level: int, expected_category: str
    ) -> None:
        """Boundary scores map to the correct level and Thai category."""
        level, category = determine_risk_level(total_score)
        assert level == expected_level
        assert category == expected_category

    @pytest.mark.parametrize("total_score", [9, 41, 0, -1, 100])
    def test_out_of_range_score_raises(self, total_score: int) -> None:
        """Scores outside 10..40 raise ValueError."""
        with pytest.raises(ValueError, match=str(total_score)):
            determine_risk_level(total_score)


class TestGetAllocation:
    """Tests for get_allocation."""

    def test_returns_columns_and_row_for_level_one(self) -> None:
        """Level 1 returns the 5 column labels and its 5 percentages."""
        columns, row = get_allocation(1)
        assert len(columns) == 5
        assert row == (">60%", "<20%", "<10%", "<5%", "<5%")

    @pytest.mark.parametrize("level", range(1, 6))
    def test_every_level_has_matching_lengths(self, level: int) -> None:
        """Columns and percentages must be the same length for every level."""
        columns, row = get_allocation(level)
        assert len(columns) == len(row)

    @pytest.mark.parametrize("level", [0, 6, -1])
    def test_invalid_level_raises(self, level: int) -> None:
        """Levels outside 1..5 raise ValueError."""
        with pytest.raises(ValueError, match=str(level)):
            get_allocation(level)
