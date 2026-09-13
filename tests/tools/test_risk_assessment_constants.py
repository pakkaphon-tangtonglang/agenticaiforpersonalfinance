"""Tests for SEC risk assessment constants (internal consistency)."""

from finance_ai.tools.risk_assessment_constants import (
    ALLOCATION_COLUMNS,
    ALLOCATION_FOOTNOTE,
    ASSET_ALLOCATION_BY_LEVEL,
    MULTI_SELECT_QUESTION_IDS,
    RISK_QUESTIONS,
    RISK_LEVELS,
    SCORED_QUESTION_IDS,
)


class TestRiskQuestions:
    """Tests for the RISK_QUESTIONS structure."""

    def test_has_twelve_questions(self) -> None:
        """The questionnaire must contain exactly 12 questions."""
        assert len(RISK_QUESTIONS) == 12

    def test_question_ids_are_one_to_twelve(self) -> None:
        """Question ids must be 1..12 in order."""
        ids = [question["id"] for question in RISK_QUESTIONS]
        assert ids == list(range(1, 13))

    def test_every_question_has_thai_text(self) -> None:
        """Every question must have non-empty Thai question text."""
        for question in RISK_QUESTIONS:
            assert isinstance(question["text"], str)
            assert len(question["text"]) > 0

    def test_scored_questions_have_four_options_points_one_to_four(self) -> None:
        """Scored questions must have options กขคง with points 1..4 in order."""
        thai_keys = ["ก", "ข", "ค", "ง"]
        for question in RISK_QUESTIONS:
            if not question["scored"]:
                continue
            options = question["options"]
            assert len(options) == 4, f"question {question['id']} must have 4 options"
            assert [option[0] for option in options] == thai_keys
            assert [option[2] for option in options] == [1, 2, 3, 4]

    def test_supplementary_questions_have_three_options_zero_points(self) -> None:
        """Q11 and Q12 (not scored) must have 3 options with 0 points each."""
        for question in RISK_QUESTIONS:
            if question["scored"]:
                continue
            assert question["id"] in (11, 12)
            options = question["options"]
            assert len(options) == 3
            assert [option[0] for option in options] == ["ก", "ข", "ค"]
            assert [option[2] for option in options] == [0, 0, 0]

    def test_only_q4_is_multi_select(self) -> None:
        """Only question 4 is multi-select."""
        multi = [question["id"] for question in RISK_QUESTIONS if question["multi_select"]]
        assert multi == [4]
        assert MULTI_SELECT_QUESTION_IDS == (4,)

    def test_scored_question_ids_are_one_to_ten(self) -> None:
        """SCORED_QUESTION_IDS must be exactly 1..10."""
        assert SCORED_QUESTION_IDS == tuple(range(1, 11))


class TestRiskLevels:
    """Tests for the score-to-level mapping."""

    def test_five_levels_with_thai_names(self) -> None:
        """Levels must be 1..5 with non-empty Thai category names."""
        levels = [entry[2] for entry in RISK_LEVELS]
        assert levels == [1, 2, 3, 4, 5]
        for entry in RISK_LEVELS:
            assert len(entry[3]) > 0

    def test_ranges_cover_10_to_40_without_gaps_or_overlaps(self) -> None:
        """Level ranges must tile 10..40 exactly."""
        covered: set[int] = set()
        for min_inclusive, max_inclusive, _level, _name in RISK_LEVELS:
            span = set(range(min_inclusive, max_inclusive + 1))
            assert not covered & span, "overlapping level ranges"
            covered |= span
        assert covered == set(range(10, 41)), "level ranges must cover 10..40"


class TestAssetAllocation:
    """Tests for the asset allocation example table."""

    def test_has_five_columns(self) -> None:
        """The allocation table must have exactly 5 column labels."""
        assert len(ALLOCATION_COLUMNS) == 5

    def test_all_five_levels_have_five_percentages(self) -> None:
        """Every level 1..5 must map to exactly 5 percentage strings."""
        for level in range(1, 6):
            row = ASSET_ALLOCATION_BY_LEVEL[level]
            assert len(row) == 5, f"level {level} must have 5 allocation values"
            for value in row:
                assert isinstance(value, str)
                assert len(value) > 0

    def test_no_extra_levels(self) -> None:
        """The allocation mapping must contain exactly levels 1..5."""
        assert sorted(ASSET_ALLOCATION_BY_LEVEL.keys()) == [1, 2, 3, 4, 5]


class TestAllocationFootnote:
    """Tests for the allocation footnote."""

    def test_footnote_is_thai_text(self) -> None:
        """The footnote must be non-empty Thai text."""
        assert isinstance(ALLOCATION_FOOTNOTE, str)
        assert len(ALLOCATION_FOOTNOTE) > 0
