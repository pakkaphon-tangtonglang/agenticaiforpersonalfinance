"""Tests for routing questions and router setups."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from finance_ai.evaluation.router_thinking.questions import (
    destination_for,
    load_all_questions,
    load_new_questions,
    load_sept_questions,
)
from finance_ai.evaluation.router_thinking.setups import COMPARISONS, SETUPS, build_router_model

NEW_YAML = """
questions:
  - {question_id: tax_01, query: "ภาษี", expected: tax}
  - question_id: his_01
    query: "แล้วตอนนี้ล่ะ"
    chat_history: [["user", "ราคา PTT"], ["assistant", "33 บาท"]]
    expected: asset_monitoring
"""
SEPT_YAML = """
version: "1.0"
cases:
  - {case_id: route_unknown_001, query: "อากาศ", expected_intent: unknown, difficulty: easy}
  - {case_id: route_tax_001, query: "ภาษี", expected_intent: tax}
"""


class TestDestination:
    """Router intents map to where the user ends up."""

    def test_general_and_unknown_both_open_general_chat(self) -> None:
        """Both intents open General Chat."""
        assert destination_for("general") == destination_for("unknown") == "general_chat"

    def test_other_intents_unchanged(self) -> None:
        """Specialist intents and clarify pass through."""
        assert destination_for("tax") == "tax"
        assert destination_for("clarify") == "clarify"


class TestLoading:
    """Question files load into RoutingQuestion objects."""

    def test_new_questions_keep_history(self, tmp_path: Path) -> None:
        """Chat history is read as (role, text) pairs."""
        path = tmp_path / "new.yaml"
        path.write_text(NEW_YAML, encoding="utf-8")
        questions = load_new_questions(path)
        assert questions[1].chat_history == [("user", "ราคา PTT"), ("assistant", "33 บาท")]

    def test_sept_questions_relabelled(self, tmp_path: Path) -> None:
        """September unknown becomes general_chat, ids are prefixed, source is sept35."""
        path = tmp_path / "sept.yaml"
        path.write_text(SEPT_YAML, encoding="utf-8")
        first = load_sept_questions(path)[0]
        assert (first.question_id, first.expected, first.source) == (
            "s35_route_unknown_001",
            "general_chat",
            "sept35",
        )

    def test_all_questions_concatenates(self, tmp_path: Path) -> None:
        """New questions come first, then September."""
        new, sept = tmp_path / "new.yaml", tmp_path / "sept.yaml"
        new.write_text(NEW_YAML, encoding="utf-8")
        sept.write_text(SEPT_YAML, encoding="utf-8")
        assert [q.source for q in load_all_questions(new, sept)] == ["new"] * 2 + ["sept35"] * 2

    def test_real_dataset_has_150_unique_questions(self) -> None:
        """The study dataset has 150 questions with unique ids."""
        ids = [q.question_id for q in load_new_questions()]
        assert len(ids) == len(set(ids)) == 150


class TestSetups:
    """Setups bind the right thinking level."""

    def test_comparisons_use_known_setups(self) -> None:
        """Every comparison names two defined setups."""
        assert all(a in SETUPS and b in SETUPS for a, b, _ in COMPARISONS)

    @patch("finance_ai.evaluation.router_thinking.setups.create_google_chat_model")
    def test_level_is_bound(self, create: MagicMock) -> None:
        """A named level is bound on the model for that setup."""
        build_router_model(SETUPS["lite_high"])
        create.return_value.bind.assert_called_once_with(thinking_level="high")
        assert create.call_args.args[0].google_model == "gemini-3.5-flash-lite"

    @patch("finance_ai.evaluation.router_thinking.setups.create_google_chat_model")
    def test_default_sends_no_level(self, create: MagicMock) -> None:
        """'default' binds nothing, so Gemini uses its own default."""
        build_router_model(SETUPS["flash_default"])
        create.return_value.bind.assert_called_once_with()
