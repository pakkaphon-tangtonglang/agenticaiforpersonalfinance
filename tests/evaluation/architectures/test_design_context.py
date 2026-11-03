"""Tests for shared design helpers and the intent catalog."""

from collections.abc import Callable
from decimal import Decimal
from typing import Any

import pytest

from finance_ai.evaluation.architectures import design_context
from finance_ai.evaluation.architectures.design_context import (
    DesignContext,
    content_to_text,
    parse_confidence,
    parse_json_reply,
    run_specialist,
)
from finance_ai.evaluation.architectures.intent_catalog import (
    SPECIALIST_AGENTS,
    describe_intents,
    examples_for,
    intent_descriptions,
    intent_examples,
)


class TestReplyParsing:
    """Model reply helpers."""

    def test_content_blocks_flattened(self) -> None:
        """Content blocks become plain text."""
        assert content_to_text([{"type": "text", "text": "a"}, "b"]) == "ab"

    def test_other_content_stringified(self) -> None:
        """Non-string, non-list content is stringified."""
        assert content_to_text(5) == "5"

    def test_json_inside_code_fence(self) -> None:
        """JSON wrapped in a code fence is parsed."""
        assert parse_json_reply('```json\n{"intent": "tax"}\n```') == {"intent": "tax"}

    @pytest.mark.parametrize("reply", ["no json", "{not json}", "[1, 2]"])
    def test_invalid_json_gives_empty_dict(self, reply: str) -> None:
        """Replies without a JSON object give an empty dict."""
        assert parse_json_reply(reply) == {}

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("0.85", Decimal("0.85")), (1.7, Decimal("1")), (-1, Decimal("0")), ("x", Decimal("0"))],
    )
    def test_confidence_clamped(self, raw: Any, expected: Decimal) -> None:
        """Confidence is clamped to [0, 1]; invalid values become 0."""
        assert parse_confidence(raw) == expected


class TestRunSpecialist:
    """run_specialist dispatches to the production executors."""

    def test_dispatches_to_named_agent(
        self, monkeypatch: pytest.MonkeyPatch, make_context: Callable[..., DesignContext]
    ) -> None:
        """The named agent's executor answers."""
        monkeypatch.setitem(
            design_context._EXECUTORS,  # pylint: disable=protected-access
            "tax",
            lambda *args: {"response": "tax answer"},
        )
        assert run_specialist("tax", "q", make_context()) == "tax answer"

    def test_unknown_agent_falls_back_to_general(
        self, monkeypatch: pytest.MonkeyPatch, make_context: Callable[..., DesignContext]
    ) -> None:
        """An unknown agent key uses general chat."""
        monkeypatch.setattr(
            design_context, "execute_general_chat", lambda *args: {"response": "general"}
        )
        assert run_specialist("nope", "q", make_context()) == "general"


class TestIntentCatalog:
    """The catalog reuses the production router wording."""

    def test_every_specialist_has_a_description(self) -> None:
        """All six specialists are described."""
        assert set(SPECIALIST_AGENTS) <= set(intent_descriptions())

    def test_examples_parsed(self) -> None:
        """The router's few-shot examples are found."""
        assert ("จ่ายค่ากาแฟ 80 บาท", "expense") in intent_examples()

    def test_describe_keeps_order(self) -> None:
        """Descriptions come out in the requested order."""
        lines = describe_intents(["report", "tax"]).splitlines()
        assert lines[0].startswith('- "report":') and lines[1].startswith('- "tax":')

    def test_examples_filtered(self) -> None:
        """Only examples of the requested intents are kept."""
        text = examples_for(["tax"])
        assert "→ tax" in text and "→ expense" not in text
