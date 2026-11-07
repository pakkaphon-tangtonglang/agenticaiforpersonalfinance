"""Routing questions for the router study and the destination labels they use."""

from pathlib import Path
from typing import Literal, Optional, cast, get_args

import yaml
from pydantic import BaseModel, Field

Destination = Literal[
    "tax",
    "expense",
    "asset_monitoring",
    "planning",
    "recommendation",
    "report",
    "general_chat",
    "clarify",
]
DESTINATIONS: tuple[str, ...] = get_args(Destination)
QuestionSource = Literal["new", "sept35"]

NEW_QUESTIONS_PATH = Path("data/evaluation/routing_v2_dataset.yaml")
SEPT_QUESTIONS_PATH = Path("data/evaluation/routing_dataset.yaml")
GENERAL_CHAT_INTENTS = ("general", "unknown")
CLARIFY_INTENT = "clarify"


class RoutingQuestion(BaseModel):
    """One routing question with the destination the router should pick.

    Example:
        >>> RoutingQuestion(question_id="tax_01", query="ภาษี", expected="tax").source
        'new'
    """

    question_id: str
    query: str
    expected: Destination
    also_accept: list[Destination] = Field(default_factory=list)
    chat_history: list[tuple[str, str]] = Field(default_factory=list)
    difficulty: str = "medium"
    source: QuestionSource = "new"


def destination_for(intent: str) -> str:
    """Map a router intent to where the user ends up.

    Args:
        intent: Router intent after the confidence threshold.

    Returns:
        "general_chat" for general/unknown (both open General Chat), else the intent.

    Example:
        >>> destination_for("unknown")
        'general_chat'
    """
    return "general_chat" if intent in GENERAL_CHAT_INTENTS else intent


def accepted_destinations(question: RoutingQuestion) -> set[str]:
    """Return every destination that counts as correct for a question.

    Args:
        question: The routing question.

    Returns:
        ``expected`` plus any ``also_accept`` labels.

    Example:
        >>> sorted(accepted_destinations(RoutingQuestion(
        ...     question_id="x", query="q", expected="planning", also_accept=["general_chat"])))
        ['general_chat', 'planning']
    """
    return {question.expected, *question.also_accept}


def load_new_questions(path: Path = NEW_QUESTIONS_PATH) -> list[RoutingQuestion]:
    """Load the 150 questions written for this study.

    Args:
        path: YAML file with a ``questions`` list.

    Returns:
        Questions in file order.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [RoutingQuestion(**item) for item in data["questions"]]


def load_sept_questions(path: Path = SEPT_QUESTIONS_PATH) -> list[RoutingQuestion]:
    """Load the 35 September routing questions, relabelled as destinations.

    Args:
        path: The September routing dataset.

    Returns:
        Questions with ids prefixed ``s35_`` and source ``sept35``.
    """
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        RoutingQuestion(
            question_id=f"s35_{case['case_id']}",
            query=case["query"],
            expected=cast(Destination, destination_for(case["expected_intent"])),
            difficulty=case.get("difficulty", "medium"),
            source="sept35",
        )
        for case in data["cases"]
    ]


def load_all_questions(
    new_path: Optional[Path] = None, sept_path: Optional[Path] = None
) -> list[RoutingQuestion]:
    """Load the new questions followed by the September set.

    Args:
        new_path: Override for the new question file.
        sept_path: Override for the September file.

    Returns:
        All questions (185 with the default files).
    """
    new = load_new_questions(new_path or NEW_QUESTIONS_PATH)
    return new + load_sept_questions(sept_path or SEPT_QUESTIONS_PATH)
