"""Router study scoring once the router can split a message into several agents."""

from dataclasses import replace
from decimal import Decimal

from finance_ai.agents.router_agent import DEFAULT_ABLATION_CONFIG, _apply_confidence_threshold
from finance_ai.agents.schemas import OrchestratorDecision, RoutedTask
from finance_ai.evaluation.router_thinking.questions import routed_correctly
from finance_ai.evaluation.router_thinking.report import rescore, summarize_setup
from finance_ai.evaluation.router_thinking.runner import production_destinations

from .conftest import make_question, make_row


def decision(intent: str, confidence: str, *tasks: tuple[str, str]) -> OrchestratorDecision:
    """A decision with optional (intent, confidence) parts."""
    return OrchestratorDecision(
        intent=intent,  # type: ignore[arg-type]
        confidence=Decimal(confidence),
        tasks=[
            RoutedTask(intent=name, sub_query=name, confidence=Decimal(conf))  # type: ignore[arg-type]
            for name, conf in tasks
        ],
    )


class TestProductionDestinations:
    """Where production sends a message."""

    def test_single_topic(self) -> None:
        """No parts: the intent decides, as before."""
        assert production_destinations(decision("tax", "0.9")) == ["tax"]

    def test_one_part_counts_as_single(self) -> None:
        """A tasks list with one agent is still a one-topic message."""
        assert production_destinations(decision("tax", "0.9", ("tax", "0.9"))) == ["tax"]

    def test_confident_parts(self) -> None:
        """Several agents: every confident part, general for unknown."""
        routed = decision("tax", "0.9", ("tax", "0.9"), ("unknown", "0.8"), ("report", "0.5"))
        assert production_destinations(routed) == ["tax", "general_chat"]

    def test_no_confident_part_asks_back(self) -> None:
        """Several agents, none confident: ask back."""
        routed = decision("tax", "0.4", ("tax", "0.4"), ("planning", "0.3"))
        assert production_destinations(routed) == ["clarify"]

    def test_threshold_keeps_parts(self) -> None:
        """The study's ask-back step keeps the parts of a multi-topic reply."""
        config = replace(DEFAULT_ABLATION_CONFIG, apply_confidence_threshold=True)
        routed = _apply_confidence_threshold(
            decision("tax", "0.5", ("tax", "0.5"), ("planning", "0.9")), config
        )
        assert routed.intent == "clarify" and len(routed.tasks) == 2
        assert production_destinations(routed) == ["planning"]


class TestSplitScoring:
    """Splitting a one-topic question is a routing miss."""

    def test_split_is_wrong(self) -> None:
        """Two destinations never count as correct for a one-topic question."""
        assert not routed_correctly(["tax", "planning"], {"tax"})
        assert routed_correctly(["tax"], {"tax"})

    def test_rescore_uses_all_destinations(self) -> None:
        """A logged split stays wrong after rescoring."""
        row = make_row(question_id="tax_01", expected="tax").model_copy(
            update={"destination": "tax", "destinations": ["tax", "planning"]}
        )
        assert not rescore([row], [make_question("tax_01", "tax")])[0].correct

    def test_old_rows_without_destinations(self) -> None:
        """Rows logged before this change are scored by their single destination."""
        row = make_row(question_id="tax_01", expected="tax", correct=False).model_copy(
            update={"destination": "tax"}
        )
        assert rescore([row], [make_question("tax_01", "tax")])[0].correct

    def test_summary_counts_splits(self) -> None:
        """The summary reports how many calls were split."""
        split = make_row().model_copy(update={"destinations": ["tax", "planning"]})
        assert summarize_setup("flash_minimal", [split, make_row()]).splits == 1
