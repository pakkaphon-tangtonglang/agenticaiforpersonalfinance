"""Integrity tests for the real routing history dataset."""

from finance_ai.evaluation.datasets import load_routing_history_dataset
from finance_ai.evaluation.models import RoutingHistoryDataset

DATASET_PATH = "data/evaluation/routing_history_dataset.yaml"


def _load_dataset() -> RoutingHistoryDataset:
    """Load the shipped routing history dataset."""
    return load_routing_history_dataset(DATASET_PATH)


class TestRoutingHistoryDatasetIntegrity:
    """The shipped dataset must be complete and well-formed."""

    def test_has_ten_cases(self) -> None:
        """Exactly 10 multi-turn cases ship with the repo."""
        dataset = _load_dataset()
        assert len(dataset.cases) == 10

    def test_case_ids_unique_and_sequential(self) -> None:
        """Case ids are hist_001..hist_010 with no duplicates."""
        dataset = _load_dataset()
        ids = [case.case_id for case in dataset.cases]
        assert ids == [f"hist_{i:03d}" for i in range(1, 11)]

    def test_every_case_has_history(self) -> None:
        """No case may have an empty chat_history."""
        dataset = _load_dataset()
        assert all(len(case.chat_history) >= 2 for case in dataset.cases)

    def test_intent_coverage(self) -> None:
        """History cases cover the target intents with the planned counts."""
        dataset = _load_dataset()
        counts: dict[str, int] = {}
        for case in dataset.cases:
            counts[case.expected_intent] = counts.get(case.expected_intent, 0) + 1
        assert counts == {
            "asset_monitoring": 2,
            "expense": 2,
            "tax": 2,
            "planning": 1,
            "recommendation": 1,
            "report": 1,
            "general": 1,
        }
