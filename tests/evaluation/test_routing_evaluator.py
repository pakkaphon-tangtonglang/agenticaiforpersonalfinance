"""Tests for routing intent classification evaluator."""

import json
from decimal import Decimal
from unittest.mock import MagicMock

from finance_ai.evaluation.models import RoutingCase, RoutingDataset
from finance_ai.evaluation.routing_evaluator import (
    evaluate_routing_dataset,
    evaluate_single_routing_case,
)


def _make_mock_model(intent: str, confidence: str = "0.95") -> MagicMock:
    """Create a mock chat model that returns a routing decision."""
    model = MagicMock()
    response = MagicMock()
    response.content = json.dumps({"intent": intent, "confidence": confidence})
    model.invoke.return_value = response
    return model


class TestEvaluateSingleRoutingCase:
    """Tests for evaluate_single_routing_case."""

    def test_correct_prediction(self) -> None:
        """Test when model predicts correctly."""
        model = _make_mock_model("tax")
        case = RoutingCase(
            case_id="r_001",
            query="คำนวณภาษี",
            expected_intent="tax",
        )
        result = evaluate_single_routing_case(model, case)
        assert result.is_correct is True
        assert result.predicted_intent == "tax"

    def test_incorrect_prediction(self) -> None:
        """Test when model predicts incorrectly."""
        model = _make_mock_model("expense")
        case = RoutingCase(
            case_id="r_002",
            query="คำนวณภาษี",
            expected_intent="tax",
        )
        result = evaluate_single_routing_case(model, case)
        assert result.is_correct is False
        assert result.predicted_intent == "expense"

    def test_latency_recorded(self) -> None:
        """Test that latency is recorded."""
        model = _make_mock_model("tax")
        case = RoutingCase(
            case_id="r_003",
            query="test",
            expected_intent="tax",
        )
        result = evaluate_single_routing_case(model, case)
        assert result.latency_seconds >= 0.0

    def test_confidence_captured(self) -> None:
        """Test that confidence score is captured."""
        model = _make_mock_model("tax", "0.85")
        case = RoutingCase(
            case_id="r_004",
            query="test",
            expected_intent="tax",
        )
        result = evaluate_single_routing_case(model, case)
        assert result.predicted_confidence == Decimal("0.85")


class TestEvaluateRoutingDataset:
    """Tests for evaluate_routing_dataset."""

    def test_all_correct(self) -> None:
        """Test dataset where all predictions are correct."""
        model = _make_mock_model("tax")
        dataset = RoutingDataset(
            version="1.0",
            cases=[
                RoutingCase(
                    case_id="r_001",
                    query="q1",
                    expected_intent="tax",
                ),
                RoutingCase(
                    case_id="r_002",
                    query="q2",
                    expected_intent="tax",
                ),
            ],
        )
        agg = evaluate_routing_dataset(model, dataset)
        assert agg.total_cases == 2
        assert agg.correct_count == 2
        assert agg.accuracy == Decimal("1.0000")

    def test_partial_accuracy(self) -> None:
        """Test dataset with mixed predictions."""
        model = _make_mock_model("tax")
        dataset = RoutingDataset(
            version="1.0",
            cases=[
                RoutingCase(
                    case_id="r_001",
                    query="q1",
                    expected_intent="tax",
                ),
                RoutingCase(
                    case_id="r_002",
                    query="q2",
                    expected_intent="expense",
                ),
            ],
        )
        agg = evaluate_routing_dataset(model, dataset)
        assert agg.total_cases == 2
        assert agg.correct_count == 1
        assert agg.accuracy == Decimal("0.5000")

    def test_confusion_matrix_built(self) -> None:
        """Test that confusion matrix is built correctly."""
        model = _make_mock_model("tax")
        dataset = RoutingDataset(
            version="1.0",
            cases=[
                RoutingCase(
                    case_id="r_001",
                    query="q1",
                    expected_intent="tax",
                ),
                RoutingCase(
                    case_id="r_002",
                    query="q2",
                    expected_intent="expense",
                ),
            ],
        )
        agg = evaluate_routing_dataset(model, dataset)
        assert "tax" in agg.confusion_matrix
        assert agg.confusion_matrix["tax"]["tax"] == 1
        assert agg.confusion_matrix["expense"]["tax"] == 1

    def test_empty_dataset(self) -> None:
        """Test evaluation with empty dataset."""
        model = _make_mock_model("tax")
        dataset = RoutingDataset(version="1.0", cases=[])
        agg = evaluate_routing_dataset(model, dataset)
        assert agg.total_cases == 0
        assert agg.correct_count == 0
