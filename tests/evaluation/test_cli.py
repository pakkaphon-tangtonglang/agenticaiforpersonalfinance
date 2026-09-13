"""Tests for evaluation CLI helpers."""

import json
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import MagicMock, patch

from finance_ai.evaluation.cli import (
    VALID_EVAL_TYPES,
    VALID_PROVIDERS,
    _build_settings,
    _create_judge_model,
    _create_model,
    _run_selected_evaluation,
    parse_args,
)


class TestEvaluationCli:
    """Tests for evaluation CLI parser."""

    def test_parse_args_defaults(self) -> None:
        """Default arguments are populated correctly."""
        args = parse_args([])
        assert args.eval == "all"
        assert args.provider == "google"
        assert args.model == "gemini-2.0-flash"
        assert args.output_dir == "data/evaluation/results"

    def test_parse_args_custom_values(self) -> None:
        """Custom arguments are parsed correctly."""
        args = parse_args(
            [
                "--eval",
                "routing",
                "--provider",
                "ollama",
                "--model",
                "minimax-m3",
                "--skip",
                "rag",
                "accuracy",
            ],
        )
        assert args.eval == "routing"
        assert args.provider == "ollama"
        assert args.model == "minimax-m3"
        assert args.skip == ["rag", "accuracy"]

    def test_valid_eval_types(self) -> None:
        """VALID_EVAL_TYPES contains expected entries."""
        assert "all" in VALID_EVAL_TYPES
        assert "routing" in VALID_EVAL_TYPES
        assert "accuracy-forced" in VALID_EVAL_TYPES
        assert "safety" in VALID_EVAL_TYPES

    def test_valid_providers(self) -> None:
        """VALID_PROVIDERS contains expected entries."""
        assert "google" in VALID_PROVIDERS
        assert "ollama" in VALID_PROVIDERS


class TestCliHelpers:
    """Tests for CLI helper functions."""

    def test_build_settings_updates_provider_and_model(self) -> None:
        """_build_settings overrides provider and model fields."""
        settings = _build_settings("google", "gemini-test")
        assert settings.llm_provider == "google"
        assert settings.google_model == "gemini-test"

    def test_build_settings_unknown_provider(self) -> None:
        """_build_settings handles unknown provider gracefully."""
        settings = _build_settings("unknown", "model-x")
        assert cast(str, settings.llm_provider) == "unknown"

    @patch("finance_ai.evaluation.cli._build_settings")
    @patch("finance_ai.agents.llm_factory.create_chat_model")
    def test_create_model(
        self,
        mock_create_chat_model: MagicMock,
        mock_build_settings: MagicMock,
    ) -> None:
        """_create_model builds settings and creates chat model."""
        mock_settings = MagicMock()
        mock_build_settings.return_value = mock_settings
        mock_create_chat_model.return_value = MagicMock()
        model = _create_model("google", "gemini-test")
        mock_build_settings.assert_called_once_with("google", "gemini-test")
        mock_create_chat_model.assert_called_once_with(settings=mock_settings)
        assert model is not None

    def test_create_judge_model_empty_name_returns_none(self) -> None:
        """_create_judge_model returns None when model name is empty."""
        assert _create_judge_model("openrouter", "") is None

    @patch("finance_ai.evaluation.cli._build_settings")
    @patch("finance_ai.agents.llm_factory.create_chat_model")
    def test_create_judge_model(
        self,
        mock_create_chat_model: MagicMock,
        mock_build_settings: MagicMock,
    ) -> None:
        """_create_judge_model creates chat model when name provided."""
        mock_settings = MagicMock()
        mock_build_settings.return_value = mock_settings
        mock_create_chat_model.return_value = MagicMock()
        model = _create_judge_model("openrouter", "gpt-4o")
        assert model is not None


class TestRunSelectedEvaluation:
    """Tests for _run_selected_evaluation dispatch."""

    def test_run_all_delegates_to_runner(self) -> None:
        """'all' eval type calls runner.run_all."""
        runner = MagicMock()
        runner.run_all.return_value = MagicMock()
        report = _run_selected_evaluation(runner, "all")
        runner.run_all.assert_called_once_with(skip=set())
        assert report is runner.run_all.return_value

    def test_run_routing(self) -> None:
        """'routing' eval type calls run_routing and sets field."""
        runner = MagicMock()
        runner.run_routing.return_value = MagicMock()
        report = _run_selected_evaluation(runner, "routing")
        runner.run_routing.assert_called_once_with()
        assert report.routing is runner.run_routing.return_value

    def test_run_safety(self) -> None:
        """'safety' eval type calls run_recommendation_safety and sets field."""
        runner = MagicMock()
        runner.run_recommendation_safety.return_value = MagicMock()
        report = _run_selected_evaluation(runner, "safety")
        runner.run_recommendation_safety.assert_called_once_with()
        assert report.recommendation_safety is runner.run_recommendation_safety.return_value

    def test_run_accuracy_forced(self) -> None:
        """'accuracy-forced' eval type calls run_tax_accuracy_forced."""
        runner = MagicMock()
        runner.run_tax_accuracy_forced.return_value = MagicMock()
        report = _run_selected_evaluation(runner, "accuracy-forced")
        runner.run_tax_accuracy_forced.assert_called_once_with()
        assert report.tax_accuracy is runner.run_tax_accuracy_forced.return_value


class TestModelComparisonCli:
    """Tests for the --compare / --models CLI surface."""

    def test_parse_args_compare_flag(self) -> None:
        """--compare selects comparison mode with an empty default list."""
        args = parse_args(["--compare"])
        assert args.compare is True
        assert args.models == []

    def test_parse_args_models_override(self) -> None:
        """--models accepts provider:model strings (colons inside names)."""
        args = parse_args(
            [
                "--compare",
                "--models",
                "ollama:minimax-m3",
                "ollama:qwen3.5:397b",
                "google:gemini-3.5",
            ],
        )
        assert args.models == [
            "ollama:minimax-m3",
            "ollama:qwen3.5:397b",
            "google:gemini-3.5",
        ]

    def test_main_compare_dispatches_to_comparison(self, tmp_path: Any, monkeypatch: Any) -> None:
        """main() with --compare runs the comparison and saves reports."""
        from finance_ai.evaluation import model_comparison  # noqa: PLC0415
        from finance_ai.evaluation.cli import main  # noqa: PLC0415
        from finance_ai.evaluation.models import (  # noqa: PLC0415
            ModelComparisonEntry,
            ModelComparisonResult,
        )

        fake_result = ModelComparisonResult(
            dimension="routing",
            generated_at=datetime.now(timezone.utc),
            entries=[
                ModelComparisonEntry(
                    provider="ollama",
                    model_name="minimax-m3",
                    status="ok",
                )
            ],
        )

        def _fake_comparison(specs: object, data_dir: str = "data/evaluation") -> object:
            return fake_result

        monkeypatch.setattr(model_comparison, "run_model_comparison", _fake_comparison)
        out_dir = tmp_path / "results"
        main(["--compare", "--output-dir", str(out_dir)])

        saved = list(out_dir.glob("model_comparison_*.json"))
        assert len(saved) == 1
        data = json.loads(saved[0].read_text(encoding="utf-8"))
        assert data["entries"][0]["model_name"] == "minimax-m3"
