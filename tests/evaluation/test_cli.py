"""Tests for evaluation CLI helpers."""

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
                "THALLE",
                "--skip",
                "rag",
                "accuracy",
            ],
        )
        assert args.eval == "routing"
        assert args.provider == "ollama"
        assert args.model == "THALLE"
        assert args.skip == ["rag", "accuracy"]

    def test_valid_eval_types(self) -> None:
        """VALID_EVAL_TYPES contains expected entries."""
        assert "all" in VALID_EVAL_TYPES
        assert "routing" in VALID_EVAL_TYPES
        assert "accuracy-forced" in VALID_EVAL_TYPES

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
        assert settings.llm_provider == "unknown"

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

    def test_run_accuracy_forced(self) -> None:
        """'accuracy-forced' eval type calls run_tax_accuracy_forced."""
        runner = MagicMock()
        runner.run_tax_accuracy_forced.return_value = MagicMock()
        report = _run_selected_evaluation(runner, "accuracy-forced")
        runner.run_tax_accuracy_forced.assert_called_once_with()
        assert report.tax_accuracy is runner.run_tax_accuracy_forced.return_value
