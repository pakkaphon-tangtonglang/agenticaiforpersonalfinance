"""Tests for evaluation CLI main entry point."""

from unittest.mock import MagicMock, patch

from finance_ai.evaluation.cli import main


class TestCliMain:
    """Tests for cli.main() orchestration."""

    @patch("finance_ai.evaluation.cli._create_model")
    @patch("finance_ai.evaluation.cli._create_judge_model")
    @patch("finance_ai.evaluation.cli._create_vector_store")
    @patch("finance_ai.evaluation.runner.EvaluationRunner")
    @patch("finance_ai.evaluation.reporter.save_json_report")
    @patch("finance_ai.evaluation.reporter.save_markdown_report")
    def test_main_runs_routing_evaluation(
        self,
        mock_save_markdown_report: MagicMock,
        mock_save_json_report: MagicMock,
        mock_runner_cls: MagicMock,
        mock_create_vector_store: MagicMock,
        mock_create_judge_model: MagicMock,
        mock_create_model: MagicMock,
    ) -> None:
        """main() runs a single evaluation and saves reports."""
        mock_create_model.return_value = MagicMock()
        mock_create_judge_model.return_value = None
        mock_create_vector_store.return_value = MagicMock()
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_report = MagicMock()
        mock_runner.run_routing.return_value = MagicMock()
        mock_runner.run_all.return_value = mock_report
        mock_save_json_report.return_value = "/tmp/report.json"
        mock_save_markdown_report.return_value = "/tmp/report.md"

        main(["--eval", "routing"])

        mock_create_model.assert_called_once()
        mock_runner_cls.assert_called_once()
        mock_save_json_report.assert_called_once()
        mock_save_markdown_report.assert_called_once()

    @patch("finance_ai.evaluation.cli._create_model")
    @patch("finance_ai.evaluation.cli._create_judge_model")
    @patch("finance_ai.evaluation.cli._create_vector_store")
    @patch("finance_ai.evaluation.runner.EvaluationRunner")
    @patch("finance_ai.evaluation.reporter.save_json_report")
    @patch("finance_ai.evaluation.reporter.save_markdown_report")
    def test_main_runs_all_evaluation(
        self,
        mock_save_markdown_report: MagicMock,
        mock_save_json_report: MagicMock,
        mock_runner_cls: MagicMock,
        mock_create_vector_store: MagicMock,
        mock_create_judge_model: MagicMock,
        mock_create_model: MagicMock,
    ) -> None:
        """main() with --eval all creates vector store and runs all."""
        mock_create_model.return_value = MagicMock()
        mock_create_judge_model.return_value = MagicMock()
        mock_create_vector_store.return_value = MagicMock()
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_report = MagicMock()
        mock_runner.run_all.return_value = mock_report
        mock_save_json_report.return_value = "/tmp/report.json"
        mock_save_markdown_report.return_value = "/tmp/report.md"

        main(["--eval", "all", "--skip", "performance"])

        mock_create_vector_store.assert_called_once()
        mock_create_judge_model.assert_called_once()
        mock_runner.run_all.assert_called_once_with(skip={"performance"})
