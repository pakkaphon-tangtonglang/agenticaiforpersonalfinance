"""Tests for the architecture study command line."""

from pathlib import Path
from typing import Any

import pytest

from finance_ai.evaluation.architectures import study_cli
from finance_ai.evaluation.architectures.study_cli import (
    PILOT_TASK_IDS,
    build_parser,
    main,
    output_folder,
    select_tasks,
)
from finance_ai.evaluation.architectures.task_models import load_architecture_tasks


class TestHelpers:
    """Folder naming, parsing and task selection."""

    def test_folder_name_is_filesystem_safe(self) -> None:
        """Colons become underscores; pilot gets a suffix."""
        assert output_folder("ollama:gpt-oss:20b", False, None).name == "ollama_gpt-oss_20b"
        assert output_folder("ollama:gpt-oss:20b", True, None).name.endswith("_pilot")

    def test_override_wins(self, tmp_path: Path) -> None:
        """An explicit folder is used as given."""
        assert output_folder("x:y", False, tmp_path) == tmp_path

    def test_defaults(self) -> None:
        """Default model is the free Ollama Cloud gpt-oss:20b with 3 rounds."""
        args = build_parser().parse_args([])
        assert (args.model, args.rounds, args.pilot) == ("ollama:gpt-oss:20b", 3, False)

    def test_pilot_selects_five_tasks_in_file_order(self) -> None:
        """The pilot keeps one task of each kind, in file order."""
        tasks = load_architecture_tasks().tasks
        picked = [task.task_id for task in select_tasks(tasks, pilot=True)]
        assert sorted(picked) == sorted(PILOT_TASK_IDS)
        assert len(select_tasks(tasks, pilot=False)) == 60


class TestMain:
    """main wires config, model and report together."""

    def test_pilot_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pilot runs 1 round on five tasks and writes the report."""
        seen: dict[str, Any] = {}

        def fake_run(config: Any, tasks: Any, model: Any) -> Path:
            seen.update(rounds=config.rounds, tasks=len(tasks), model=model)
            return Path(config.output_dir) / "runs.jsonl"

        monkeypatch.setattr(study_cli, "run_study", fake_run)
        monkeypatch.setattr(study_cli, "build_chat_model", lambda label: f"model:{label}")
        monkeypatch.setattr(study_cli, "write_report", lambda log, cats: log.parent / "report.md")
        report = main(["--pilot", "--output", str(tmp_path)])
        assert seen == {"rounds": 1, "tasks": 5, "model": "model:ollama:gpt-oss:20b"}
        assert report == tmp_path / "report.md"

    def test_report_only_skips_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--report-only rebuilds the report without running anything."""
        monkeypatch.setattr(study_cli, "run_study", lambda *a: pytest.fail("should not run"))
        monkeypatch.setattr(study_cli, "write_report", lambda log, cats: log.parent / "report.md")
        assert main(["--report-only", "--output", str(tmp_path)]) == tmp_path / "report.md"
