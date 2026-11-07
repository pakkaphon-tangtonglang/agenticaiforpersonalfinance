"""Tests for the router study report, CLI and thinking probe."""

import json
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage

from finance_ai.evaluation.router_thinking import cli, thinking_probe
from finance_ai.evaluation.router_thinking.report import (
    rescore,
    Comparison,
    bootstrap_mean_interval,
    compare,
    summarize_setup,
    verdict,
    write_report,
)

from .conftest import fake_router_model, make_question, make_row


def _comparison(low: float, high: float, p: float) -> Comparison:
    """A comparison with a given accuracy CI and McNemar p."""
    return Comparison(
        first="a",
        second="b",
        label="",
        pairs=10,
        accuracy_diff=(low + high) / 2,
        accuracy_diff_low=low,
        accuracy_diff_high=high,
        only_first=0,
        only_second=0,
        mcnemar_p=p,
        latency_diff_median=0,
        latency_diff_low=0,
        latency_diff_high=0,
        sign_p=1,
    )


class TestSummaries:
    """Per-setup numbers."""

    def test_accuracy_counts_errors_as_wrong(self) -> None:
        """An error row lowers accuracy and is counted."""
        rows = [make_row(), make_row(question_id="b", correct=False, error="boom")]
        summary = summarize_setup("flash_minimal", rows)
        assert (summary.accuracy, summary.errors) == (0.5, 1)

    def test_accuracy_by_source_and_destination(self) -> None:
        """Accuracy is split by question set and expected destination."""
        rows = [make_row(), make_row(question_id="s", source="sept35", correct=False)]
        summary = summarize_setup("flash_minimal", rows)
        assert (summary.accuracy_new, summary.accuracy_sept35) == (1.0, 0.0)
        assert summary.by_destination["tax"] == 0.5


class TestCompare:
    """Paired comparisons on the same questions."""

    def test_pairs_and_discordant_counts(self) -> None:
        """Only matched (round, question) pairs are compared."""
        rows = [
            make_row("lite_minimal", "q1", correct=True, latency=1.0),
            make_row("flash_minimal", "q1", correct=False, latency=2.0),
            make_row("lite_minimal", "q2", correct=True, latency=1.0),
        ]
        result = compare(rows, "lite_minimal", "flash_minimal", "x")
        assert (result.pairs, result.only_first, result.only_second) == (1, 1, 0)
        assert result.latency_diff_median == -1.0

    def test_bootstrap_empty(self) -> None:
        """No pairs → zero interval."""
        assert bootstrap_mean_interval([]) == (0.0, 0.0)

    def test_verdicts(self) -> None:
        """No loss / worse / inconclusive against the 3-point margin."""
        assert verdict(_comparison(-0.02, 0.01, 1.0)).startswith("no loss")
        assert verdict(_comparison(-0.10, -0.04, 0.01)) == "worse (significant)"
        assert verdict(_comparison(-0.06, 0.02, 0.4)).startswith("inconclusive")


class TestWriteReport:
    """Report files are written next to the log."""

    def test_writes_markdown_and_json(self, tmp_path: Path) -> None:
        """report.md lists wrong calls; summary.json has the comparisons."""
        log = tmp_path / "calls.jsonl"
        rows = [
            make_row("flash_minimal", "q1"),
            make_row("flash_default", "q1", correct=False),
        ]
        log.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")
        text = write_report(log).read_text(encoding="utf-8")
        assert "| q1 | flash_default | 1 | tax | expense |" in text
        summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert summary["comparisons"][0]["first"] == "flash_minimal"


class TestCli:
    """Command line."""

    def test_pilot_selects_ten_questions(self) -> None:
        """The pilot keeps only the listed ids."""
        questions = [make_question(qid) for qid in (*cli.PILOT_QUESTION_IDS, "other")]
        assert len(cli.select_questions(questions, pilot=True)) == 10
        assert len(cli.select_questions(questions, pilot=False)) == 11

    def test_runs_with_fake_models(self, tmp_path: Path) -> None:
        """main() runs every setup and writes the report."""
        with (
            patch.object(cli, "load_all_questions", return_value=[make_question()]),
            patch.object(cli, "build_router_model", return_value=fake_router_model()),
        ):
            report = cli.main(["--output", str(tmp_path)])
        assert "flash_minimal" in report.read_text(encoding="utf-8")

    def test_setups_filter(self, tmp_path: Path) -> None:
        """--setups runs only the named setups."""
        with (
            patch.object(cli, "load_all_questions", return_value=[make_question()]),
            patch.object(cli, "build_router_model", return_value=fake_router_model()),
        ):
            cli.main(["--output", str(tmp_path), "--setups", "flash_minimal"])
        rows = (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(rows) == 1 and '"setup":"flash_minimal"' in rows[0].replace(" ", "")


class TestProbe:
    """Thinking probe rows and table."""

    def test_row_reads_thinking_tokens(self) -> None:
        """Reasoning tokens come from output_token_details."""
        reply = AIMessage(
            content="7",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 30,
                "total_tokens": 40,
                "output_token_details": {"reasoning": 28},
            },
        )
        row = thinking_probe._row("m", None, "reasoning", "math_1", reply, 1.0)  # noqa: SLF001
        assert (row.level, row.thinking_tokens) == ("default", 28)

    def test_summary_table(self) -> None:
        """One table line per model, level and task with data."""
        row = thinking_probe.ProbeRow(
            model="gemini-3.5-flash",
            level="low",
            task="routing",
            prompt_id="tax_01",
            thinking_tokens=0,
            output_tokens=15,
            latency_seconds=1.0,
            reply="{}",
        )
        lines = thinking_probe.summary_lines([row])
        assert lines[-1].startswith("| gemini-3.5-flash | low | routing | 1 | 0 |")


class TestProbeRun:
    """The probe runs every model × level × prompt and writes its report."""

    def test_main_with_fake_model(self, tmp_path: Path) -> None:
        """2 models × 5 levels × (10 routing + 3 reasoning) rows, table written."""
        with (
            patch.object(thinking_probe, "probe_model", return_value=fake_router_model()),
            patch.object(thinking_probe, "OUTPUT_FOLDER", tmp_path),
        ):
            report = thinking_probe.main()
        rows = (tmp_path / "probe.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(rows) == 2 * 5 * 13
        assert "default (no setting)" in report.read_text(encoding="utf-8")

    @patch("finance_ai.evaluation.router_thinking.thinking_probe.create_google_chat_model")
    def test_probe_model_levels(self, create) -> None:  # type: ignore[no-untyped-def]
        """A level is bound; None binds nothing."""
        thinking_probe.probe_model("gemini-3.5-flash", "low")
        create.return_value.bind.assert_called_with(thinking_level="low")
        thinking_probe.probe_model("gemini-3.5-flash", None)
        create.return_value.bind.assert_called_with()


class TestRescore:
    """Correctness is recomputed from the current labels."""

    def test_second_label_counts(self) -> None:
        """A destination in also_accept becomes correct."""
        question = make_question("q1", expected="planning")
        question = question.model_copy(update={"also_accept": ["general_chat"]})
        row = make_row(question_id="q1", correct=False, expected="planning")
        row = row.model_copy(update={"destination": "general_chat"})
        assert rescore([row], [question])[0].correct

    def test_unknown_question_and_errors_unchanged(self) -> None:
        """Rows for other questions, or with errors, keep their value."""
        errored = make_row(question_id="q1", correct=False, error="boom")
        other = make_row(question_id="zz", correct=True)
        rescored = rescore([errored, other], [make_question("q1")])
        assert [row.correct for row in rescored] == [False, True]
