"""Tests for the router ablation runner."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from finance_ai.agents.router_agent import (
    DEFAULT_ABLATION_CONFIG,
    RouterAblationConfig,
)
from finance_ai.evaluation.router_ablation import (
    ABLATION_VARIANTS,
    AblationCallRow,
    AblationRunner,
    parse_model_spec,
)


class TestAblationVariants:
    """Variant configurations follow the one-at-a-time + full design."""

    def test_six_variants_defined(self) -> None:
        """Exactly the six planned variants exist, in order."""
        assert list(ABLATION_VARIANTS) == [
            "baseline",
            "+few_shot",
            "+history",
            "+symbol",
            "+threshold",
            "full",
        ]

    def test_baseline_all_flags_off(self) -> None:
        """Baseline disables every helper."""
        assert ABLATION_VARIANTS["baseline"] == RouterAblationConfig(
            include_chat_history=False,
            include_few_shot_examples=False,
            resolve_asset_hint=False,
            apply_confidence_threshold=False,
        )

    def test_each_single_feature_variant_toggles_one_flag(self) -> None:
        """Single-feature variants turn on exactly their own flag."""
        base = RouterAblationConfig(
            include_chat_history=False,
            include_few_shot_examples=False,
            resolve_asset_hint=False,
            apply_confidence_threshold=False,
        )
        expected = {
            "+few_shot": "include_few_shot_examples",
            "+history": "include_chat_history",
            "+symbol": "resolve_asset_hint",
            "+threshold": "apply_confidence_threshold",
        }
        for variant_name, flag_name in expected.items():
            config = ABLATION_VARIANTS[variant_name]
            toggled = replace(base, **{flag_name: True})
            assert config == toggled, variant_name

    def test_full_equals_production_default(self) -> None:
        """full reproduces DEFAULT_ABLATION_CONFIG exactly."""
        assert ABLATION_VARIANTS["full"] == DEFAULT_ABLATION_CONFIG


class TestParseModelSpec:
    """Model spec strings like 'ollama:minimax-m3' parse into pairs."""

    def test_parses_provider_and_model(self) -> None:
        """'ollama:minimax-m3' -> ('ollama', 'minimax-m3')."""
        assert parse_model_spec("ollama:minimax-m3") == ("ollama", "minimax-m3")

    def test_parses_google_model(self) -> None:
        """'google:gemini-3.5-flash' -> ('google', 'gemini-3.5-flash')."""
        assert parse_model_spec("google:gemini-3.5-flash") == (
            "google",
            "gemini-3.5-flash",
        )

    def test_missing_colon_raises(self) -> None:
        """A spec without ':' raises ValueError."""
        with pytest.raises(ValueError, match="provider:model"):
            parse_model_spec("minimax-m3")

    def test_empty_parts_raise(self) -> None:
        """Empty provider or model raises ValueError."""
        with pytest.raises(ValueError):
            parse_model_spec("ollama:")
        with pytest.raises(ValueError):
            parse_model_spec(":minimax-m3")


FIXTURE_DIR = "tests/evaluation/fixtures/ablation"


def _mock_model_returning(content: str) -> MagicMock:
    """Build a MagicMock chat model whose invoke returns an AIMessage."""
    model = MagicMock()
    model.invoke.return_value = AIMessage(content=content)
    return model


def _patch_asset_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the Yahoo symbol search inside router_agent for tests."""
    monkeypatch.setattr("finance_ai.agents.router_agent.search_asset_symbols", lambda _query: [])


def _make_runner(tmp_path: Path, model: MagicMock) -> tuple["AblationRunner", Path]:
    """Build a runner against fixture data with a tmp log path."""
    log_path = tmp_path / "ablation_log.jsonl"
    runner = AblationRunner(
        models={"ollama:fake": model},
        log_path=log_path,
        data_dir=FIXTURE_DIR,
    )
    return runner, log_path


class TestAblationRunner:
    """Execution loop: JSONL logging, resume, error rows."""

    def test_run_writes_expected_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """2 variants x 2 rounds x 3 cases (2 base + 1 history) = 12 rows."""
        _patch_asset_search(monkeypatch)
        model = _mock_model_returning('{"intent": "tax", "confidence": 0.9}')
        runner, log_path = _make_runner(tmp_path, model)
        variants = {
            "baseline": ABLATION_VARIANTS["baseline"],
            "full": ABLATION_VARIANTS["full"],
        }
        executed = runner.run(variants=variants, rounds=2)
        assert executed == 12
        rows = runner.read_rows()
        assert len(rows) == 12
        assert log_path.exists()

    def test_history_case_carries_history_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fixture history case is logged with case_group='history'."""
        _patch_asset_search(monkeypatch)
        model = _mock_model_returning('{"intent": "expense", "confidence": 0.9}')
        runner, _ = _make_runner(tmp_path, model)
        runner.run(variants={"baseline": ABLATION_VARIANTS["baseline"]}, rounds=1)
        groups = {row.case_id: row.case_group for row in runner.read_rows()}
        assert groups["fx_hist_001"] == "history"
        assert groups["fx_001"] == "base"

    def test_resume_skips_completed_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second run over the same grid executes zero new calls."""
        _patch_asset_search(monkeypatch)
        model = _mock_model_returning('{"intent": "tax", "confidence": 0.9}')
        runner, _ = _make_runner(tmp_path, model)
        variants = {"baseline": ABLATION_VARIANTS["baseline"]}
        assert runner.run(variants=variants, rounds=1) == 3
        assert runner.run(variants=variants, rounds=1) == 0

    def test_failed_calls_logged_and_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raising model produces an error row; the next run retries it."""
        _patch_asset_search(monkeypatch)
        model = MagicMock()
        model.invoke.side_effect = RuntimeError("boom")
        runner, _ = _make_runner(tmp_path, model)
        variants = {"baseline": ABLATION_VARIANTS["baseline"]}
        assert runner.run(variants=variants, rounds=1) == 3  # all errored
        rows = runner.read_rows()
        assert all(row.error == "boom" for row in rows)

        model.invoke.side_effect = None
        model.invoke.return_value = AIMessage(content='{"intent": "tax", "confidence": 0.9}')
        assert runner.run(variants=variants, rounds=1) == 3  # error rows retried

    def test_row_serializes_to_json_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each log line is valid JSON matching AblationCallRow."""
        _patch_asset_search(monkeypatch)
        model = _mock_model_returning('{"intent": "tax", "confidence": 0.9}')
        runner, log_path = _make_runner(tmp_path, model)
        runner.run(variants={"baseline": ABLATION_VARIANTS["baseline"]}, rounds=1)
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            row = AblationCallRow(**json.loads(line))
            assert row.model == "ollama:fake"
