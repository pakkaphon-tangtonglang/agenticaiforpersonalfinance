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
