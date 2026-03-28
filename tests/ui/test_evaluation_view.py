"""Tests for evaluation view UI helper functions."""

import pytest


@pytest.fixture()
def _eval_dimensions() -> list[dict[str, str]]:
    """Load EVAL_DIMENSIONS, skipping if streamlit not installed."""
    st = pytest.importorskip("streamlit")  # noqa: F841
    from finance_ai.ui.evaluation_view import EVAL_DIMENSIONS

    return EVAL_DIMENSIONS


@pytest.mark.usefixtures()
class TestEvalDimensions:
    """Tests for evaluation dimension configuration."""

    def test_has_six_dimensions(self, _eval_dimensions: list[dict[str, str]]) -> None:
        """Test that all 6 evaluation dimensions are configured."""
        assert len(_eval_dimensions) == 6

    def test_all_have_required_keys(self, _eval_dimensions: list[dict[str, str]]) -> None:
        """Test that each dimension has key, label, desc."""
        for dim in _eval_dimensions:
            assert "key" in dim
            assert "label" in dim
            assert "desc" in dim

    def test_unique_keys(self, _eval_dimensions: list[dict[str, str]]) -> None:
        """Test that all dimension keys are unique."""
        keys = [d["key"] for d in _eval_dimensions]
        assert len(keys) == len(set(keys))

    def test_expected_keys_present(self, _eval_dimensions: list[dict[str, str]]) -> None:
        """Test that all expected dimension keys exist."""
        keys = {d["key"] for d in _eval_dimensions}
        expected = {"routing", "rag", "accuracy", "hallucination", "quality", "performance"}
        assert keys == expected
