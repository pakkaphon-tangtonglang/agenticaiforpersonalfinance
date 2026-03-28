"""Tests for evaluation dataset loaders."""

import tempfile
from pathlib import Path

import pytest

from finance_ai.evaluation.datasets import (
    _load_yaml_file,
    load_hallucination_dataset,
    load_quality_dataset,
    load_rag_dataset,
    load_routing_dataset,
    load_tax_accuracy_dataset,
)


def _write_temp_yaml(content: str) -> str:
    """Write YAML content to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    tmp.write(content)
    tmp.close()
    return tmp.name


class TestLoadYamlFile:
    """Tests for the _load_yaml_file helper."""

    def test_valid_yaml(self) -> None:
        """Test loading a valid YAML file."""
        path = _write_temp_yaml("name: test\nversion: '1.0'")
        data = _load_yaml_file(path)
        assert data["name"] == "test"
        Path(path).unlink()

    def test_file_not_found(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _load_yaml_file("/nonexistent/path.yaml")

    def test_invalid_yaml_type(self) -> None:
        """Test that non-dict YAML raises ValueError."""
        path = _write_temp_yaml("- item1\n- item2")
        with pytest.raises(ValueError, match="Expected YAML dict"):
            _load_yaml_file(path)
        Path(path).unlink()


class TestLoadRoutingDataset:
    """Tests for load_routing_dataset."""

    def test_valid_routing_dataset(self) -> None:
        """Test loading a valid routing dataset."""
        yaml_content = """
name: routing_evaluation
version: "1.0"
cases:
  - case_id: "route_001"
    query: "คำนวณภาษีปี 2024"
    expected_intent: "tax"
    difficulty: "easy"
  - case_id: "route_002"
    query: "จ่ายค่ากาแฟ 80 บาท"
    expected_intent: "expense"
"""
        path = _write_temp_yaml(yaml_content)
        dataset = load_routing_dataset(path)
        assert len(dataset.cases) == 2
        assert dataset.cases[0].expected_intent == "tax"
        Path(path).unlink()


class TestLoadTaxAccuracyDataset:
    """Tests for load_tax_accuracy_dataset."""

    def test_valid_tax_dataset(self) -> None:
        """Test loading a valid tax accuracy dataset."""
        yaml_content = """
name: tax_accuracy_evaluation
version: "1.0"
cases:
  - case_id: "tax_001"
    query: "คำนวณภาษี"
    gross_income: "600000"
    deductions_by_type:
      personal_allowance: "60000"
    expected_total_tax: "29000"
    expected_effective_rate: "0.0483"
    tolerance_thb: "100"
"""
        path = _write_temp_yaml(yaml_content)
        dataset = load_tax_accuracy_dataset(path)
        assert len(dataset.cases) == 1
        assert dataset.cases[0].gross_income == 600000
        Path(path).unlink()


class TestLoadRAGDataset:
    """Tests for load_rag_dataset."""

    def test_valid_rag_dataset(self) -> None:
        """Test loading a valid RAG dataset."""
        yaml_content = """
name: rag_retrieval_evaluation
version: "1.0"
cases:
  - case_id: "rag_001"
    query: "ค่าลดหย่อนส่วนตัว"
    domain_filter: "tax"
    expected_source_files:
      - "tax_deductions_guide.md"
    expected_keywords:
      - "60,000"
"""
        path = _write_temp_yaml(yaml_content)
        dataset = load_rag_dataset(path)
        assert len(dataset.cases) == 1
        assert dataset.cases[0].domain_filter == "tax"
        Path(path).unlink()


class TestLoadHallucinationDataset:
    """Tests for load_hallucination_dataset."""

    def test_valid_hallucination_dataset(self) -> None:
        """Test loading a valid hallucination dataset."""
        yaml_content = """
name: hallucination_evaluation
version: "1.0"
cases:
  - case_id: "hal_001"
    query: "ค่าลดหย่อนส่วนตัว"
    category: "fabricated_number"
    known_facts:
      - "ค่าลดหย่อนส่วนตัว 60,000 บาท"
    forbidden_patterns:
      - "70,000"
"""
        path = _write_temp_yaml(yaml_content)
        dataset = load_hallucination_dataset(path)
        assert len(dataset.cases) == 1
        assert dataset.cases[0].category == "fabricated_number"
        Path(path).unlink()


class TestLoadQualityDataset:
    """Tests for load_quality_dataset."""

    def test_valid_quality_dataset(self) -> None:
        """Test loading a valid quality dataset."""
        yaml_content = """
name: quality_evaluation
version: "1.0"
cases:
  - case_id: "qual_001"
    query: "คำนวณภาษีปี 2024"
    expected_agent: "tax"
"""
        path = _write_temp_yaml(yaml_content)
        dataset = load_quality_dataset(path)
        assert len(dataset.cases) == 1
        assert dataset.cases[0].expected_agent == "tax"
        Path(path).unlink()
