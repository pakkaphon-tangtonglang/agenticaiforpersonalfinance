"""YAML dataset loaders for evaluation framework."""

from pathlib import Path
from typing import Any

import yaml

from finance_ai.evaluation.models import (
    HallucinationDataset,
    QualityDataset,
    RAGDataset,
    RoutingDataset,
    TaxAccuracyDataset,
)


def _load_yaml_file(path: str) -> dict[str, Any]:
    """Load and parse a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid YAML.

    Example:
        >>> data = _load_yaml_file("data/evaluation/routing_dataset.yaml")
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    content = file_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected YAML dict, got {type(parsed).__name__}")
    return parsed


def load_routing_dataset(path: str) -> RoutingDataset:
    """Load a routing evaluation dataset from YAML.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed RoutingDataset.

    Example:
        >>> dataset = load_routing_dataset("data/evaluation/routing_dataset.yaml")
    """
    data = _load_yaml_file(path)
    return RoutingDataset(**data)


def load_tax_accuracy_dataset(path: str) -> TaxAccuracyDataset:
    """Load a tax accuracy evaluation dataset from YAML.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed TaxAccuracyDataset.

    Example:
        >>> dataset = load_tax_accuracy_dataset("data/evaluation/tax_accuracy_dataset.yaml")
    """
    data = _load_yaml_file(path)
    return TaxAccuracyDataset(**data)


def load_rag_dataset(path: str) -> RAGDataset:
    """Load a RAG retrieval evaluation dataset from YAML.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed RAGDataset.

    Example:
        >>> dataset = load_rag_dataset("data/evaluation/rag_retrieval_dataset.yaml")
    """
    data = _load_yaml_file(path)
    return RAGDataset(**data)


def load_hallucination_dataset(path: str) -> HallucinationDataset:
    """Load a hallucination evaluation dataset from YAML.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed HallucinationDataset.

    Example:
        >>> dataset = load_hallucination_dataset("data/evaluation/hallucination_dataset.yaml")
    """
    data = _load_yaml_file(path)
    return HallucinationDataset(**data)


def load_quality_dataset(path: str) -> QualityDataset:
    """Load a quality evaluation dataset from YAML.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed QualityDataset.

    Example:
        >>> dataset = load_quality_dataset("data/evaluation/quality_dataset.yaml")
    """
    data = _load_yaml_file(path)
    return QualityDataset(**data)
