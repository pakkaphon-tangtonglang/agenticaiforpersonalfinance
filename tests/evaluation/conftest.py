"""Shared fixtures for evaluation tests."""

from decimal import Decimal

import pytest

from finance_ai.evaluation.models import (
    HallucinationCase,
    HallucinationDataset,
    QualityCase,
    QualityDataset,
    RAGCase,
    RAGDataset,
    RoutingCase,
    RoutingDataset,
    TaxAccuracyCase,
    TaxAccuracyDataset,
)


@pytest.fixture()
def sample_routing_cases() -> list[RoutingCase]:
    """Create sample routing test cases."""
    return [
        RoutingCase(
            case_id="route_001",
            query="คำนวณภาษีปี 2024",
            expected_intent="tax",
            difficulty="easy",
        ),
        RoutingCase(
            case_id="route_002",
            query="จ่ายค่ากาแฟ 80 บาท",
            expected_intent="expense",
            difficulty="easy",
        ),
        RoutingCase(
            case_id="route_003",
            query="ดูพอร์ตของฉัน",
            expected_intent="asset_monitoring",
            difficulty="easy",
        ),
        RoutingCase(
            case_id="route_004",
            query="ลงทุน RMF เพื่อลดหย่อนภาษี",
            expected_intent="tax",
            difficulty="hard",
        ),
        RoutingCase(
            case_id="route_005",
            query="วันนี้อากาศเป็นยังไง",
            expected_intent="unknown",
            difficulty="easy",
        ),
    ]


@pytest.fixture()
def sample_routing_dataset(
    sample_routing_cases: list[RoutingCase],
) -> RoutingDataset:
    """Create a sample routing dataset."""
    return RoutingDataset(version="1.0", cases=sample_routing_cases)


@pytest.fixture()
def sample_tax_cases() -> list[TaxAccuracyCase]:
    """Create sample tax accuracy test cases."""
    return [
        TaxAccuracyCase(
            case_id="tax_001",
            query="คำนวณภาษี เงินเดือน 50,000 บาทต่อเดือน",
            gross_income=Decimal("600000"),
            deductions_by_type={"personal_allowance": Decimal("60000")},
            expected_total_tax=Decimal("29000"),
            expected_effective_rate=Decimal("0.0483"),
            tolerance_thb=Decimal("100"),
        ),
        TaxAccuracyCase(
            case_id="tax_002",
            query="รายได้ 1.2 ล้าน ลดหย่อน RMF 200,000",
            gross_income=Decimal("1200000"),
            deductions_by_type={
                "personal_allowance": Decimal("60000"),
                "rmf": Decimal("200000"),
            },
            expected_total_tax=Decimal("80000"),
            expected_effective_rate=Decimal("0.0667"),
            tolerance_thb=Decimal("500"),
        ),
    ]


@pytest.fixture()
def sample_tax_dataset(
    sample_tax_cases: list[TaxAccuracyCase],
) -> TaxAccuracyDataset:
    """Create a sample tax accuracy dataset."""
    return TaxAccuracyDataset(version="1.0", cases=sample_tax_cases)


@pytest.fixture()
def sample_rag_cases() -> list[RAGCase]:
    """Create sample RAG retrieval test cases."""
    return [
        RAGCase(
            case_id="rag_001",
            query="ค่าลดหย่อนส่วนตัว",
            domain_filter="tax",
            expected_source_files=["tax_deductions_guide.md"],
            expected_keywords=["60,000", "ลดหย่อน"],
        ),
        RAGCase(
            case_id="rag_002",
            query="กองทุน RMF คืออะไร",
            domain_filter="investment",
            expected_source_files=["investment_mutual_funds.md"],
            expected_keywords=["RMF"],
        ),
    ]


@pytest.fixture()
def sample_rag_dataset(sample_rag_cases: list[RAGCase]) -> RAGDataset:
    """Create a sample RAG dataset."""
    return RAGDataset(version="1.0", cases=sample_rag_cases)


@pytest.fixture()
def sample_hallucination_cases() -> list[HallucinationCase]:
    """Create sample hallucination test cases."""
    return [
        HallucinationCase(
            case_id="hal_001",
            query="ค่าลดหย่อนส่วนตัวเท่าไหร่",
            category="fabricated_number",
            known_facts=["ค่าลดหย่อนส่วนตัว 60,000 บาท"],
            forbidden_patterns=[r"70,000", r"80,000"],
        ),
    ]


@pytest.fixture()
def sample_hallucination_dataset(
    sample_hallucination_cases: list[HallucinationCase],
) -> HallucinationDataset:
    """Create a sample hallucination dataset."""
    return HallucinationDataset(version="1.0", cases=sample_hallucination_cases)


@pytest.fixture()
def sample_quality_cases() -> list[QualityCase]:
    """Create sample quality test cases."""
    return [
        QualityCase(
            case_id="qual_001",
            query="คำนวณภาษีปี 2024 เงินเดือน 50,000",
            expected_agent="tax",
        ),
        QualityCase(
            case_id="qual_002",
            query="สรุปค่าใช้จ่ายเดือนนี้",
            expected_agent="expense",
        ),
    ]


@pytest.fixture()
def sample_quality_dataset(
    sample_quality_cases: list[QualityCase],
) -> QualityDataset:
    """Create a sample quality dataset."""
    return QualityDataset(version="1.0", cases=sample_quality_cases)
