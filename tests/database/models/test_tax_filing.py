"""Tests for the TaxFiling database model."""

import json
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from finance_ai.database.models.tax_filing import TaxFiling
from finance_ai.database.models.user import User


class TestTaxFilingModel:
    """Tests for TaxFiling model creation, constraints, and relationships."""

    def test_create_tax_filing(self, test_session: Session, sample_user: User) -> None:
        """Test creating a tax filing with all fields."""
        breakdown = json.dumps(
            [
                {"bracket": "0-150000", "rate": "0%", "tax": "0"},
                {"bracket": "150001-300000", "rate": "5%", "tax": "7500"},
            ]
        )
        filing = TaxFiling(
            user_id=sample_user.id,
            tax_year=2024,
            gross_income=Decimal("960000.00"),
            total_deductions=Decimal("160000.00"),
            net_income=Decimal("800000.00"),
            total_tax=Decimal("75000.00"),
            effective_tax_rate=Decimal("0.0781"),
            withholding_tax_paid=Decimal("60000.00"),
            tax_due_or_refund=Decimal("15000.00"),
            filing_status="draft",
            tax_breakdown_json=breakdown,
        )
        test_session.add(filing)
        test_session.commit()
        test_session.refresh(filing)
        assert filing.id is not None
        assert filing.total_tax == Decimal("75000.00")
        assert filing.effective_tax_rate == Decimal("0.0781")

    def test_tax_filing_unique_constraint(self, test_session: Session, sample_user: User) -> None:
        """Test that only one filing per user per year is allowed."""
        filing_one = TaxFiling(
            user_id=sample_user.id,
            tax_year=2024,
            gross_income=Decimal("960000.00"),
            total_deductions=Decimal("160000.00"),
            net_income=Decimal("800000.00"),
            total_tax=Decimal("75000.00"),
            effective_tax_rate=Decimal("0.0781"),
            tax_due_or_refund=Decimal("15000.00"),
        )
        test_session.add(filing_one)
        test_session.commit()
        filing_two = TaxFiling(
            user_id=sample_user.id,
            tax_year=2024,
            gross_income=Decimal("1000000.00"),
            total_deductions=Decimal("200000.00"),
            net_income=Decimal("800000.00"),
            total_tax=Decimal("80000.00"),
            effective_tax_rate=Decimal("0.0800"),
            tax_due_or_refund=Decimal("20000.00"),
        )
        test_session.add(filing_two)
        with pytest.raises(IntegrityError):
            test_session.commit()

    def test_tax_filing_default_withholding(self, test_session: Session, sample_user: User) -> None:
        """Test that withholding_tax_paid defaults to zero."""
        filing = TaxFiling(
            user_id=sample_user.id,
            tax_year=2024,
            gross_income=Decimal("500000.00"),
            total_deductions=Decimal("60000.00"),
            net_income=Decimal("440000.00"),
            total_tax=Decimal("21500.00"),
            effective_tax_rate=Decimal("0.0430"),
            tax_due_or_refund=Decimal("21500.00"),
        )
        test_session.add(filing)
        test_session.commit()
        test_session.refresh(filing)
        assert filing.withholding_tax_paid == Decimal("0")
        assert filing.filing_status == "draft"

    def test_tax_filing_relationship_to_user(
        self, test_session: Session, sample_user: User
    ) -> None:
        """Test that filing links back to its user."""
        filing = TaxFiling(
            user_id=sample_user.id,
            tax_year=2024,
            gross_income=Decimal("500000.00"),
            total_deductions=Decimal("60000.00"),
            net_income=Decimal("440000.00"),
            total_tax=Decimal("21500.00"),
            effective_tax_rate=Decimal("0.0430"),
            tax_due_or_refund=Decimal("21500.00"),
        )
        test_session.add(filing)
        test_session.commit()
        test_session.refresh(filing)
        assert filing.user.id == sample_user.id
        assert filing in sample_user.tax_filings

    def test_tax_breakdown_json_roundtrip(self, test_session: Session, sample_user: User) -> None:
        """Test that JSON tax breakdown stores and retrieves correctly."""
        breakdown_data = [{"bracket": "0-150000", "rate": "0%", "tax": "0"}]
        filing = TaxFiling(
            user_id=sample_user.id,
            tax_year=2023,
            gross_income=Decimal("100000.00"),
            total_deductions=Decimal("60000.00"),
            net_income=Decimal("40000.00"),
            total_tax=Decimal("0.00"),
            effective_tax_rate=Decimal("0.0000"),
            tax_due_or_refund=Decimal("0.00"),
            tax_breakdown_json=json.dumps(breakdown_data),
        )
        test_session.add(filing)
        test_session.commit()
        test_session.refresh(filing)
        loaded = json.loads(filing.tax_breakdown_json)  # type: ignore[arg-type]
        assert loaded == breakdown_data
