"""Tests for agent response accuracy evaluator."""

from decimal import Decimal
from unittest.mock import MagicMock, patch

from finance_ai.evaluation.accuracy_evaluator import (
    EVAL_TAX_TOOLS,
    _build_eval_tax_graph,
    compute_tax_ground_truth,
    evaluate_single_tax_case,
    evaluate_single_tax_case_forced,
    evaluate_tax_accuracy_dataset,
    extract_tax_from_response,
)
from finance_ai.evaluation.models import TaxAccuracyCase, TaxAccuracyDataset


def _make_tax_case(
    gross_income: str = "600000",
    expected_tax: str = "21500",
) -> TaxAccuracyCase:
    """Create a sample tax accuracy case."""
    return TaxAccuracyCase(
        case_id="tax_001",
        query="คำนวณภาษี",
        gross_income=Decimal(gross_income),
        deductions_by_type={"personal_allowance": Decimal("60000")},
        expected_total_tax=Decimal(expected_tax),
        expected_effective_rate=Decimal("0.0358"),
        tolerance_thb=Decimal("100"),
    )


class TestComputeTaxGroundTruth:
    """Tests for compute_tax_ground_truth."""

    def test_returns_calculation_result(self) -> None:
        """Test that ground truth computation returns valid result."""
        case = _make_tax_case()
        result = compute_tax_ground_truth(case)
        assert result.gross_income == Decimal("600000")
        assert result.total_tax > Decimal("0")

    def test_zero_income(self) -> None:
        """Test ground truth for zero income."""
        case = TaxAccuracyCase(
            case_id="tax_zero",
            query="test",
            gross_income=Decimal("0"),
            expected_total_tax=Decimal("0"),
            expected_effective_rate=Decimal("0"),
        )
        result = compute_tax_ground_truth(case)
        assert result.total_tax == Decimal("0")


class TestExtractTaxFromResponse:
    """Tests for extract_tax_from_response."""

    def test_simple_extraction(self) -> None:
        """Test extracting tax from a simple response."""
        result = extract_tax_from_response("ภาษีที่ต้องจ่าย 29,000 บาท")
        assert result == Decimal("29000")

    def test_extraction_with_context(self) -> None:
        """Test extracting tax from a longer response."""
        text = "คำนวณภาษีเงินได้แล้ว ภาษีรวม 145,500 บาท สำหรับปี 2024"
        result = extract_tax_from_response(text)
        assert result == Decimal("145500")

    def test_no_match_returns_none(self) -> None:
        """Test that non-matching text returns None."""
        result = extract_tax_from_response("ไม่มีข้อมูลภาษี")
        assert result is None

    def test_decimal_amount(self) -> None:
        """Test extracting a decimal tax amount."""
        result = extract_tax_from_response("ภาษีที่ต้องจ่าย 29,000.50 บาท")
        assert result == Decimal("29000.50")

    def test_markdown_bold_format(self) -> None:
        """Test extracting tax from markdown bold response."""
        text = "**ภาษีที่ต้องจ่าย:** 29,000 บาท"
        result = extract_tax_from_response(text)
        assert result == Decimal("29000")

    def test_markdown_bold_with_stars(self) -> None:
        """Test extracting from bold with extra asterisks."""
        text = "**ภาษีที่ต้องชำระ:** 33,500 บาท"
        result = extract_tax_from_response(text)
        assert result == Decimal("33500")

    def test_ruam_pasi_pattern(self) -> None:
        """Test extracting from รวมภาษี pattern."""
        text = "**รวมภาษีที่ต้องจ่ายทั้งสิ้น:** 103,000 บาท"
        result = extract_tax_from_response(text)
        assert result == Decimal("103000")

    def test_zero_tax_not_confused_with_income(self) -> None:
        """Test that 0 tax is not confused with income numbers."""
        text = (
            "**รายได้รวม:** 150,000 บาท\n"
            "**เงินได้สุทธิ:** 90,000 บาท\n"
            "**ภาษีที่ต้องจ่าย:** 0 บาท"
        )
        result = extract_tax_from_response(text)
        assert result == Decimal("0")

    def test_income_not_extracted(self) -> None:
        """Test that income amounts are not extracted as tax."""
        text = "**รายได้รวม:** 300,000 บาท\n" "**ภาษีที่ต้องจ่าย:** 4,500 บาท"
        result = extract_tax_from_response(text)
        assert result == Decimal("4500")

    def test_tang_mot_pattern(self) -> None:
        """Test extracting from ภาษีที่ต้องจ่ายทั้งหมด pattern."""
        text = "**ภาษีที่ต้องจ่ายทั้งหมด:** 33,500 บาท"
        result = extract_tax_from_response(text)
        assert result == Decimal("33500")

    def test_chamra_tang_mot_pattern(self) -> None:
        """Test extracting from ภาษีที่ต้องชำระทั้งหมด pattern."""
        text = "**ภาษีที่ต้องชำระทั้งหมด:** 89,000 บาท"
        result = extract_tax_from_response(text)
        assert result == Decimal("89000")

    def test_zero_tax_mai_tong_sia(self) -> None:
        """Test zero tax detection: ไม่ต้องเสียภาษี."""
        text = "เงินได้สุทธิ 15,000 บาท ซึ่งคุณไม่ต้องเสียภาษีในปีนี้"
        result = extract_tax_from_response(text)
        assert result == Decimal("0")

    def test_zero_tax_mai_tong_jai(self) -> None:
        """Test zero tax detection: ไม่ต้องจ่ายภาษี."""
        text = "สรุป: คุณไม่ต้องจ่ายภาษี เพราะเงินได้สุทธิต่ำกว่า 150,000 บาท"
        result = extract_tax_from_response(text)
        assert result == Decimal("0")

    def test_zero_tax_mai_tong_chamra(self) -> None:
        """Test zero tax detection: ไม่ต้องชำระภาษี."""
        text = "คุณไม่ต้องชำระภาษีในปีนี้ค่ะ"
        result = extract_tax_from_response(text)
        assert result == Decimal("0")

    def test_greedy_pattern_does_not_cross_newline(self) -> None:
        """Test that pattern does not match numbers across newlines.

        Case 006 regression: 'สรุปภาษีที่ต้องจ่าย:\\n...15,000 บาท'
        should NOT extract 15,000 as tax.
        """
        text = (
            "**สรุปภาษีที่ต้องจ่าย:**\n"
            "คุณมีเงินได้สุทธิ 15,000 บาท\n"
            "**ภาษีที่ต้องจ่าย:** 0.00 บาท\n"
            "คุณไม่ต้องเสียภาษีในปีนี้"
        )
        result = extract_tax_from_response(text)
        assert result == Decimal("0")

    def test_case_011_regression(self) -> None:
        """Case 011 regression: income 100k, no deductions, tax=0.

        Should not extract 50,000 (expense deduction) as tax.
        """
        text = (
            "**รายได้รวม:** 100,000 บาท\n"
            "**ค่าใช้จ่าย:** 50,000 บาท\n"
            "**เงินได้สุทธิ:** 50,000 บาท\n"
            "**ดังนั้น คุณไม่ต้องเสียภาษีในปีนี้**"
        )
        result = extract_tax_from_response(text)
        assert result == Decimal("0")

    def test_pasi_ngoen_dai_pattern(self) -> None:
        """Test ภาษีเงินได้ pattern."""
        text = "ภาษีเงินได้: 145,000 บาท"
        result = extract_tax_from_response(text)
        assert result == Decimal("145000")

    def test_generic_pasi_pattern(self) -> None:
        """Test generic ภาษี: pattern as last resort."""
        text = "ภาษี: 8,500 บาท"
        result = extract_tax_from_response(text)
        assert result == Decimal("8500")


class TestEvaluateSingleTaxCase:
    """Tests for evaluate_single_tax_case."""

    @patch("finance_ai.evaluation.accuracy_evaluator.execute_tax_agent")
    def test_within_tolerance(self, mock_execute: MagicMock) -> None:
        """Test case where agent answer is within tolerance."""
        mock_execute.return_value = {
            "intent": "tax",
            "response": "ภาษีที่ต้องจ่าย 21,550 บาท",
        }
        model = MagicMock()
        case = _make_tax_case()
        result = evaluate_single_tax_case(model, case)
        assert result.is_within_tolerance is True
        assert result.absolute_error_thb == Decimal("50")

    @patch("finance_ai.evaluation.accuracy_evaluator.execute_tax_agent")
    def test_outside_tolerance(self, mock_execute: MagicMock) -> None:
        """Test case where agent answer exceeds tolerance."""
        mock_execute.return_value = {
            "intent": "tax",
            "response": "ภาษีที่ต้องจ่าย 25,000 บาท",
        }
        model = MagicMock()
        case = _make_tax_case()
        result = evaluate_single_tax_case(model, case)
        assert result.is_within_tolerance is False
        assert result.absolute_error_thb == Decimal("3500")

    @patch("finance_ai.evaluation.accuracy_evaluator.execute_tax_agent")
    def test_extraction_fails(self, mock_execute: MagicMock) -> None:
        """Test case where tax extraction fails."""
        mock_execute.return_value = {
            "intent": "tax",
            "response": "ไม่สามารถคำนวณได้",
        }
        model = MagicMock()
        case = _make_tax_case()
        result = evaluate_single_tax_case(model, case)
        assert result.is_within_tolerance is False
        assert result.extracted_total_tax is None

    @patch("finance_ai.evaluation.accuracy_evaluator.execute_tax_agent")
    def test_latency_recorded(self, mock_execute: MagicMock) -> None:
        """Test that latency is recorded."""
        mock_execute.return_value = {
            "intent": "tax",
            "response": "ภาษีที่ต้องจ่าย 21,500 บาท",
        }
        model = MagicMock()
        case = _make_tax_case()
        result = evaluate_single_tax_case(model, case)
        assert result.latency_seconds >= 0.0


class TestEvaluateTaxAccuracyDataset:
    """Tests for evaluate_tax_accuracy_dataset."""

    @patch("finance_ai.evaluation.accuracy_evaluator.execute_tax_agent")
    def test_aggregate_results(self, mock_execute: MagicMock) -> None:
        """Test dataset aggregation."""
        mock_execute.return_value = {
            "intent": "tax",
            "response": "ภาษีที่ต้องจ่าย 21,500 บาท",
        }
        model = MagicMock()
        dataset = TaxAccuracyDataset(
            version="1.0",
            cases=[_make_tax_case(), _make_tax_case()],
        )
        agg = evaluate_tax_accuracy_dataset(model, dataset)
        assert agg.total_cases == 2
        assert agg.agent_type == "tax"
        assert len(agg.results) == 2

    @patch("finance_ai.evaluation.accuracy_evaluator.execute_tax_agent")
    def test_empty_dataset(self, mock_execute: MagicMock) -> None:
        """Test with empty dataset."""
        model = MagicMock()
        dataset = TaxAccuracyDataset(version="1.0", cases=[])
        agg = evaluate_tax_accuracy_dataset(model, dataset)
        assert agg.total_cases == 0
        assert agg.within_tolerance_count == 0


class TestBuildEvalTaxGraph:
    """Tests for the eval-only forced tool graph."""

    def test_binds_only_calculate_thai_tax(self) -> None:
        """Eval graph binds only calculate_thai_tax, not all tools."""
        mock_model = MagicMock()
        _build_eval_tax_graph(mock_model)
        first_call = mock_model.bind_tools.call_args_list[0]
        assert first_call[0][0] == EVAL_TAX_TOOLS
        assert first_call[1]["tool_choice"] == "any"

    def test_respond_binds_without_force(self) -> None:
        """Respond node binds tools without forcing."""
        mock_model = MagicMock()
        _build_eval_tax_graph(mock_model)
        second_call = mock_model.bind_tools.call_args_list[1]
        assert second_call[0][0] == EVAL_TAX_TOOLS
        assert "tool_choice" not in second_call[1]


class TestEvaluateSingleTaxCaseForced:
    """Tests for evaluate_single_tax_case_forced."""

    @patch("finance_ai.evaluation.accuracy_evaluator._execute_eval_tax_agent")
    def test_forced_within_tolerance(self, mock_exec: MagicMock) -> None:
        """Test forced mode with correct answer."""
        mock_exec.return_value = "ภาษีที่ต้องจ่าย 21,500 บาท"
        model = MagicMock()
        case = _make_tax_case()
        result = evaluate_single_tax_case_forced(model, case)
        assert result.is_within_tolerance is True
        assert result.extracted_total_tax == Decimal("21500")

    @patch("finance_ai.evaluation.accuracy_evaluator._execute_eval_tax_agent")
    def test_forced_extraction_fails(self, mock_exec: MagicMock) -> None:
        """Test forced mode when extraction fails."""
        mock_exec.return_value = "ไม่สามารถคำนวณได้"
        model = MagicMock()
        case = _make_tax_case()
        result = evaluate_single_tax_case_forced(model, case)
        assert result.is_within_tolerance is False
        assert result.extracted_total_tax is None


class TestExpenseDeductionIntegration:
    """Tests verifying expense deduction is applied in evaluation."""

    def test_ground_truth_includes_expense_deduction(self) -> None:
        """Ground truth result includes expense_deduction field."""
        case = _make_tax_case()
        result = compute_tax_ground_truth(case)
        assert result.expense_deduction == Decimal("100000")

    def test_ground_truth_expense_deduction_at_cap(self) -> None:
        """High income gets max 100,000 expense deduction."""
        case = _make_tax_case(gross_income="1200000", expected_tax="83000")
        case.deductions_by_type = {
            "personal_allowance": Decimal("60000"),
            "rmf": Decimal("200000"),
        }
        result = compute_tax_ground_truth(case)
        assert result.expense_deduction == Decimal("100000")
        assert result.total_tax == Decimal("83000.00")

    def test_ground_truth_expense_deduction_below_cap(self) -> None:
        """Low income gets 50% expense deduction (below 100k cap)."""
        case = _make_tax_case(gross_income="150000", expected_tax="0")
        result = compute_tax_ground_truth(case)
        assert result.expense_deduction == Decimal("75000.00")
        assert result.total_tax == Decimal("0.0000")

    def test_extract_tax_with_expense_deduction_format(self) -> None:
        """Extract tax from response that mentions expense deduction."""
        text = (
            "**ค่าใช้จ่าย:** 100,000 บาท\n"
            "**ค่าลดหย่อนรวม:** 60,000 บาท\n"
            "**ภาษีที่ต้องจ่าย:** 21,500 บาท"
        )
        result = extract_tax_from_response(text)
        assert result == Decimal("21500")

    @patch("finance_ai.evaluation.accuracy_evaluator.execute_tax_agent")
    def test_evaluate_case_correct_after_expense_deduction(
        self,
        mock_execute: MagicMock,
    ) -> None:
        """End-to-end case is correct with expense deduction applied."""
        mock_execute.return_value = {
            "intent": "tax",
            "response": "ภาษีที่ต้องจ่าย 21,500 บาท",
        }
        model = MagicMock()
        case = _make_tax_case()
        result = evaluate_single_tax_case(model, case)
        assert result.is_within_tolerance is True
        assert result.absolute_error_thb == Decimal("0")

    @patch("finance_ai.evaluation.accuracy_evaluator.execute_tax_agent")
    def test_evaluate_case_zero_tax_low_income(
        self,
        mock_execute: MagicMock,
    ) -> None:
        """Low income with expense deduction results in zero tax."""
        mock_execute.return_value = {
            "intent": "tax",
            "response": "ภาษีที่ต้องจ่าย 0 บาท",
        }
        model = MagicMock()
        case = _make_tax_case(gross_income="300000", expected_tax="0")
        result = evaluate_single_tax_case(model, case)
        assert result.is_within_tolerance is True

    @patch("finance_ai.evaluation.accuracy_evaluator._execute_eval_tax_agent")
    def test_forced_case_with_expense_deduction(
        self,
        mock_exec: MagicMock,
    ) -> None:
        """Forced mode correct with expense deduction applied."""
        mock_exec.return_value = "ภาษีที่ต้องจ่าย 21,500 บาท"
        model = MagicMock()
        case = _make_tax_case()
        result = evaluate_single_tax_case_forced(model, case)
        assert result.is_within_tolerance is True
        assert result.extracted_total_tax == Decimal("21500")

    def test_dataset_ground_truth_matches_expected(self) -> None:
        """All dataset cases have ground truth matching expected_total_tax."""
        cases = [
            _make_tax_case(gross_income="600000", expected_tax="21500"),
            _make_tax_case(gross_income="150000", expected_tax="0"),
        ]
        for case in cases:
            result = compute_tax_ground_truth(case)
            assert abs(result.total_tax - case.expected_total_tax) <= Decimal("1")

    @patch("finance_ai.evaluation.accuracy_evaluator._execute_eval_tax_agent")
    def test_forced_expense_deduction_shown_in_response(
        self,
        mock_exec: MagicMock,
    ) -> None:
        """Verify extraction works when response mentions ค่าใช้จ่าย."""
        mock_exec.return_value = "ค่าใช้จ่าย 100,000 บาท " "ภาษีที่ต้องชำระ 21,500 บาท"
        model = MagicMock()
        case = _make_tax_case()
        result = evaluate_single_tax_case_forced(model, case)
        assert result.extracted_total_tax == Decimal("21500")
        assert result.is_within_tolerance is True

    @patch("finance_ai.evaluation.accuracy_evaluator.execute_tax_agent")
    def test_aggregate_results_with_updated_values(
        self,
        mock_execute: MagicMock,
    ) -> None:
        """Aggregate still works correctly with updated expected values."""
        mock_execute.return_value = {
            "intent": "tax",
            "response": "ภาษีที่ต้องจ่าย 21,500 บาท",
        }
        model = MagicMock()
        dataset = TaxAccuracyDataset(
            version="1.0",
            cases=[_make_tax_case(), _make_tax_case()],
        )
        agg = evaluate_tax_accuracy_dataset(model, dataset)
        assert agg.total_cases == 2
        assert agg.within_tolerance_count == 2
        assert agg.accuracy_rate == Decimal("1.0000")
