"""Tests for finance_ai.tools.receipt_ocr (general document OCR + categorization)."""

from datetime import date
from decimal import Decimal
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from finance_ai.tools.receipt_ocr import (
    ReceiptOcrResult,
    _build_vision_message,
    _parse_ocr_json_response,
    _preprocess_image,
    _shrink_image_bytes,
    build_document_ocr_prompt,
    extract_document_fields,
)


def _chat_model_returning(content: str) -> Any:
    """Build a mock BaseChatModel whose invoke returns content."""
    mock = MagicMock(spec=BaseChatModel)
    mock.invoke.return_value = AIMessage(content=content)
    return mock


class TestBuildDocumentOcrPrompt:
    """Tests for build_document_ocr_prompt."""

    def test_prompt_mentions_supported_documents(self) -> None:
        """Prompt covers receipts, payslips, invoices, transfer slips."""
        prompt = build_document_ocr_prompt()
        assert "ใบเสร็จ" in prompt
        assert "สลิปเงินเดือน" in prompt
        assert "ใบกำกับภาษี" in prompt
        assert "สลิปโอนเงิน" in prompt

    def test_prompt_requests_json_array(self) -> None:
        """Prompt asks for a JSON array output."""
        prompt = build_document_ocr_prompt()
        assert "JSON" in prompt
        assert "array" in prompt.lower() or "อาร์เรย์" in prompt

    def test_prompt_lists_required_fields(self) -> None:
        """Prompt lists the required fields."""
        prompt = build_document_ocr_prompt()
        assert "transaction_type" in prompt
        assert "amount" in prompt
        assert "category" in prompt
        assert "transaction_date" in prompt


class TestExtractDocumentFields:
    """Tests for extract_document_fields."""

    def test_single_expense_receipt(self) -> None:
        """A single expense receipt yields one expense draft."""
        json_text = (
            '[{"transaction_type":"expense","amount":"350.00",'
            '"transaction_date":"2026-03-01","description":"ร้านอาหารสุกี้",'
            '"category":"food","confidence":0.9}]'
        )
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning(json_text),
        )
        assert len(drafts) == 1
        draft = drafts[0]
        assert draft.transaction_type == "expense"
        assert draft.amount == Decimal("350.00")
        assert draft.transaction_date == date(2026, 3, 1)
        assert draft.category == "food"
        assert draft.description == "ร้านอาหารสุกี้"
        assert draft.confidence == pytest.approx(0.9)

    def test_payslip_yields_income_with_metadata(self) -> None:
        """A payslip yields an income draft with WHT and employer."""
        json_text = (
            '[{"transaction_type":"income","amount":"50000.00",'
            '"transaction_date":"2026-03-31","description":"เงินเดือนมีนาคม",'
            '"category":"other","income_type":"salary",'
            '"withholding_tax":"2500.00","employer_name":"ABC จำกัด",'
            '"confidence":0.95}]'
        )
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning(json_text),
        )
        assert len(drafts) == 1
        draft = drafts[0]
        assert draft.transaction_type == "income"
        assert draft.income_type == "salary"
        assert draft.withholding_tax == Decimal("2500.00")
        assert draft.employer_name == "ABC จำกัด"

    def test_multi_item_receipt_yields_list(self) -> None:
        """A receipt with several items yields multiple drafts."""
        json_text = (
            '[{"transaction_type":"expense","amount":"80.00",'
            '"transaction_date":"2026-03-01","description":"กาแฟ",'
            '"category":"food","confidence":0.9},'
            '{"transaction_type":"expense","amount":"200.00",'
            '"transaction_date":"2026-03-01","description":"Grab",'
            '"category":"transport","confidence":0.8}]'
        )
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning(json_text),
        )
        assert len(drafts) == 2
        assert drafts[0].category == "food"
        assert drafts[1].category == "transport"

    def test_empty_model_response_returns_empty_list(self) -> None:
        """An empty/unreadable image yields an empty draft list."""
        drafts = extract_document_fields(
            b"",
            "image/png",
            _chat_model_returning(""),
        )
        assert drafts == []

    def test_garbage_json_returns_empty_list(self) -> None:
        """Non-JSON model output yields an empty list (no raise)."""
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning("sorry I cannot read this"),
        )
        assert drafts == []

    def test_non_array_json_returns_empty_list(self) -> None:
        """Valid JSON that is not an array yields an empty list."""
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning('{"transaction_type": "expense"}'),
        )
        assert drafts == []

    def test_non_dict_array_item_dropped(self) -> None:
        """Array items that are not objects are dropped."""
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning('["not-an-object", 42]'),
        )
        assert drafts == []

    def test_invalid_amount_dropped(self) -> None:
        """A row whose amount cannot parse is dropped, others kept."""
        json_text = (
            '[{"transaction_type":"expense","amount":"not-a-number",'
            '"transaction_date":"2026-03-01","description":"bad","category":"food"},'
            '{"transaction_type":"expense","amount":"100.00",'
            '"transaction_date":"2026-03-01","description":"good","category":"food"}]'
        )
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning(json_text),
        )
        assert len(drafts) == 1
        assert drafts[0].description == "good"

    def test_buddhist_era_date_converted(self) -> None:
        """Buddhist-era year (2569) is converted to CE (2026)."""
        json_text = (
            '[{"transaction_type":"expense","amount":"350.00",'
            '"transaction_date":"12/03/2569","description":"ร้านอาหาร",'
            '"category":"food"}]'
        )
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning(json_text),
        )
        assert drafts[0].transaction_date == date(2026, 3, 12)

    def test_unknown_category_falls_back_to_description(self) -> None:
        """An out-of-set category is reclassified from the description."""
        json_text = (
            '[{"transaction_type":"expense","amount":"350.00",'
            '"transaction_date":"2026-03-01","description":"ร้านอาหาร",'
            '"category":"banana"}]'
        )
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning(json_text),
        )
        assert drafts[0].category == "food"

    def test_invalid_transaction_type_falls_back_to_expense(self) -> None:
        """An invalid transaction_type defaults to expense for positive amount."""
        json_text = (
            '[{"transaction_type":"maybe","amount":"350.00",'
            '"transaction_date":"2026-03-01","description":"ร้านอาหาร",'
            '"category":"food"}]'
        )
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning(json_text),
        )
        assert drafts[0].transaction_type == "expense"

    def test_income_description_detected_when_type_missing(self) -> None:
        """Missing type but salary-like description becomes income."""
        json_text = (
            '[{"transaction_type":"","amount":"50000.00",'
            '"transaction_date":"2026-03-31","description":"Salary March",'
            '"category":"other"}]'
        )
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning(json_text),
        )
        assert drafts[0].transaction_type == "income"

    def test_markdown_fenced_json_parsed(self) -> None:
        """JSON wrapped in markdown code fences is still parsed."""
        json_text = (
            '```json\n[{"transaction_type":"expense","amount":"350.00",'
            '"transaction_date":"2026-03-01","description":"ร้านอาหาร",'
            '"category":"food"}]\n```'
        )
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning(json_text),
        )
        assert len(drafts) == 1
        assert drafts[0].amount == Decimal("350.00")

    def test_dict_wrapped_array_extracts_drafts(self) -> None:
        """A dict-wrapped array (JSON-mode shape) still yields drafts."""
        json_text = (
            '{"transactions": [{"transaction_type":"expense","amount":"350.00",'
            '"transaction_date":"2026-03-01","description":"ร้านอาหาร",'
            '"category":"food","confidence":0.9}]}'
        )
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning(json_text),
        )
        assert len(drafts) == 1
        assert drafts[0].amount == Decimal("350.00")

    def test_invalid_withholding_tax_defaults_to_zero(self) -> None:
        """An unparseable withholding_tax falls back to 0 for income."""
        json_text = (
            '[{"transaction_type":"income","amount":"50000.00",'
            '"transaction_date":"2026-03-31","description":"เงินเดือน",'
            '"category":"other","withholding_tax":"abc"}]'
        )
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            _chat_model_returning(json_text),
        )
        assert drafts[0].withholding_tax == Decimal("0")

    def test_unsupported_mime_type_raises(self) -> None:
        """A non-image, non-pdf mime type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            extract_document_fields(
                b"data",
                "text/plain",
                _chat_model_returning("[]"),
            )


class TestParseOcrJsonResponse:
    """Tests for _parse_ocr_json_response tolerant parsing."""

    def test_top_level_array_returned(self) -> None:
        """A top-level JSON array is returned as-is."""
        items = _parse_ocr_json_response('[{"transaction_type": "expense", "amount": "350"}]')
        assert items == [{"transaction_type": "expense", "amount": "350"}]

    def test_dict_wrapped_array_unwrapped(self) -> None:
        """A dict wrapping the array under a common key is unwrapped."""
        items = _parse_ocr_json_response(
            '{"transactions": [{"transaction_type": "expense", "amount": "350"}]}'
        )
        assert items == [{"transaction_type": "expense", "amount": "350"}]

    def test_numbered_object_shape_unwrapped(self) -> None:
        """A numbered-object shape (JSON-mode output) is unwrapped."""
        items = _parse_ocr_json_response(
            '{"0": {"transaction_type": "expense", "amount": "350"},'
            ' "1": {"transaction_type": "expense", "amount": "80"}}'
        )
        assert len(items) == 2
        assert items[0]["amount"] == "350"
        assert items[1]["amount"] == "80"

    def test_single_item_dict_returned(self) -> None:
        """A single item dict with an amount key is returned as one item."""
        items = _parse_ocr_json_response('{"transaction_type": "expense", "amount": "350"}')
        assert items == [{"transaction_type": "expense", "amount": "350"}]

    def test_bare_fence_returns_empty(self) -> None:
        """A bare code fence with no newline returns empty (no IndexError)."""
        assert _parse_ocr_json_response("```") == []
        assert _parse_ocr_json_response("```json") == []

    def test_non_json_returns_empty(self) -> None:
        """Non-JSON text returns an empty list."""
        assert _parse_ocr_json_response("sorry, no data") == []

    def test_non_dict_items_dropped(self) -> None:
        """Non-dict array items are dropped."""
        assert _parse_ocr_json_response('["a", 1]') == []


class TestReceiptOcrResultModelDefaults:
    """Tests for the ReceiptOcrResult Pydantic model defaults."""

    def test_defaults_for_expense(self) -> None:
        """Expense draft has sane defaults for income-only fields."""
        result = ReceiptOcrResult(
            transaction_type="expense",
            amount=Decimal("100"),
            transaction_date=date(2026, 1, 1),
            description="test",
            category="other",
        )
        assert result.income_type == "other"
        assert result.withholding_tax == Decimal("0")
        assert result.employer_name is None
        assert result.confidence == 0.0


class TestBuildVisionMessage:
    """Tests for multimodal message construction."""

    def _model_named(self, name: str) -> Any:
        """Return a BaseChatModel mock with the given class name."""
        mock = MagicMock(spec=BaseChatModel)
        mock.__class__.__name__ = name
        return mock

    def _first_part(self, message: HumanMessage) -> Any:
        """Return the first content part of a HumanMessage as Any."""
        return cast(list[Any], message.content)[0]

    def test_ollama_uses_image_url_data_block(self) -> None:
        """Ollama gets a base64 data URL in an image_url block."""
        message = _build_vision_message(b"\x89PNGfake", "image/png")
        image_part = self._first_part(message)
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_google_uses_image_url_data_block(self) -> None:
        """Google Gemini also gets a base64 data URL in an image_url block."""
        message = _build_vision_message(b"\x89PNGfake", "image/png")
        image_part = self._first_part(message)
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_openai_style_model_uses_image_url(self) -> None:
        """OpenAI-compatible models also get an image_url data block."""
        message = _build_vision_message(b"\x89PNGfake", "image/png")
        image_part = self._first_part(message)
        assert image_part["type"] == "image_url"
        assert "data:image/png;base64," in image_part["image_url"]["url"]

    def test_prompt_is_included_as_text_part(self) -> None:
        """The message always contains the OCR prompt as a text block."""
        message = _build_vision_message(b"\x89PNGfake", "image/png")
        text_part = cast(list[Any], message.content)[1]
        assert text_part["type"] == "text"
        assert "JSON" in text_part["text"]

    def test_ollama_invokes_without_json_format(self) -> None:
        """Ollama models are no longer forced into JSON mode."""
        model = self._model_named("ChatOllama")
        model.invoke.return_value = AIMessage(content="[]")
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            model,
        )
        assert drafts == []
        model.invoke.assert_called_once()
        call_args = model.invoke.call_args
        assert "format" not in call_args.kwargs

    def test_google_invokes_without_json_format(self) -> None:
        """Google models are not forced into JSON mode."""
        model = self._model_named("ChatGoogleGenerativeAI")
        model.invoke.return_value = AIMessage(content="[]")
        drafts = extract_document_fields(
            b"\x89PNGfake",
            "image/png",
            model,
        )
        assert drafts == []
        call_args = model.invoke.call_args
        assert "format" not in call_args.kwargs


def _make_png_bytes(size: int) -> bytes:
    """Build a real PNG of the given square size using Pillow."""
    import io
    from PIL import Image

    img = Image.new("RGB", (size, size), color=(255, 0, 0))
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


class TestPreprocessImage:
    """Tests for image preprocessing before OCR."""

    def test_large_png_is_downscaled_to_jpeg(self) -> None:
        """A 3000px PNG is shrunk below max_edge and re-encoded as JPEG."""
        big = _make_png_bytes(3000)
        processed, mime = _preprocess_image(big, "image/png", max_edge=1568)
        assert mime == "image/jpeg"
        import io
        from PIL import Image

        with Image.open(io.BytesIO(processed)) as img:
            assert max(img.size) <= 1568
            assert img.format == "JPEG"

    def test_small_png_still_reencoded_as_jpeg(self) -> None:
        """A small image is kept within bounds and re-encoded as JPEG."""
        small = _make_png_bytes(64)
        processed, mime = _preprocess_image(small, "image/png", max_edge=1568)
        assert mime == "image/jpeg"
        assert len(processed) > 0

    def test_pdf_bytes_passed_through_unchanged(self) -> None:
        """PDFs are not processed by Pillow and returned as-is."""
        payload = b"%PDF-1.4 fake pdf bytes"
        processed, mime = _preprocess_image(payload, "application/pdf", max_edge=1568)
        assert processed == payload
        assert mime == "application/pdf"

    def test_unreadable_bytes_fall_back_to_original(self) -> None:
        """Bytes Pillow cannot decode are returned unchanged (no hard failure)."""
        junk = b"\x89PNGfake not really an image"
        processed, mime = _shrink_image_bytes(junk, "image/png", max_edge=1568)
        assert processed == junk
        assert mime == "image/png"

    def test_large_png_reduces_byte_size(self) -> None:
        """Downscaling + JPEG re-encode shrinks the payload sent to the model."""
        big = _make_png_bytes(3000)
        processed, _ = _preprocess_image(big, "image/png", max_edge=1568)
        assert len(processed) < len(big)
