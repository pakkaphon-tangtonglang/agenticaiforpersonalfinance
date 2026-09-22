"""Tests for orchestrator system prompt composition."""

from finance_ai.agents.prompts import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    build_orchestrator_system_prompt,
)


def _boundary_examples_from_prompts_module() -> tuple[str, ...]:
    """Return the few-shot example lines shipped in the production prompt.

    One boundary example per intent: expense, expense(balance),
    planning, asset_monitoring, recommendation, report, tax, general,
    unknown. Extracted verbatim from prompts.py (never retyped).
    """
    return (
        '- "จ่ายค่ากาแฟ 80 บาท" → {"intent": "expense", "confidence": 0.95}',
        '- "มีเงินออม 50,000 บาท" → {"intent": "expense", "confidence": 0.9}  (บันทึกยอดที่มีอยู่ '
        "ไม่ใช่การวางแผน)",
        '- "อยากออมเงิน 100,000 บาทภายในปีหน้า" → {"intent": "planning", "confidence": 0.9}  '
        "(ตั้งเป้าหมายใหม่)",
        '- "ดูราคา PTT ล่าสุด" → {"intent": "asset_monitoring", "confidence": 0.95}',
        '- "ควรลงทุนอะไรดี" → {"intent": "recommendation", "confidence": 0.85}  (ขอคำแนะนำ '
        "ไม่ใช่ดูราคา)",
        '- "สรุปสถานะการเงินทั้งหมดให้หน่อย" → {"intent": "report", "confidence": 0.9}',
        '- "ลดหย่อนภาษีด้วยประกันสุขภาพได้ไหม" → {"intent": "tax", "confidence": 0.95}',
        '- "ดอกเบี้ยเงินฝากออมทรัพย์คืออะไร" → {"intent": "general", "confidence": 0.85}  (คำถามความรู้ทั่วไป ไม่ต้องเรียก '
        "tool)",
        '- "สอนทำผัดกระเพราหน่อย" → {"intent": "unknown", "confidence": 0.9}',
    )


class TestBuildOrchestratorSystemPrompt:
    """Tests for building the router prompt with/without few-shot examples."""

    def test_with_examples_matches_production_prompt(self) -> None:
        """Prompt built with examples equals the production constant."""
        assert build_orchestrator_system_prompt(True) == ORCHESTRATOR_SYSTEM_PROMPT

    def test_without_examples_excludes_example_block(self) -> None:
        """Prompt built without examples contains no 'ตัวอย่าง:' block."""
        prompt = build_orchestrator_system_prompt(False)
        assert "ตัวอย่าง:" not in prompt
        assert "จ่ายค่ากาแฟ 80 บาท" not in prompt

    def test_without_examples_keeps_categories_and_json_rule(self) -> None:
        """Baseline prompt still has all intents and the JSON output rule."""
        prompt = build_orchestrator_system_prompt(False)
        assert '"tax"' in prompt
        assert "unknown" in prompt
        assert "ตอบเป็น JSON เท่านั้น" in prompt


class TestProductionPromptRegression:
    """Regression guards for the few-shot examples in the shipped prompt."""

    def test_production_prompt_contains_all_boundary_examples(self) -> None:
        """All 9 few-shot examples survive the refactor."""
        examples = _boundary_examples_from_prompts_module()
        for example in examples:
            assert example in ORCHESTRATOR_SYSTEM_PROMPT
