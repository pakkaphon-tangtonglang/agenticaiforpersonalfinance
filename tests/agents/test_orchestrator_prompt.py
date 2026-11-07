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


class TestVagueMessageRule:
    """Plan 3.1: one-word messages like 'เงิน' must score low so the router asks back."""

    def test_rule_in_head_survives_no_example_ablation(self) -> None:
        """The low-confidence rule is a classification rule, not only an example."""
        prompt = build_orchestrator_system_prompt(False)
        assert "กว้างเกินไป" in prompt
        assert "0.5" in prompt

    def test_production_prompt_has_vague_example(self) -> None:
        """The shipped prompt shows a vague one-word message with low confidence."""
        assert '"เงิน" → {"intent": "general", "confidence": 0.4}' in ORCHESTRATOR_SYSTEM_PROMPT


class TestAssetPromptPortfolioVsWatchlist:
    """The asset agent must not answer holdings questions from the watchlist."""

    def test_prompt_separates_holdings_from_watchlist(self) -> None:
        """The prompt names the portfolio tool and says holdings are not the watchlist."""
        from finance_ai.agents.prompts import (  # pylint: disable=import-outside-toplevel
            ASSET_MONITORING_AGENT_SYSTEM_PROMPT,
        )

        assert "get_portfolio_holdings" in ASSET_MONITORING_AGENT_SYSTEM_PROMPT
        assert "รายการถือหุ้น" in ASSET_MONITORING_AGENT_SYSTEM_PROMPT
        assert "ไม่ใช่ Watchlist" in ASSET_MONITORING_AGENT_SYSTEM_PROMPT


class TestRouterPromptFixes:
    """Fixes from the 4.11 evaluations (2026-11-07)."""

    def test_greeting_is_general_with_high_confidence(self) -> None:
        """A greeting must not fall under the 0.7 ask-back threshold."""
        assert (
            '- "หวัดดีครับ" → {"intent": "general", "confidence": 0.9}'
            in ORCHESTRATOR_SYSTEM_PROMPT
        )
        assert "คำทักทาย" in ORCHESTRATOR_SYSTEM_PROMPT

    def test_paid_premium_with_tax_question_goes_to_tax(self) -> None:
        """'Paid X, how much tax?' is a tax question, not an expense entry."""
        example = (
            '- "ปีนี้จ่ายเบี้ยประกันสุขภาพไป 12,000 แล้วต้องเสียภาษีเท่าไร" → '
            '{"intent": "tax", "confidence": 0.9}'
        )
        assert example in ORCHESTRATOR_SYSTEM_PROMPT
        assert "ถามว่าต้องเสียภาษีเท่าไร" in ORCHESTRATOR_SYSTEM_PROMPT
