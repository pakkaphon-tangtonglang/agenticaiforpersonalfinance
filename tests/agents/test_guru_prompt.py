"""Tests for the guru system prompt builder."""

from finance_ai.agents.guru_personas import get_guru_persona
from finance_ai.agents.prompts import GURU_BASE_SYSTEM_PROMPT, build_guru_system_prompt


class TestBuildGuruSystemPrompt:
    """Tests for build_guru_system_prompt."""

    def test_includes_persona_name_and_principles(self) -> None:
        """The persona's name and every core principle appear in the prompt."""
        persona = get_guru_persona("buffett")

        prompt = build_guru_system_prompt(persona)

        assert "Warren Buffett" in prompt
        for principle in persona.core_principles:
            assert principle in prompt

    def test_includes_avoid_topics(self) -> None:
        """Topics the guru avoids are listed so the model can decline them."""
        persona = get_guru_persona("dalio")

        prompt = build_guru_system_prompt(persona)

        for topic in persona.avoid:
            assert topic in prompt

    def test_starts_with_shared_safety_rules(self) -> None:
        """Every guru prompt carries the shared third-person and disclaimer rules."""
        persona = get_guru_persona("niwes")

        prompt = build_guru_system_prompt(persona)

        assert prompt.startswith(GURU_BASE_SYSTEM_PROMPT)

    def test_base_prompt_forbids_impersonation_and_requires_confirmation(self) -> None:
        """The shared rules cover impersonation, citations and write confirmation."""
        assert "ห้ามอ้างว่าเป็นบุคคลจริง" in GURU_BASE_SYSTEM_PROMPT
        assert "search_guru_knowledge" in GURU_BASE_SYSTEM_PROMPT
        assert "get_guru_scorecard" in GURU_BASE_SYSTEM_PROMPT
        assert "ยืนยัน" in GURU_BASE_SYSTEM_PROMPT
