"""Tests for the guru persona registry."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from finance_ai.agents.guru_personas import (
    GuruPersona,
    UnknownGuruError,
    get_guru_persona,
    load_guru_persona_file,
    load_guru_personas,
)

VALID_PROFILE_YAML = """\
guru_id: tester
display_name: Test Guru
region: US
language: en
tagline: Buy wonderful businesses at fair prices
core_principles:
  - Stay within your circle of competence
focus_areas: [stocks]
tone_rules: [plain language]
typical_questions:
  - Should I buy index funds?
avoid: [day trading tips]
profile_as_of: 2026-10-01
sources:
  - title: Shareholder letter
    url: https://example.com/letter
"""


def _write_profile(directory: Path, guru_id: str, content: str) -> Path:
    """Write a profile.yaml under directory/guru_id and return its path."""
    guru_directory = directory / guru_id
    guru_directory.mkdir(parents=True)
    profile_path = guru_directory / "profile.yaml"
    profile_path.write_text(content, encoding="utf-8")
    return profile_path


class TestLoadGuruPersonaFile:
    """Tests for load_guru_persona_file."""

    def test_valid_profile_parses_all_fields(self, tmp_path: Path) -> None:
        """A complete profile becomes a GuruPersona."""
        profile_path = _write_profile(tmp_path, "tester", VALID_PROFILE_YAML)

        persona = load_guru_persona_file(profile_path)

        assert persona.guru_id == "tester"
        assert persona.region == "US"
        assert persona.profile_as_of == date(2026, 10, 1)
        assert persona.sources[0].url == "https://example.com/letter"

    def test_missing_principles_raises_validation_error(self, tmp_path: Path) -> None:
        """A profile without core principles is rejected."""
        broken_yaml = VALID_PROFILE_YAML.replace(
            "core_principles:\n  - Stay within your circle of competence\n",
            "core_principles: []\n",
        )
        profile_path = _write_profile(tmp_path, "tester", broken_yaml)

        with pytest.raises(ValidationError):
            load_guru_persona_file(profile_path)

    def test_non_mapping_yaml_raises_value_error(self, tmp_path: Path) -> None:
        """A YAML list instead of a mapping gives a clear error."""
        profile_path = _write_profile(tmp_path, "tester", "- just\n- a list\n")

        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_guru_persona_file(profile_path)


class TestLoadGuruPersonas:
    """Tests for load_guru_personas."""

    def test_loads_every_profile_keyed_by_guru_id(self, tmp_path: Path) -> None:
        """Each subfolder with a profile.yaml is loaded."""
        _write_profile(tmp_path, "tester", VALID_PROFILE_YAML)
        _write_profile(tmp_path, "other", VALID_PROFILE_YAML.replace("tester", "other"))

        personas = load_guru_personas(tmp_path)

        assert sorted(personas) == ["other", "tester"]

    def test_folder_name_mismatch_raises_value_error(self, tmp_path: Path) -> None:
        """guru_id must match its folder so URLs and files stay in sync."""
        _write_profile(tmp_path, "wrong_folder", VALID_PROFILE_YAML)

        with pytest.raises(ValueError, match="does not match folder"):
            load_guru_personas(tmp_path)

    def test_missing_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        """A missing profiles directory fails loudly."""
        with pytest.raises(FileNotFoundError, match="Guru profiles directory not found"):
            load_guru_personas(tmp_path / "missing")


class TestGetGuruPersona:
    """Tests for get_guru_persona."""

    def test_returns_requested_persona(self, tmp_path: Path) -> None:
        """A known guru_id returns its persona."""
        _write_profile(tmp_path, "tester", VALID_PROFILE_YAML)
        personas = load_guru_personas(tmp_path)

        persona = get_guru_persona("tester", personas)

        assert isinstance(persona, GuruPersona)
        assert persona.display_name == "Test Guru"

    def test_unknown_guru_lists_available_ids(self, tmp_path: Path) -> None:
        """An unknown guru_id names the valid choices."""
        _write_profile(tmp_path, "tester", VALID_PROFILE_YAML)
        personas = load_guru_personas(tmp_path)

        with pytest.raises(UnknownGuruError, match="Unknown guru: 'nobody'. Available: tester"):
            get_guru_persona("nobody", personas)


class TestShippedProfiles:
    """The profiles committed in docs/gurus must all load."""

    def test_all_four_gurus_are_present(self) -> None:
        """Buffett, Dalio, ดร.นิเวศน์ and โค้ชหนุ่ม ship with the repo."""
        personas = load_guru_personas(Path("docs/gurus"))

        assert set(personas) == {"buffett", "dalio", "niwes", "coach_noom"}
