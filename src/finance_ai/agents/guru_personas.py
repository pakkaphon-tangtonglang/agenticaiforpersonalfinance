"""Guru persona registry loaded from docs/gurus/<guru_id>/profile.yaml.

Each persona describes a public investor or money coach whose published
philosophy the Guru agent uses as a lens. The agent never claims to be the
real person; profiles only hold curated principles and their sources.
"""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

PROFILE_FILENAME = "profile.yaml"
DEFAULT_PROFILES_DIRECTORY = Path(__file__).resolve().parents[3] / "docs" / "gurus"


class UnknownGuruError(ValueError):
    """Raised when a guru_id is not in the registry."""


class GuruSource(BaseModel):
    """A public source a persona profile is based on."""

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)


class GuruPersona(BaseModel):
    """Curated profile of one guru's public money philosophy."""

    guru_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    display_name: str = Field(min_length=1)
    region: Literal["US", "TH"]
    language: Literal["en", "th"]
    tagline: str = Field(min_length=1)
    core_principles: list[str] = Field(min_length=1)
    focus_areas: list[str] = Field(default_factory=list)
    tone_rules: list[str] = Field(default_factory=list)
    typical_questions: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    profile_as_of: date
    sources: list[GuruSource] = Field(min_length=1)


def load_guru_persona_file(profile_path: Path) -> GuruPersona:
    """Parse and validate one profile.yaml file.

    Args:
        profile_path: Path to a profile.yaml file.

    Returns:
        The validated GuruPersona.

    Raises:
        ValueError: If the YAML is not a mapping.
        pydantic.ValidationError: If required fields are missing or invalid.

    Example:
        >>> load_guru_persona_file(Path("docs/gurus/buffett/profile.yaml")).guru_id
        'buffett'
    """
    raw_profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(raw_profile, dict):
        raise ValueError(f"Guru profile {profile_path} must be a YAML mapping.")
    return GuruPersona.model_validate(raw_profile)


def load_guru_personas(profiles_directory: Path) -> dict[str, GuruPersona]:
    """Load every <guru_id>/profile.yaml under a directory.

    Args:
        profiles_directory: Directory holding one subfolder per guru.

    Returns:
        Personas keyed by guru_id.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If a profile's guru_id does not match its folder name.

    Example:
        >>> sorted(load_guru_personas(Path("docs/gurus")))
        ['buffett', 'coach_noom', 'dalio', 'niwes']
    """
    if not profiles_directory.is_dir():
        raise FileNotFoundError(f"Guru profiles directory not found: {profiles_directory}")
    personas: dict[str, GuruPersona] = {}
    for profile_path in sorted(profiles_directory.glob(f"*/{PROFILE_FILENAME}")):
        persona = load_guru_persona_file(profile_path)
        folder_name = profile_path.parent.name
        if persona.guru_id != folder_name:
            raise ValueError(f"guru_id '{persona.guru_id}' does not match folder '{folder_name}'.")
        personas[persona.guru_id] = persona
    return personas


@lru_cache(maxsize=1)
def get_default_guru_personas() -> dict[str, GuruPersona]:
    """Load the shipped personas once and cache them.

    Returns:
        Personas from docs/gurus keyed by guru_id.

    Example:
        >>> "buffett" in get_default_guru_personas()
        True
    """
    return load_guru_personas(DEFAULT_PROFILES_DIRECTORY)


def get_guru_persona(
    guru_id: str,
    personas: dict[str, GuruPersona] | None = None,
) -> GuruPersona:
    """Look up one persona by id.

    Args:
        guru_id: Persona id such as "buffett".
        personas: Registry to search. Defaults to the shipped personas.

    Returns:
        The matching GuruPersona.

    Raises:
        UnknownGuruError: If guru_id is not registered.

    Example:
        >>> get_guru_persona("dalio").display_name
        'Ray Dalio'
    """
    registry = personas if personas is not None else get_default_guru_personas()
    if guru_id not in registry:
        available = ", ".join(sorted(registry))
        raise UnknownGuruError(f"Unknown guru: '{guru_id}'. Available: {available}")
    return registry[guru_id]
