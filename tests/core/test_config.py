"""Tests for configuration management."""

import pytest
from pydantic import ValidationError
from finance_ai.core.config import Settings, get_settings


def test_settings_default_values() -> None:
    """Test Settings creates with default values."""
    settings = Settings(_env_file=None, google_api_key="test-key")

    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.llm_provider == "google"


def test_settings_google_provider() -> None:
    """Test Settings validates Google provider configuration."""
    settings = Settings(
        llm_provider="google",
        google_api_key="test-key",
        google_model="gemini-2.5-flash",
    )

    assert settings.llm_provider == "google"
    assert settings.google_api_key == "test-key"
    assert settings.google_model == "gemini-2.5-flash"


def test_settings_ollama_provider() -> None:
    """Test Settings validates OLLAMA provider configuration."""
    settings = Settings(_env_file=None, llm_provider="ollama", ollama_model="minimax-m3")

    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "minimax-m3"


def test_settings_invalid_provider() -> None:
    """Test Settings rejects invalid provider."""
    with pytest.raises(ValidationError):
        Settings(llm_provider="invalid")  # type: ignore[arg-type]


def test_settings_allows_missing_google_key() -> None:
    """Test Settings allows creation without key (validation in factory)."""
    settings = Settings(llm_provider="google")

    assert settings.llm_provider == "google"
    assert not settings.google_api_key  # None or empty string


def test_settings_recommendation_temperature_default() -> None:
    """Test recommendation agent uses its own conservative temperature."""
    settings = Settings(_env_file=None, google_api_key="test-key")

    assert settings.llm_recommendation_temperature == 0.3
    assert settings.llm_temperature == 0.7


def test_get_settings_returns_settings() -> None:
    """Test get_settings returns Settings instance."""
    settings = get_settings()

    assert isinstance(settings, Settings)
