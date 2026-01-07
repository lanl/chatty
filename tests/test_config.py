"""Tests for configuration management."""

import pytest

from chatty.config import Config, load_config


def test_default_config() -> None:
    """Test default configuration values."""
    config = Config()
    assert config.model == "gpt-4.1"
    assert config.temperature == 0.2
    assert config.stream is True
    assert config.verify_tls is True
    assert config.timeout_s == 60


def test_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test configuration from environment variables."""
    monkeypatch.setenv("CHATTY_BASE_URL", "https://test.api/v1")
    monkeypatch.setenv("CHATTY_MODEL", "gpt-3.5-turbo")
    monkeypatch.setenv("CHATTY_TEMPERATURE", "0.5")

    config = Config()
    assert config.base_url == "https://test.api/v1"
    assert config.model == "gpt-3.5-turbo"
    assert config.temperature == 0.5


def test_load_config() -> None:
    """Test load_config function."""
    config = load_config()
    assert isinstance(config, Config)


def test_display_redacts_secrets(mock_config: Config) -> None:
    """Test that display() redacts API key."""
    display = mock_config.display()
    assert display["api_key"] == "********"
    assert "test-api-key" not in str(display)


def test_get_api_key_from_config(mock_config: Config) -> None:
    """Test getting API key from config."""
    key = mock_config.get_api_key()
    assert key == "test-api-key"


def test_get_api_key_missing() -> None:
    """Test error when no API key configured."""
    config = Config()
    with pytest.raises(ValueError, match="No API key configured"):
        config.get_api_key()
