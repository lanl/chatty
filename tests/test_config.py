"""Tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from chatty.config import (
    Config,
    ConfigWithSources,
    format_config_with_sources,
    get_config_path,
    load_config,
    load_toml_config,
)


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
    monkeypatch.setenv("CHATTY_MODEL", "gpt-3.5-turbo")
    monkeypatch.setenv("CHATTY_TEMPERATURE", "0.5")

    config = Config()
    assert config.model == "gpt-3.5-turbo"
    assert config.temperature == 0.5


def test_openai_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OPENAI_* environment variable aliases."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    config = Config()
    assert config.base_url == "https://api.openai.com/v1"
    assert config.api_key is not None
    assert config.api_key.get_secret_value() == "sk-test-key"


def test_load_config_returns_config_with_sources() -> None:
    """Test load_config returns ConfigWithSources."""
    result = load_config()
    assert isinstance(result, ConfigWithSources)
    assert isinstance(result.config, Config)
    assert isinstance(result.sources, dict)


def test_load_config_with_cli_overrides() -> None:
    """Test CLI overrides take precedence."""
    result = load_config(cli_overrides={"model": "cli-model", "temperature": 0.9})
    assert result.config.model == "cli-model"
    assert result.config.temperature == 0.9
    assert result.sources["model"] == "cli"
    assert result.sources["temperature"] == "cli"


def test_load_config_source_attribution_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test source attribution for environment variables."""
    monkeypatch.setenv("CHATTY_MODEL", "env-model")

    result = load_config()
    assert result.config.model == "env-model"
    assert result.sources["model"] == "env:CHATTY_MODEL"


def test_load_config_source_attribution_default() -> None:
    """Test source attribution for default values."""
    result = load_config()
    assert result.sources["model"] == "default"
    assert result.sources["temperature"] == "default"


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


def test_get_api_key_from_file(tmp_path: Path) -> None:
    """Test loading API key from file."""
    key_file = tmp_path / "api-key"
    key_file.write_text("file-api-key\n")
    key_file.chmod(0o600)

    config = Config(api_key_file=key_file)
    assert config.get_api_key() == "file-api-key"


def test_get_api_key_file_bad_permissions(tmp_path: Path) -> None:
    """Test API key file with insecure permissions fails."""
    key_file = tmp_path / "api-key"
    key_file.write_text("secret-key")
    key_file.chmod(0o644)  # World-readable

    config = Config(api_key_file=key_file)
    with pytest.raises(ValueError, match="readable by group/others"):
        config.get_api_key()


def test_get_api_key_file_not_found() -> None:
    """Test API key file not found error."""
    config = Config(api_key_file=Path("/nonexistent/key"))
    with pytest.raises(ValueError, match="API key file not found"):
        config.get_api_key()


def test_format_config_with_sources() -> None:
    """Test formatting config with source attribution."""
    config = Config(model="test-model")
    sources = {"model": "cli", "temperature": "default"}
    config_with_sources = ConfigWithSources(config=config, sources=sources)

    output = format_config_with_sources(config_with_sources)
    assert "model: test-model (from: cli)" in output
    assert "temperature: 0.2 (from: default)" in output


def test_get_config_path() -> None:
    """Test config path location."""
    path = get_config_path()
    assert path == Path.home() / ".config" / "chatty" / "config.toml"


def test_load_toml_config_no_file() -> None:
    """Test TOML loading when no file exists."""
    with patch("chatty.config.get_config_path") as mock_path:
        mock_path.return_value = Path("/nonexistent/config.toml")
        result = load_toml_config()
        assert result == {}


def test_load_toml_config_with_file(tmp_path: Path) -> None:
    """Test TOML loading from file."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
base_url = "https://toml.example.com/v1"
model = "toml-model"
temperature = 0.7
""")

    with patch("chatty.config.get_config_path") as mock_path:
        mock_path.return_value = config_file
        result = load_toml_config()
        assert result["base_url"] == "https://toml.example.com/v1"
        assert result["model"] == "toml-model"
        assert result["temperature"] == 0.7


def test_config_precedence_cli_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI flags take precedence over env vars."""
    monkeypatch.setenv("CHATTY_MODEL", "env-model")

    result = load_config(cli_overrides={"model": "cli-model"})
    assert result.config.model == "cli-model"
    assert result.sources["model"] == "cli"


def test_config_precedence_env_over_toml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test env vars take precedence over TOML file."""
    config_file = tmp_path / "config.toml"
    config_file.write_text('model = "toml-model"')

    monkeypatch.setenv("CHATTY_MODEL", "env-model")

    with patch("chatty.config.get_config_path") as mock_path:
        mock_path.return_value = config_file
        result = load_config()
        assert result.config.model == "env-model"
        assert result.sources["model"] == "env:CHATTY_MODEL"


def test_config_precedence_toml_over_default(tmp_path: Path) -> None:
    """Test TOML file takes precedence over defaults."""
    config_file = tmp_path / "config.toml"
    config_file.write_text('model = "toml-model"')

    # Clear any env vars that might interfere
    env = os.environ.copy()
    for key in list(env.keys()):
        if key.startswith("CHATTY_") or key.startswith("OPENAI_"):
            del os.environ[key]

    with patch("chatty.config.get_config_path") as mock_path:
        mock_path.return_value = config_file
        result = load_config()
        assert result.config.model == "toml-model"
        assert result.sources["model"] == "config.toml"
