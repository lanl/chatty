"""Tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from chatty.config import (
    Config,
    ConfigWithSources,
    find_config_path,
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
    # Mock find_config_path to ensure no config file is found
    with patch("chatty.config.find_config_path") as mock_find:
        mock_find.return_value = None
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
    """Test config path location defaults to user config when no file found."""
    # When no config file exists, get_config_path returns the default user location
    with patch("chatty.config.find_config_path") as mock_find:
        mock_find.return_value = None
        path = get_config_path()
        assert path == Path.home() / ".config" / "chatty" / "config.toml"


def test_load_toml_config_no_file() -> None:
    """Test TOML loading when no file exists."""
    with patch("chatty.config.find_config_path") as mock_path:
        mock_path.return_value = None
        result, source = load_toml_config()
        assert result == {}
        assert source == ""


def test_load_toml_config_with_file(tmp_path: Path) -> None:
    """Test TOML loading from file."""
    config_file = tmp_path / "config.toml"
    config_file.write_text("""
base_url = "https://toml.example.com/v1"
model = "toml-model"
temperature = 0.7
""")

    with patch("chatty.config.find_config_path") as mock_path:
        mock_path.return_value = config_file
        result, source = load_toml_config()
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

    with patch("chatty.config.find_config_path") as mock_path:
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

    with patch("chatty.config.find_config_path") as mock_path:
        mock_path.return_value = config_file
        result = load_config()
        assert result.config.model == "toml-model"
        # Source will include the path info
        assert "config" in result.sources["model"].lower() or "toml" in result.sources["model"]


def test_find_config_path_chatty_config_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test CHATTY_CONFIG env var takes priority."""
    config_file = tmp_path / "custom-config.toml"
    config_file.write_text('model = "env-config-model"')

    monkeypatch.setenv("CHATTY_CONFIG", str(config_file))

    result = find_config_path()
    assert result == config_file


def test_find_config_path_chatty_config_env_nonexistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test CHATTY_CONFIG env var with nonexistent path falls through."""
    monkeypatch.setenv("CHATTY_CONFIG", "/nonexistent/path/to/config.toml")

    # When CHATTY_CONFIG points to nonexistent file, find_config_path falls through
    # to check other locations. Just verify it doesn't crash.
    find_config_path()


def test_find_config_path_repo_local_chatty_toml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test ./chatty.toml is found when it exists."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)

    # Create chatty.toml in temp directory
    config_file = tmp_path / "chatty.toml"
    config_file.write_text('model = "local-model"')

    # Clear CHATTY_CONFIG if set
    monkeypatch.delenv("CHATTY_CONFIG", raising=False)

    result = find_config_path()
    assert result is not None
    assert result.name == "chatty.toml"


def test_get_transcript_path_expands_tilde() -> None:
    """Test get_transcript_path expands ~ to home directory."""
    config = Config(transcript_path="~/.config/chatty/transcripts")
    path = config.get_transcript_path()

    assert "~" not in str(path)
    assert str(path).startswith(str(Path.home()))


def test_get_transcript_path_relative() -> None:
    """Test get_transcript_path handles relative paths."""
    config = Config(transcript_path="./transcripts")
    path = config.get_transcript_path()

    assert path == Path("./transcripts")


def test_config_display_all_fields() -> None:
    """Test display() includes all expected fields."""
    config = Config(
        base_url="https://example.com/v1",
        model="test-model",
        temperature=0.5,
        stream=True,
        timeout_s=120,
        show_timestamps=True,
        transcript_enabled=True,
        transcript_path="./transcripts",
    )
    display = config.display()

    assert display["base_url"] == "https://example.com/v1"
    assert display["model"] == "test-model"
    assert display["temperature"] == "0.5"
    assert display["stream"] == "True"
    assert display["timeout_s"] == "120"
    assert display["show_timestamps"] == "True"
    assert display["transcript_enabled"] == "True"
    assert display["transcript_path"] == "./transcripts"


def test_format_config_shows_config_file_path(tmp_path: Path) -> None:
    """Test format_config_with_sources shows which config file was loaded."""
    config_file = tmp_path / "chatty.toml"
    config_file.write_text('model = "file-model"')

    with patch("chatty.config.find_config_path") as mock_find:
        mock_find.return_value = config_file

        config = Config(model="file-model")
        sources = {"model": "chatty.toml"}
        config_with_sources = ConfigWithSources(config=config, sources=sources)

        output = format_config_with_sources(config_with_sources)
        assert f"Config file: {config_file}" in output


def test_format_config_shows_no_config_file() -> None:
    """Test format_config_with_sources shows when no config file found."""
    with patch("chatty.config.find_config_path") as mock_find:
        mock_find.return_value = None

        config = Config()
        sources = {"model": "default"}
        config_with_sources = ConfigWithSources(config=config, sources=sources)

        output = format_config_with_sources(config_with_sources)
        assert "Config file: (none found)" in output


def test_load_toml_config_source_chatty_toml(tmp_path: Path) -> None:
    """Test source is 'chatty.toml' for repo-local config."""
    config_file = tmp_path / "chatty.toml"
    config_file.write_text('model = "local-model"')

    with patch("chatty.config.find_config_path") as mock_path:
        mock_path.return_value = config_file
        _result, source = load_toml_config()
        assert source == "chatty.toml"


def test_copy_fallback_path_default() -> None:
    """Test copy_fallback_path has default value."""
    config = Config()
    assert config.copy_fallback_path == "./copies"


def test_get_copy_fallback_path_expands_tilde() -> None:
    """Test get_copy_fallback_path expands ~ to home directory."""
    config = Config(copy_fallback_path="~/custom/copies")
    path = config.get_copy_fallback_path()

    assert "~" not in str(path)
    assert str(path).startswith(str(Path.home()))


def test_config_display_includes_copy_fallback_path() -> None:
    """Test display() includes copy_fallback_path."""
    config = Config(copy_fallback_path="./my-copies")
    display = config.display()

    assert display["copy_fallback_path"] == "./my-copies"


# ============================================================================
# Validation Tests (v0.2.3e)
# ============================================================================


def test_temperature_valid_range() -> None:
    """Temperature 0.0, 1.0, 2.0 all accepted."""
    # Minimum
    config = Config(temperature=0.0)
    assert config.temperature == 0.0

    # Middle
    config = Config(temperature=1.0)
    assert config.temperature == 1.0

    # Maximum
    config = Config(temperature=2.0)
    assert config.temperature == 2.0


def test_temperature_too_high() -> None:
    """Temperature > 2.0 raises ValidationError."""
    with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
        Config(temperature=2.1)


def test_temperature_negative() -> None:
    """Negative temperature raises ValidationError."""
    with pytest.raises(ValueError, match="temperature must be between 0.0 and 2.0"):
        Config(temperature=-0.1)


def test_timeout_positive() -> None:
    """Positive timeout values accepted."""
    config = Config(timeout_s=1)
    assert config.timeout_s == 1

    config = Config(timeout_s=60)
    assert config.timeout_s == 60

    config = Config(timeout_s=300)
    assert config.timeout_s == 300


def test_timeout_zero() -> None:
    """Zero timeout raises ValidationError."""
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        Config(timeout_s=0)


def test_timeout_negative() -> None:
    """Negative timeout raises ValidationError."""
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        Config(timeout_s=-1)


# ============================================================================
# RAG System Prompt Tests (v0.4.1)
# ============================================================================


def test_get_effective_system_prompt_default_no_rag() -> None:
    """Default prompt used when RAG is disabled."""
    config = Config()
    prompt = config.get_effective_system_prompt()
    assert prompt == "You are a helpful assistant."


def test_get_effective_system_prompt_rag_enabled() -> None:
    """RAG default prompt used when RAG is enabled and no custom prompt."""
    config = Config(rag_provider="litkit")
    prompt = config.get_effective_system_prompt()
    assert "precise scientific assistant" in prompt
    assert "cite sources" in prompt.lower()


def test_get_effective_system_prompt_custom_no_rag() -> None:
    """Custom prompt used when explicitly set (no RAG)."""
    config = Config(system_prompt="You are a pirate assistant. Arrr!")
    prompt = config.get_effective_system_prompt()
    assert prompt == "You are a pirate assistant. Arrr!"


def test_get_effective_system_prompt_custom_with_rag() -> None:
    """Custom prompt takes precedence even when RAG is enabled."""
    config = Config(
        rag_provider="litkit",
        system_prompt="You are a medical expert. Cite papers when possible.",
    )
    prompt = config.get_effective_system_prompt()
    assert prompt == "You are a medical expert. Cite papers when possible."
    assert "precise scientific assistant" not in prompt


def test_rag_default_system_prompt_content() -> None:
    """RAG default prompt contains required citation instructions."""
    from chatty.config import RAG_DEFAULT_SYSTEM_PROMPT

    assert "ONLY" in RAG_DEFAULT_SYSTEM_PROMPT
    assert "[1]" in RAG_DEFAULT_SYSTEM_PROMPT
    assert "[2]" in RAG_DEFAULT_SYSTEM_PROMPT
    assert "cite" in RAG_DEFAULT_SYSTEM_PROMPT.lower()
    assert "context" in RAG_DEFAULT_SYSTEM_PROMPT.lower()
