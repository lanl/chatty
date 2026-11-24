"""Tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from chatty.config import (
    DEFAULT_RAG_WORKSPACE,
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


@pytest.mark.parametrize("source", ["environment", ".env"])
def test_generic_variables_do_not_configure_chatty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    monkeypatch.chdir(tmp_path)
    unrelated = {
        "API_KEY": "another-tools-key",
        "BASE_URL": "https://other.example/v1",
        "RAG_WORKSPACE": "other-workspace",
    }
    if source == "environment":
        for name, value in unrelated.items():
            monkeypatch.setenv(name, value)
    else:
        (tmp_path / ".env").write_text("\n".join(f"{k}={v}" for k, v in unrelated.items()))

    result = load_config()
    for config in (result.config, Config()):
        assert config.api_key is None
        assert config.base_url == ""
        assert config.rag_workspace == DEFAULT_RAG_WORKSPACE
    assert all(result.sources[key.lower()] == "default" for key in unrelated)


@pytest.mark.parametrize(
    ("variable", "field", "expected"),
    [
        ("OPENAI_BASE_URL", "base_url", "https://example.test/v1"),
        ("CHATTY_BASE_URL", "base_url", "https://example.test/v1"),
        ("OPENAI_API_KEY", "api_key", SecretStr("test-key")),
        ("CHATTY_API_KEY", "api_key", SecretStr("test-key")),
        ("CHATTY_API_KEY_FILE", "api_key_file", Path("keyfile")),
        ("CHATTY_MODEL", "model", "custom-model"),
        ("CHATTY_CA_BUNDLE", "ca_bundle", "custom.pem"),
        ("CHATTY_VERIFY_TLS", "verify_tls", False),
        ("CHATTY_HTTP_PROXY", "http_proxy", "http://proxy.test"),
        ("CHATTY_CONTEXT_WINDOW", "context_window", 8192),
        ("CHATTY_TEMPERATURE", "temperature", 0.75),
        ("CHATTY_STREAM", "stream", False),
        ("CHATTY_SYSTEM_PROMPT", "system_prompt", "custom prompt"),
        ("CHATTY_TIMEOUT", "timeout_s", 120),
        ("CHATTY_TIMEOUT_S", "timeout_s", 90),
        ("CHATTY_SHOW_TIMESTAMPS", "show_timestamps", True),
        ("CHATTY_TRANSCRIPT_ENABLED", "transcript_enabled", True),
        ("CHATTY_TRANSCRIPT_PATH", "transcript_path", "logs"),
        ("CHATTY_RAG_PROVIDER", "rag_provider", "litkit"),
        ("CHATTY_RAG_WORKSPACE", "rag_workspace", "workspace"),
        ("LITKIT_WORKSPACE", "rag_workspace", "workspace"),
        ("CHATTY_RAG_TOP_PAPERS", "rag_top_papers", 50),
        ("CHATTY_RAG_TOP_CHUNKS", "rag_top_chunks", 12),
        ("CHATTY_RAG_REWRITE_ENABLED", "rag_rewrite_enabled", False),
        ("CHATTY_RAG_EXPAND_SYNONYMS", "rag_expand_synonyms", False),
        ("CHATTY_RAG_RERANK", "rag_rerank", True),
        ("CHATTY_RAG_RERANK_TOP_N", "rag_rerank_top_n", 6),
        ("CHATTY_RAG_MULTI_QUERY", "rag_multi_query", True),
        ("CHATTY_RAG_MULTI_QUERY_COUNT", "rag_multi_query_count", 7),
        ("CHATTY_SESSION_PATH", "session_path", "history"),
        ("CHATTY_COPY_FALLBACK_PATH", "copy_fallback_path", "clipboard"),
        ("CHATTY_EXPORT_PATH", "export_path", "markdown"),
    ],
)
def test_env_setting_value_and_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, variable: str, field: str, expected: object
) -> None:
    monkeypatch.chdir(tmp_path)
    raw = expected.get_secret_value() if isinstance(expected, SecretStr) else str(expected)
    monkeypatch.setenv(variable, raw)
    result = load_config()
    assert getattr(result.config, field) == expected
    assert result.sources[field] == f"env:{variable}"


def test_timeout_alias_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "chatty.toml").write_text("timeout_s = 30\n")
    monkeypatch.setenv("CHATTY_TIMEOUT_S", "90")
    monkeypatch.setenv("CHATTY_TIMEOUT", "120")

    result = load_config()
    assert result.config.timeout_s == 120
    assert result.sources["timeout_s"] == "env:CHATTY_TIMEOUT"
    assert Config().timeout_s == 120
    result = load_config(cli_overrides={"timeout_s": 45})
    assert result.config.timeout_s == 45
    assert result.sources["timeout_s"] == "cli"

    monkeypatch.setenv("CHATTY_TIMEOUT", "0")
    with pytest.raises(ValidationError, match="timeout_s must be positive"):
        load_config()


def test_aliased_setting_across_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_BASE_URL=https://dotenv.example/v1\n")
    result = load_config()
    assert (result.config.base_url, result.sources["base_url"]) == (
        "https://dotenv.example/v1",
        ".env",
    )

    (tmp_path / "chatty.toml").write_text('base_url = "https://toml.example/v1"\n')
    result = load_config()
    assert (result.config.base_url, result.sources["base_url"]) == (
        "https://toml.example/v1",
        "chatty.toml",
    )

    monkeypatch.setenv("chatty_base_url", "https://env.example/v1")
    result = load_config()
    assert (result.config.base_url, result.sources["base_url"]) == (
        "https://env.example/v1",
        "env:CHATTY_BASE_URL",
    )
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    result = load_config()
    assert (result.config.base_url, result.sources["base_url"]) == ("", "env:OPENAI_BASE_URL")

    result = load_config(cli_overrides={"base_url": "https://cli.example/v1"})
    assert (result.config.base_url, result.sources["base_url"]) == ("https://cli.example/v1", "cli")


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


@pytest.mark.skipif(os.name == "nt", reason="Windows has no POSIX file modes")
def test_get_api_key_file_bad_permissions(tmp_path: Path) -> None:
    """Test API key file with insecure permissions fails."""
    key_file = tmp_path / "api-key"
    key_file.write_text("secret-key")
    key_file.chmod(0o644)  # World-readable

    config = Config(api_key_file=key_file)
    with pytest.raises(ValueError, match="readable by group/others"):
        config.get_api_key()


def test_get_api_key_file_mode_not_checked_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows, where stat() reports 0o666 whatever chmod set, the mode is not checked."""
    key_file = tmp_path / "api-key"
    key_file.write_text("file-api-key\n")
    key_file.chmod(0o644)
    monkeypatch.setattr("chatty.config.CHECK_KEY_FILE_MODE", False)

    config = Config(api_key_file=key_file)
    assert config.get_api_key() == "file-api-key"


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

    assert path == Path.home() / ".config" / "chatty" / "transcripts"


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

    assert path == Path.home() / "custom" / "copies"


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


@pytest.mark.parametrize("value", [1, 10, 100])
def test_rag_rerank_top_n_valid_range(value: int) -> None:
    """Reranking accepts values inside its documented positive range."""
    assert Config(rag_rerank_top_n=value).rag_rerank_top_n == value


@pytest.mark.parametrize("value", [0, -1, 101])
def test_rag_rerank_top_n_out_of_range(value: int) -> None:
    """Reranking rejects values that would drop all or unintended chunks."""
    with pytest.raises(ValueError, match="rag_rerank_top_n must be between 1 and 100"):
        Config(rag_rerank_top_n=value)


@pytest.mark.parametrize("value", [1, 3, 10])
def test_rag_multi_query_count_valid_range(value: int) -> None:
    """Multi-query retrieval accepts the supported number of variants."""
    assert Config(rag_multi_query_count=value).rag_multi_query_count == value


@pytest.mark.parametrize("value", [0, -2, 11])
def test_rag_multi_query_count_out_of_range(value: int) -> None:
    """Multi-query retrieval rejects non-positive and excessive variant counts."""
    with pytest.raises(ValueError, match="rag_multi_query_count must be between 1 and 10"):
        Config(rag_multi_query_count=value)


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
    # Uses [T.N] turn-prefixed format for citations
    assert "[T.N]" in RAG_DEFAULT_SYSTEM_PROMPT
    assert "[1.3]" in RAG_DEFAULT_SYSTEM_PROMPT  # Example in prompt
    assert "cite" in RAG_DEFAULT_SYSTEM_PROMPT.lower()
    assert "context" in RAG_DEFAULT_SYSTEM_PROMPT.lower()


# ============================================================================
# Proxy Environment Variable Tests (v0.4.4)
# ============================================================================


def test_http_proxy_from_chatty_http_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test http_proxy reads from CHATTY_HTTP_PROXY env var."""
    monkeypatch.setenv("CHATTY_HTTP_PROXY", "http://proxy.example.com:8080")

    config = Config()
    assert config.http_proxy == "http://proxy.example.com:8080"


def test_http_proxy_not_from_https_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test http_proxy does NOT read from HTTPS_PROXY (httpx handles via trust_env)."""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:8080")

    config = Config()
    # HTTPS_PROXY should not populate http_proxy - httpx handles it via trust_env
    assert config.http_proxy is None


def test_http_proxy_not_from_http_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test http_proxy does NOT read from HTTP_PROXY (httpx handles via trust_env)."""
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.com:8080")

    config = Config()
    # HTTP_PROXY should not populate http_proxy - httpx handles it via trust_env
    assert config.http_proxy is None


def test_http_proxy_default_none() -> None:
    """Test http_proxy defaults to None."""
    config = Config()
    assert config.http_proxy is None
