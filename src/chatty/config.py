"""Configuration management for chatty."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass
class ConfigSource:
    """Tracks the source of a configuration value."""

    value: Any
    source: str  # "default", "config.toml", "env:VAR_NAME", "cli"


def find_config_path() -> Path | None:
    """Find the config file using search order.

    Search order:
    1. CHATTY_CONFIG environment variable (explicit override)
    2. ./chatty.toml (repo-local config)
    3. ~/.config/chatty/config.toml (user default)

    Returns:
        Path to config file if found, None otherwise.
    """
    # 1. CHATTY_CONFIG env var
    env_config = os.environ.get("CHATTY_CONFIG")
    if env_config:
        path = Path(env_config).expanduser()
        if path.exists():
            return path
        # If explicitly set but doesn't exist, that's an error handled later

    # 2. Repo-local config
    local_config = Path("chatty.toml")
    if local_config.exists():
        return local_config

    # 3. User config
    user_config = Path.home() / ".config" / "chatty" / "config.toml"
    if user_config.exists():
        return user_config

    return None


def get_config_path() -> Path:
    """Get the path to the config file (for backwards compatibility).

    Returns the found config path or the default user config location.
    """
    found = find_config_path()
    if found:
        return found
    return Path.home() / ".config" / "chatty" / "config.toml"


def load_toml_config() -> tuple[dict[str, Any], str]:
    """Load configuration from TOML file if it exists.

    Returns:
        Tuple of (config dict, source description).
        Source is one of: "chatty.toml", "~/.config/chatty/config.toml", "env:CHATTY_CONFIG"
    """
    config_path = find_config_path()
    if config_path is None:
        return {}, ""

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Determine source description
    if os.environ.get("CHATTY_CONFIG"):
        source = f"env:CHATTY_CONFIG ({config_path})"
    elif config_path.name == "chatty.toml":
        source = "chatty.toml"
    else:
        source = "~/.config/chatty/config.toml"

    return config, source


class Config(BaseSettings):
    """Chatty configuration with environment variable and TOML file support.

    Precedence (highest to lowest):
    1. CLI flags (applied after Config instantiation)
    2. Environment variables
    3. TOML config file (~/.config/chatty/config.toml)
    4. Defaults
    """

    model_config = SettingsConfigDict(
        env_prefix="CHATTY_",
        env_file=".env",
        extra="ignore",
    )

    # LLM Endpoint
    # Accepts: OPENAI_BASE_URL, CHATTY_BASE_URL, or base_url in TOML
    base_url: str = Field(
        default="",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "CHATTY_BASE_URL", "base_url"),
    )
    model: str = "gpt-4.1"

    # Authentication
    # Accepts: OPENAI_API_KEY, CHATTY_API_KEY, or api_key in TOML
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "CHATTY_API_KEY", "api_key"),
    )
    api_key_file: Path | None = None

    # Network / TLS
    ca_bundle: str | None = None
    verify_tls: bool = True
    http_proxy: str | None = None
    no_proxy: str = "localhost,127.0.0.1"

    # Behavior
    temperature: float = 0.2
    stream: bool = True
    system_prompt: str = "You are a helpful assistant."
    timeout_s: int = 60

    # UI & Display
    show_timestamps: bool = False

    # Transcript Logging
    transcript_enabled: bool = False
    transcript_path: str = "~/.config/chatty/transcripts"

    # RAG Provider
    rag_provider: str = "none"  # "none" (v0.1-0.2), "litkit" (v0.3+)

    # Session Persistence
    session_path: str = "~/.config/chatty/sessions"

    def get_api_key(self) -> str:
        """Load API key from file (preferred) or direct config."""
        if self.api_key_file:
            path = self.api_key_file.expanduser()
            if not path.exists():
                raise ValueError(f"API key file not found: {self.api_key_file}")
            # Check file permissions (should be 600)
            mode = path.stat().st_mode & 0o777
            if mode & 0o077:
                raise ValueError(
                    f"{path} is readable by group/others (mode {oct(mode)}). "
                    f"Run: chmod 600 {path}"
                )
            return path.read_text().strip()
        if self.api_key:
            return self.api_key.get_secret_value()
        raise ValueError("No API key configured. Set OPENAI_API_KEY or api_key_file.")

    def display(self) -> dict[str, str]:
        """Return config dict with secrets redacted for display."""
        return {
            "base_url": self.base_url or "(not set)",
            "model": self.model,
            "api_key": "********" if (self.api_key or self.api_key_file) else "(not set)",
            "api_key_file": str(self.api_key_file) if self.api_key_file else "(not set)",
            "ca_bundle": self.ca_bundle or "system",
            "verify_tls": str(self.verify_tls),
            "http_proxy": self.http_proxy or "(not set)",
            "temperature": str(self.temperature),
            "stream": str(self.stream),
            "timeout_s": str(self.timeout_s),
            "show_timestamps": str(self.show_timestamps),
            "transcript_enabled": str(self.transcript_enabled),
            "transcript_path": self.transcript_path,
            "rag_provider": self.rag_provider,
            "session_path": self.session_path,
        }

    def get_transcript_path(self) -> Path:
        """Get the resolved transcript path with ~ expanded."""
        return Path(self.transcript_path).expanduser()

    def get_session_path(self) -> Path:
        """Get the resolved session path with ~ expanded."""
        return Path(self.session_path).expanduser()


@dataclass
class ConfigWithSources:
    """Configuration with source attribution for each value."""

    config: Config
    sources: dict[str, str] = field(default_factory=dict)


def load_config(
    *,
    cli_overrides: dict[str, Any] | None = None,
) -> ConfigWithSources:
    """Load configuration from all sources with attribution.

    Config file search order:
    1. CHATTY_CONFIG environment variable
    2. ./chatty.toml (repo-local)
    3. ~/.config/chatty/config.toml (user default)

    Value precedence (highest to lowest):
    1. cli_overrides (passed from CLI flags)
    2. Environment variables
    3. TOML config file
    4. Defaults

    Returns:
        ConfigWithSources with the resolved config and source attribution.
    """
    sources: dict[str, str] = {}
    toml_config, toml_source = load_toml_config()
    cli_overrides = cli_overrides or {}

    # Get defaults from the model
    defaults = {
        "base_url": "",
        "model": "gpt-4.1",
        "api_key": None,
        "api_key_file": None,
        "ca_bundle": None,
        "verify_tls": True,
        "http_proxy": None,
        "no_proxy": "localhost,127.0.0.1",
        "temperature": 0.2,
        "stream": True,
        "system_prompt": "You are a helpful assistant.",
        "timeout_s": 60,
        "show_timestamps": False,
        "transcript_enabled": False,
        "transcript_path": "~/.config/chatty/transcripts",
        "rag_provider": "none",
        "session_path": "~/.config/chatty/sessions",
    }

    # Map env var names to config keys
    env_mappings = {
        "base_url": ["OPENAI_BASE_URL", "CHATTY_BASE_URL"],
        "api_key": ["OPENAI_API_KEY", "CHATTY_API_KEY"],
        "api_key_file": ["CHATTY_API_KEY_FILE"],
        "model": ["CHATTY_MODEL"],
        "ca_bundle": ["CHATTY_CA_BUNDLE"],
        "verify_tls": ["CHATTY_VERIFY_TLS"],
        "http_proxy": ["HTTPS_PROXY", "CHATTY_HTTP_PROXY"],
        "no_proxy": ["NO_PROXY", "CHATTY_NO_PROXY"],
        "temperature": ["CHATTY_TEMPERATURE"],
        "stream": ["CHATTY_STREAM"],
        "system_prompt": ["CHATTY_SYSTEM_PROMPT"],
        "timeout_s": ["CHATTY_TIMEOUT", "CHATTY_TIMEOUT_S"],
        "show_timestamps": ["CHATTY_SHOW_TIMESTAMPS"],
        "transcript_enabled": ["CHATTY_TRANSCRIPT_ENABLED"],
        "transcript_path": ["CHATTY_TRANSCRIPT_PATH"],
        "rag_provider": ["CHATTY_RAG_PROVIDER"],
        "session_path": ["CHATTY_SESSION_PATH"],
    }

    # Determine source for each config value
    for key in defaults:
        # Check CLI first
        if key in cli_overrides:
            sources[key] = "cli"
            continue

        # Check environment variables
        env_found = False
        for env_var in env_mappings.get(key, []):
            if os.environ.get(env_var):
                sources[key] = f"env:{env_var}"
                env_found = True
                break

        if env_found:
            continue

        # Check TOML config
        if key in toml_config:
            sources[key] = toml_source or "config.toml"
            continue

        # Use default
        sources[key] = "default"

    # Build final config values, respecting precedence:
    # CLI > env vars > TOML > defaults
    final_values: dict[str, Any] = {}

    # Only apply TOML values if no env var is set for that key
    for key, value in toml_config.items():
        source = sources.get(key, "")
        if key in defaults and (source.startswith("chatty.toml") or "config" in source):
            final_values[key] = value

    # Apply CLI overrides (highest precedence)
    for key, value in cli_overrides.items():
        if value is not None:
            final_values[key] = value

    # Create config - pydantic-settings will read env vars automatically
    # We only pass values that aren't from env vars
    config = Config(**final_values)

    return ConfigWithSources(config=config, sources=sources)


def format_config_with_sources(config_with_sources: ConfigWithSources) -> str:
    """Format configuration for display with source attribution."""
    config = config_with_sources.config
    sources = config_with_sources.sources
    lines = []

    # Show which config file was loaded
    config_path = find_config_path()
    if config_path:
        lines.append(f"Config file: {config_path}")
    else:
        lines.append("Config file: (none found)")
    lines.append("")

    display_values = {
        "base_url": config.base_url or "(not set)",
        "model": config.model,
        "api_key": "********" if (config.api_key or config.api_key_file) else "(not set)",
        "api_key_file": str(config.api_key_file) if config.api_key_file else "(not set)",
        "ca_bundle": config.ca_bundle or "system",
        "verify_tls": str(config.verify_tls),
        "http_proxy": config.http_proxy or "(not set)",
        "no_proxy": config.no_proxy,
        "temperature": str(config.temperature),
        "stream": str(config.stream),
        "system_prompt": (
            config.system_prompt[:40] + "..."
            if len(config.system_prompt) > 40
            else config.system_prompt
        ),
        "timeout_s": str(config.timeout_s),
        "show_timestamps": str(config.show_timestamps),
        "transcript_enabled": str(config.transcript_enabled),
        "transcript_path": config.transcript_path,
        "rag_provider": config.rag_provider,
        "session_path": config.session_path,
    }

    for key, value in display_values.items():
        source = sources.get(key, "unknown")
        lines.append(f"  {key}: {value} (from: {source})")

    return "\n".join(lines)
