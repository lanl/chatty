"""Configuration management for chatty."""

from __future__ import annotations

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


def get_config_path() -> Path:
    """Get the path to the config file."""
    return Path.home() / ".config" / "chatty" / "config.toml"


def load_toml_config() -> dict[str, Any]:
    """Load configuration from TOML file if it exists."""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    return {}


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

    def get_api_key(self) -> str:
        """Load API key from file (preferred) or direct config."""
        if self.api_key_file:
            path = self.api_key_file
            if not path.exists():
                raise ValueError(f"API key file not found: {path}")
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
        }


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

    Precedence (highest to lowest):
    1. cli_overrides (passed from CLI flags)
    2. Environment variables
    3. TOML config file
    4. Defaults

    Returns:
        ConfigWithSources with the resolved config and source attribution.
    """
    import os

    sources: dict[str, str] = {}
    toml_config = load_toml_config()
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
            sources[key] = "config.toml"
            continue

        # Use default
        sources[key] = "default"

    # Build final config values, respecting precedence:
    # CLI > env vars > TOML > defaults
    final_values: dict[str, Any] = {}

    # Only apply TOML values if no env var is set for that key
    for key, value in toml_config.items():
        if key in defaults and sources.get(key, "").startswith("config.toml"):
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
    }

    for key, value in display_values.items():
        source = sources.get(key, "unknown")
        lines.append(f"  {key}: {value} (from: {source})")

    return "\n".join(lines)
