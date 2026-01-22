"""Configuration management for chatty."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default system prompt for non-RAG mode
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# Default system prompt when RAG is enabled (litkit-style)
# Instructs the LLM to cite sources using bracketed references [1], [2], etc.
RAG_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise scientific assistant. Answer questions using ONLY "
    "the provided context chunks—do not use prior knowledge. Each chunk "
    "is numbered [1], [2], etc. You MUST cite sources using these numbers "
    "in your response. When multiple chunks support a claim, cite all of "
    "them. If sources conflict, acknowledge the disagreement. If the "
    "context is insufficient to answer, say so briefly."
)


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

    # Context Window
    # "auto" = fetch from /models endpoint
    # integer = explicit size in tokens
    context_window: str | int = "auto"

    # Behavior
    temperature: float = 0.2
    stream: bool = True
    system_prompt: str = "You are a helpful assistant."
    timeout_s: int = 60

    @field_validator("context_window")
    @classmethod
    def validate_context_window(cls, v: str | int) -> str | int:
        """Validate context_window is 'auto' or a positive integer."""
        if isinstance(v, str):
            if v.lower() != "auto":
                # Try to parse as integer
                try:
                    v = int(v)
                except ValueError:
                    raise ValueError(
                        "context_window must be 'auto' or a positive integer"
                    ) from None
            else:
                return "auto"
        if isinstance(v, int) and v <= 0:
            raise ValueError("context_window must be a positive integer")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is in OpenAI API range (0.0-2.0)."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v

    @field_validator("timeout_s")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is positive."""
        if v <= 0:
            raise ValueError("timeout_s must be positive")
        return v

    # UI & Display
    show_timestamps: bool = False

    # Transcript Logging
    transcript_enabled: bool = False
    transcript_path: str = "~/.config/chatty/transcripts"

    # RAG Provider
    rag_provider: str = "none"  # "none" (v0.1-0.3), "litkit" (v0.4+)

    # RAG Settings (v0.4+)
    rag_workspace: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CHATTY_RAG_WORKSPACE", "LITKIT_WORKSPACE", "rag_workspace"),
    )
    rag_top_papers: int = 500  # Stage 1: papers to shortlist
    rag_top_chunks: int = 30  # Stage 2: chunks for LLM context

    @field_validator("rag_top_papers")
    @classmethod
    def validate_rag_top_papers(cls, v: int) -> int:
        """Validate rag_top_papers is in reasonable range (1-5000)."""
        if not 1 <= v <= 5000:
            raise ValueError("rag_top_papers must be between 1 and 5000")
        return v

    @field_validator("rag_top_chunks")
    @classmethod
    def validate_rag_top_chunks(cls, v: int) -> int:
        """Validate rag_top_chunks is in reasonable range (1-100)."""
        if not 1 <= v <= 100:
            raise ValueError("rag_top_chunks must be between 1 and 100")
        return v

    # Session Persistence
    session_path: str = "./sessions"

    # Copy to Clipboard
    copy_fallback_path: str = "./copies"

    # Markdown Export
    export_path: str = "./exports"

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
            "context_window": str(self.context_window),
            "temperature": str(self.temperature),
            "stream": str(self.stream),
            "timeout_s": str(self.timeout_s),
            "show_timestamps": str(self.show_timestamps),
            "transcript_enabled": str(self.transcript_enabled),
            "transcript_path": self.transcript_path,
            "rag_provider": self.rag_provider,
            "rag_workspace": self.rag_workspace or "(not set)",
            "rag_top_papers": str(self.rag_top_papers),
            "rag_top_chunks": str(self.rag_top_chunks),
            "session_path": self.session_path,
            "copy_fallback_path": self.copy_fallback_path,
            "export_path": self.export_path,
        }

    def get_transcript_path(self) -> Path:
        """Get the resolved transcript path with ~ expanded."""
        return Path(self.transcript_path).expanduser()

    def get_session_path(self) -> Path:
        """Get the resolved session path with ~ expanded."""
        return Path(self.session_path).expanduser()

    def get_copy_fallback_path(self) -> Path:
        """Get the resolved copy fallback path with ~ expanded."""
        return Path(self.copy_fallback_path).expanduser()

    def get_export_path(self) -> Path:
        """Get the resolved export path with ~ expanded."""
        return Path(self.export_path).expanduser()

    def get_rag_workspace(self) -> Path | None:
        """Get the resolved RAG workspace path with ~ expanded.

        Returns:
            Path to workspace if configured, None otherwise.
        """
        if self.rag_workspace:
            return Path(self.rag_workspace).expanduser()
        return None

    def get_effective_system_prompt(self) -> str:
        """Get the effective system prompt based on RAG configuration.

        When RAG is enabled (rag_provider != "none") and the user hasn't
        explicitly configured a system_prompt, returns RAG_DEFAULT_SYSTEM_PROMPT
        which instructs the LLM to cite sources.

        When RAG is disabled or user has set a custom system_prompt, returns
        the configured system_prompt.

        Returns:
            The effective system prompt to use.
        """
        # Check if user explicitly set a system prompt (not the default)
        if self.system_prompt != DEFAULT_SYSTEM_PROMPT:
            # User provided custom prompt - use it regardless of RAG
            return self.system_prompt

        # No custom prompt - use RAG prompt if RAG is enabled
        if self.rag_provider != "none":
            return RAG_DEFAULT_SYSTEM_PROMPT

        # Non-RAG mode with default prompt
        return self.system_prompt


@dataclass
class ConfigWithSources:
    """Configuration with source attribution for each value."""

    config: Config
    sources: dict[str, str] = field(default_factory=dict)


# Config field names for iteration (derived from Config class fields)
_CONFIG_FIELDS = [
    "base_url",
    "model",
    "api_key",
    "api_key_file",
    "ca_bundle",
    "verify_tls",
    "http_proxy",
    "no_proxy",
    "context_window",
    "temperature",
    "stream",
    "system_prompt",
    "timeout_s",
    "show_timestamps",
    "transcript_enabled",
    "transcript_path",
    "rag_provider",
    "rag_workspace",
    "rag_top_papers",
    "rag_top_chunks",
    "session_path",
    "copy_fallback_path",
    "export_path",
]

# Map env var names to config keys for source attribution
_ENV_MAPPINGS = {
    "base_url": ["OPENAI_BASE_URL", "CHATTY_BASE_URL"],
    "api_key": ["OPENAI_API_KEY", "CHATTY_API_KEY"],
    "api_key_file": ["CHATTY_API_KEY_FILE"],
    "model": ["CHATTY_MODEL"],
    "ca_bundle": ["CHATTY_CA_BUNDLE"],
    "verify_tls": ["CHATTY_VERIFY_TLS"],
    "http_proxy": ["HTTPS_PROXY", "CHATTY_HTTP_PROXY"],
    "no_proxy": ["NO_PROXY", "CHATTY_NO_PROXY"],
    "context_window": ["CHATTY_CONTEXT_WINDOW"],
    "temperature": ["CHATTY_TEMPERATURE"],
    "stream": ["CHATTY_STREAM"],
    "system_prompt": ["CHATTY_SYSTEM_PROMPT"],
    "timeout_s": ["CHATTY_TIMEOUT", "CHATTY_TIMEOUT_S"],
    "show_timestamps": ["CHATTY_SHOW_TIMESTAMPS"],
    "transcript_enabled": ["CHATTY_TRANSCRIPT_ENABLED"],
    "transcript_path": ["CHATTY_TRANSCRIPT_PATH"],
    "rag_provider": ["CHATTY_RAG_PROVIDER"],
    "rag_workspace": ["CHATTY_RAG_WORKSPACE", "LITKIT_WORKSPACE"],
    "rag_top_papers": ["CHATTY_RAG_TOP_PAPERS"],
    "rag_top_chunks": ["CHATTY_RAG_TOP_CHUNKS"],
    "session_path": ["CHATTY_SESSION_PATH"],
    "copy_fallback_path": ["CHATTY_COPY_FALLBACK_PATH"],
    "export_path": ["CHATTY_EXPORT_PATH"],
}


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
    4. Defaults (defined in Config class)

    Returns:
        ConfigWithSources with the resolved config and source attribution.
    """
    sources: dict[str, str] = {}
    toml_config, toml_source = load_toml_config()
    cli_overrides = cli_overrides or {}

    # Determine source for each config value
    for key in _CONFIG_FIELDS:
        # Check CLI first
        if key in cli_overrides:
            sources[key] = "cli"
            continue

        # Check environment variables
        env_found = False
        for env_var in _ENV_MAPPINGS.get(key, []):
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

        # Use default (from Config class)
        sources[key] = "default"

    # Build final config values, respecting precedence:
    # CLI > env vars > TOML > defaults
    final_values: dict[str, Any] = {}

    # Only apply TOML values if no env var is set for that key
    for key, value in toml_config.items():
        source = sources.get(key, "")
        if key in _CONFIG_FIELDS and (source.startswith("chatty.toml") or "config" in source):
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
        "context_window": str(config.context_window),
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
        "rag_workspace": config.rag_workspace or "(not set)",
        "rag_top_papers": str(config.rag_top_papers),
        "rag_top_chunks": str(config.rag_top_chunks),
        "session_path": config.session_path,
        "copy_fallback_path": config.copy_fallback_path,
        "export_path": config.export_path,
    }

    for key, value in display_values.items():
        source = sources.get(key, "unknown")
        lines.append(f"  {key}: {value} (from: {source})")

    return "\n".join(lines)
