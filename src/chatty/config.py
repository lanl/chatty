"""Configuration management for chatty."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Chatty configuration with environment variable and TOML file support."""

    model_config = SettingsConfigDict(
        env_prefix="CHATTY_",
        env_file=".env",
        extra="ignore",
    )

    # LLM Endpoint (OPENAI_BASE_URL or CHATTY_BASE_URL)
    base_url: str = ""
    model: str = "gpt-4.1"

    # Authentication (OPENAI_API_KEY or CHATTY_API_KEY)
    api_key: SecretStr | None = None
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


def load_config() -> Config:
    """Load configuration from environment and config file."""
    return Config()
