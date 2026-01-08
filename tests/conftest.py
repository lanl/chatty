"""Shared pytest fixtures for chatty tests."""

import os
import tempfile
from collections.abc import Iterator

import pytest
from pydantic import SecretStr

from chatty.config import Config


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate tests from user config files.

    Sets HOME and XDG_CONFIG_HOME to temp directories so tests don't pick up
    ~/.config/chatty/config.toml or other user-specific configuration.

    This fixture runs automatically for all tests.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("HOME", tmpdir)
        monkeypatch.setenv("XDG_CONFIG_HOME", os.path.join(tmpdir, ".config"))
        # Also clear any chatty-specific env vars that might interfere
        for var in list(os.environ.keys()):
            if var.startswith("CHATTY_") or var.startswith("OPENAI_"):
                monkeypatch.delenv(var, raising=False)
        yield


@pytest.fixture
def mock_config() -> Iterator[Config]:
    """Create a test configuration."""
    yield Config(
        base_url="https://api.test/v1",
        model="gpt-4-test",
        api_key=SecretStr("test-api-key"),
        verify_tls=True,
        temperature=0.2,
        stream=True,
        timeout_s=30,
    )


@pytest.fixture
def sample_messages() -> Iterator[list[dict[str, str]]]:
    """Create sample conversation messages."""
    yield [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
    ]
