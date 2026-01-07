"""Shared pytest fixtures for chatty tests."""

from collections.abc import Iterator

import pytest
from pydantic import SecretStr

from chatty.config import Config


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
