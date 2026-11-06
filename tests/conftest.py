"""Shared pytest fixtures for chatty tests."""

import contextlib
import os
import socket
import tempfile
from collections.abc import Iterator
from typing import Any

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


_LOOPBACK = {"localhost", "127.0.0.1", "::1"}
_real_getaddrinfo = socket.getaddrinfo


@pytest.fixture(scope="session", autouse=True)
def tiktoken_encodings() -> None:
    """Load the tiktoken encodings the code uses before DNS is blocked.

    Session-scoped, so pytest runs it before the function-scoped block_dns.
    tiktoken downloads them on first use and caches them, so a fresh machine needs
    the network once. If the download fails, the tests that count tokens fail.
    """
    import tiktoken

    for name in ("cl100k_base", "o200k_base"):
        with contextlib.suppress(Exception):  # the tests that need it report it
            tiktoken.get_encoding(name)


@pytest.fixture(autouse=True)
def block_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse DNS lookups for anything but loopback.

    The suite must not reach a real endpoint. Without this, an app test whose
    config leaves context_window = "auto" looks up its fake base_url host at
    startup, and teardown waits for the lookup: about 5 s per test on macOS, where
    a .local name goes to mDNS, and a hang on the GitHub macOS runner.
    """

    def guarded(host: Any, *args: Any, **kwargs: Any) -> Any:
        name = host.decode() if isinstance(host, bytes) else host
        if name not in _LOOPBACK:
            raise socket.gaierror(socket.EAI_NONAME, f"DNS blocked in tests: {name}")
        return _real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded)


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
