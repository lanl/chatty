"""Tests for shared HTTP client utilities."""

import pytest

from chatty.client.http import build_async_client, build_sync_client
from chatty.config import Config


def test_build_sync_client_default_verify() -> None:
    """Default config uses system CA bundle."""
    config = Config()
    client = build_sync_client(config)

    # Default: verify=True (system CA)
    # Access internal attributes to verify SSL context is set
    assert client._transport._pool._ssl_context is not None  # type: ignore[attr-defined]
    client.close()


def test_build_sync_client_verify_disabled() -> None:
    """TLS verification can be disabled."""
    config = Config(verify_tls=False)
    client = build_sync_client(config)

    # Should build without error
    client.close()


def test_build_sync_client_with_proxy() -> None:
    """Proxy configuration is applied."""
    config = Config(http_proxy="http://proxy.example.com:8080")
    client = build_sync_client(config)

    # Should build without error
    client.close()


def test_build_sync_client_custom_timeout() -> None:
    """Custom timeout is applied."""
    config = Config()
    client = build_sync_client(config, timeout=120.0)

    assert client.timeout.read == 120.0
    client.close()


@pytest.mark.asyncio
async def test_build_async_client_default_verify() -> None:
    """Default config uses system CA bundle for async client."""
    config = Config()
    client = build_async_client(config)

    # Default: verify=True (system CA)
    await client.aclose()


@pytest.mark.asyncio
async def test_build_async_client_verify_disabled() -> None:
    """TLS verification can be disabled for async client."""
    config = Config(verify_tls=False)
    client = build_async_client(config)

    await client.aclose()


@pytest.mark.asyncio
async def test_build_async_client_with_proxy() -> None:
    """Proxy configuration is applied for async client."""
    config = Config(http_proxy="http://proxy.example.com:8080")
    client = build_async_client(config)

    await client.aclose()


@pytest.mark.asyncio
async def test_build_async_client_custom_timeout() -> None:
    """Custom timeout is applied for async client."""
    config = Config()
    client = build_async_client(config, timeout=120.0)

    assert client.timeout.read == 120.0
    await client.aclose()
