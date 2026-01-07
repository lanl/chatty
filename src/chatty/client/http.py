"""Shared HTTP client utilities for chatty."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from chatty.config import Config


def build_sync_client(config: Config, *, timeout: float = 60.0) -> httpx.Client:
    """Build a synchronous httpx client with TLS/proxy settings.

    Args:
        config: Configuration object with TLS/proxy settings
        timeout: Request timeout in seconds

    Returns:
        Configured httpx Client
    """
    # TLS verification: path to CA bundle, True for system default, False to disable
    verify: bool | str = True
    if config.ca_bundle:
        verify = config.ca_bundle
    if not config.verify_tls:
        verify = False

    # Proxy configuration
    proxy = config.http_proxy if config.http_proxy else None

    return httpx.Client(
        verify=verify,
        proxy=proxy,
        timeout=httpx.Timeout(timeout),
    )


def build_async_client(config: Config, *, timeout: float = 60.0) -> httpx.AsyncClient:
    """Build an asynchronous httpx client with TLS/proxy settings.

    Args:
        config: Configuration object with TLS/proxy settings
        timeout: Request timeout in seconds

    Returns:
        Configured httpx AsyncClient
    """
    # TLS verification: path to CA bundle, True for system default, False to disable
    verify: bool | str = True
    if config.ca_bundle:
        verify = config.ca_bundle
    if not config.verify_tls:
        verify = False

    # Proxy configuration
    proxy = config.http_proxy if config.http_proxy else None

    return httpx.AsyncClient(
        verify=verify,
        proxy=proxy,
        timeout=httpx.Timeout(timeout),
    )
