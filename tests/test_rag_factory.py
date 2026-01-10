"""Tests for RAG provider factory."""

import pytest

from chatty.config import Config
from chatty.rag import NullProvider, UnknownProviderError, get_provider


def test_get_provider_returns_null_provider_by_default() -> None:
    """Default config returns NullProvider."""
    config = Config()
    provider = get_provider(config)

    assert isinstance(provider, NullProvider)


def test_get_provider_returns_null_provider_explicit() -> None:
    """Explicit 'none' config returns NullProvider."""
    config = Config(rag_provider="none")
    provider = get_provider(config)

    assert isinstance(provider, NullProvider)


def test_get_provider_case_insensitive() -> None:
    """Provider name is case-insensitive."""
    config = Config(rag_provider="NONE")
    provider = get_provider(config)

    assert isinstance(provider, NullProvider)

    config2 = Config(rag_provider="None")
    provider2 = get_provider(config2)

    assert isinstance(provider2, NullProvider)


def test_get_provider_unknown_raises_error() -> None:
    """Unknown provider raises UnknownProviderError."""
    config = Config(rag_provider="unknown_provider")

    with pytest.raises(UnknownProviderError) as exc_info:
        get_provider(config)

    assert "unknown_provider" in str(exc_info.value)
    assert "Valid options" in str(exc_info.value)


def test_get_provider_litkit_not_implemented_yet() -> None:
    """Litkit provider raises error (not implemented in v0.2)."""
    config = Config(rag_provider="litkit")

    with pytest.raises(UnknownProviderError) as exc_info:
        get_provider(config)

    # Should mention litkit is coming in v0.3
    assert "litkit" in str(exc_info.value).lower()
    assert "v0.3" in str(exc_info.value)


def test_config_rag_provider_default() -> None:
    """Config has rag_provider defaulting to 'none'."""
    config = Config()
    assert config.rag_provider == "none"


def test_config_rag_provider_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """rag_provider can be set via environment variable."""
    monkeypatch.setenv("CHATTY_RAG_PROVIDER", "none")

    config = Config()
    assert config.rag_provider == "none"


def test_config_rag_provider_in_display() -> None:
    """rag_provider appears in config display."""
    config = Config(rag_provider="none")
    display = config.display()

    assert "rag_provider" in display
    assert display["rag_provider"] == "none"
