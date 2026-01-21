"""RAG provider protocol and implementations."""

from chatty.config import Config
from chatty.rag.provider import NullProvider, RAGMetadata, RAGProvider


class UnknownProviderError(ValueError):
    """Raised when an unknown RAG provider is requested."""

    pass


class ConfigurationError(ValueError):
    """Raised when RAG configuration is invalid."""

    pass


def get_provider(config: Config) -> RAGProvider:
    """Factory function to create a RAG provider based on config.

    Args:
        config: Application configuration containing rag_provider setting.

    Returns:
        RAGProvider instance (NullProvider for "none", LitkitProvider for "litkit")

    Raises:
        UnknownProviderError: If the configured provider is not recognized.
        ConfigurationError: If the provider configuration is invalid.
    """
    provider_name = config.rag_provider.lower()

    if provider_name == "none":
        return NullProvider()

    if provider_name == "litkit":
        return _create_litkit_provider(config)

    raise UnknownProviderError(
        f"Unknown RAG provider: '{config.rag_provider}'. " f"Valid options: 'none', 'litkit'."
    )


def _create_litkit_provider(config: Config) -> RAGProvider:
    """Create a LitkitProvider with configuration.

    Args:
        config: Application configuration.

    Returns:
        LitkitProvider instance.

    Raises:
        ConfigurationError: If litkit is not installed or workspace is not configured.
    """
    # Import here to avoid import errors if litkit is not installed
    try:
        from chatty.rag.litkit_provider import (
            LitkitError,
            LitkitNotInstalledError,
            LitkitProvider,
        )
    except ImportError:
        raise ConfigurationError(
            "litkit package not installed. "
            "Run `uv add litkit` or set `rag_provider = 'none'` in config."
        ) from None

    # Resolve workspace path
    workspace = config.get_rag_workspace()
    if workspace is None:
        raise ConfigurationError(
            "RAG workspace not configured. "
            "Set 'rag_workspace' in config or LITKIT_WORKSPACE environment variable."
        )

    try:
        return LitkitProvider(
            workspace=workspace,
            top_papers=config.rag_top_papers,
            top_chunks=config.rag_top_chunks,
        )
    except LitkitNotInstalledError:
        raise ConfigurationError(
            "litkit package not installed. "
            "Run `uv add litkit` or set `rag_provider = 'none'` in config."
        ) from None
    except LitkitError as e:
        # Re-raise litkit errors as configuration errors with clear messages
        raise ConfigurationError(str(e)) from e


__all__ = [
    "RAGProvider",
    "NullProvider",
    "RAGMetadata",
    "UnknownProviderError",
    "ConfigurationError",
    "get_provider",
]
