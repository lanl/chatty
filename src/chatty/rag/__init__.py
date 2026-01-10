"""RAG provider protocol and implementations."""

from chatty.config import Config
from chatty.rag.provider import NullProvider, RAGMetadata, RAGProvider


class UnknownProviderError(ValueError):
    """Raised when an unknown RAG provider is requested."""

    pass


def get_provider(config: Config) -> RAGProvider:
    """Factory function to create a RAG provider based on config.

    Args:
        config: Application configuration containing rag_provider setting.

    Returns:
        RAGProvider instance (NullProvider for "none", LitkitProvider for "litkit" in v0.3+)

    Raises:
        UnknownProviderError: If the configured provider is not recognized.
    """
    provider_name = config.rag_provider.lower()

    if provider_name == "none":
        return NullProvider()

    # litkit provider will be added in v0.3
    # if provider_name == "litkit":
    #     from chatty.rag.litkit_provider import LitkitProvider
    #     return LitkitProvider(...)

    raise UnknownProviderError(
        f"Unknown RAG provider: '{config.rag_provider}'. "
        f"Valid options: 'none'. (litkit support coming in v0.3)"
    )


__all__ = [
    "RAGProvider",
    "NullProvider",
    "RAGMetadata",
    "UnknownProviderError",
    "get_provider",
]
