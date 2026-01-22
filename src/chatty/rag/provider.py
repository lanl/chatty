"""RAG provider protocol and implementations."""

from dataclasses import dataclass, field
from typing import Protocol

from chatty.client.openai_client import Message
from chatty.core.conversation import Conversation


@dataclass
class RAGSource:
    """A single source document from RAG retrieval.

    Attributes:
        title: Document/paper title.
        pmid: PubMed ID if available.
        pmcid: PubMed Central ID if available.
        snippet: Preview text from the retrieved chunk (first ~200 chars).
        full_text: Complete text of the retrieved chunk.
        score: Relevance score from retrieval (higher is better).
    """

    title: str
    pmid: str | None = None
    pmcid: str | None = None
    snippet: str = ""
    full_text: str = ""
    score: float = 0.0


@dataclass
class RAGMetadata:
    """Metadata about retrieved documents.

    Attributes:
        sources: List of RAGSource objects with structured source info.
        retrieval_time_s: Time taken for retrieval in seconds.
        chunk_count: Number of chunks retrieved before budget fitting.
    """

    sources: list[RAGSource] = field(default_factory=list)
    retrieval_time_s: float = 0.0
    chunk_count: int = 0


class RAGProvider(Protocol):
    """Protocol defining the RAG augmentation interface."""

    async def augment(
        self,
        conversation: Conversation,
        user_text: str,
    ) -> tuple[list[Message], RAGMetadata]:
        """Augment messages with retrieved context.

        Args:
            conversation: Current conversation history
            user_text: New user input

        Returns:
            Tuple of (augmented messages, retrieval metadata)
        """
        ...


class NullProvider:
    """Passthrough provider — no augmentation (v0.1 default)."""

    async def augment(
        self,
        conversation: Conversation,
        user_text: str,
    ) -> tuple[list[Message], RAGMetadata]:
        """Return messages unchanged with empty metadata."""
        messages = list(conversation.messages) + [Message(role="user", content=user_text)]
        return messages, RAGMetadata(sources=[])
