"""RAG provider protocol and implementations."""

from dataclasses import dataclass, field
from typing import Protocol

from chatty.client.openai_client import Message
from chatty.core.conversation import Conversation


@dataclass
class RAGMetadata:
    """Metadata about retrieved documents."""

    sources: list[dict[str, str]] = field(default_factory=list)


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
