"""Tests for RAG provider protocol and implementations."""

import pytest

from chatty.client.openai_client import Message
from chatty.core.conversation import Conversation
from chatty.rag.provider import NullProvider, RAGMetadata, RAGSource

# === RAGSource Tests ===


def test_rag_source_minimal() -> None:
    """Test RAGSource with only required field."""
    source = RAGSource(title="Test Paper")
    assert source.title == "Test Paper"
    assert source.pmid is None
    assert source.pmcid is None
    assert source.snippet == ""
    assert source.score == 0.0


def test_rag_source_full() -> None:
    """Test RAGSource with all fields."""
    source = RAGSource(
        title="Test Paper",
        pmid="12345678",
        pmcid="PMC9876543",
        snippet="This is a test snippet...",
        score=0.95,
    )
    assert source.title == "Test Paper"
    assert source.pmid == "12345678"
    assert source.pmcid == "PMC9876543"
    assert source.snippet == "This is a test snippet..."
    assert source.score == 0.95


def test_rag_source_equality() -> None:
    """Test RAGSource dataclass equality."""
    source1 = RAGSource(title="Paper A", pmid="123")
    source2 = RAGSource(title="Paper A", pmid="123")
    source3 = RAGSource(title="Paper B", pmid="123")

    assert source1 == source2
    assert source1 != source3


# === RAGMetadata Tests ===


def test_rag_metadata_init() -> None:
    """Test RAGMetadata initialization with defaults."""
    metadata = RAGMetadata()
    assert metadata.sources == []
    assert metadata.retrieval_time_s == 0.0
    assert metadata.chunk_count == 0


def test_rag_metadata_with_sources() -> None:
    """Test RAGMetadata with RAGSource objects."""
    sources = [
        RAGSource(title="Doc 1", score=0.9),
        RAGSource(title="Doc 2", score=0.8),
    ]
    metadata = RAGMetadata(sources=sources)

    assert len(metadata.sources) == 2
    assert metadata.sources[0].title == "Doc 1"
    assert metadata.sources[0].score == 0.9


def test_rag_metadata_with_retrieval_time() -> None:
    """Test RAGMetadata with retrieval timing info."""
    sources = [RAGSource(title="Paper A")]
    metadata = RAGMetadata(
        sources=sources,
        retrieval_time_s=1.5,
        chunk_count=30,
    )

    assert metadata.retrieval_time_s == 1.5
    assert metadata.chunk_count == 30


def test_rag_metadata_empty_sources_with_time() -> None:
    """Test RAGMetadata with no sources but retrieval time."""
    metadata = RAGMetadata(
        sources=[],
        retrieval_time_s=0.8,
        chunk_count=0,
    )

    assert len(metadata.sources) == 0
    assert metadata.retrieval_time_s == 0.8


@pytest.mark.asyncio
async def test_null_provider_augment_empty_conversation() -> None:
    """Test NullProvider with empty conversation."""
    provider = NullProvider()
    conv = Conversation()

    messages, metadata = await provider.augment(conv, "Hello!")

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hello!"
    assert metadata.sources == []


@pytest.mark.asyncio
async def test_null_provider_augment_with_history() -> None:
    """Test NullProvider preserves conversation history."""
    provider = NullProvider()
    conv = Conversation()
    conv.add_system_message("You are helpful.")
    conv.add_user_message("Previous question")
    conv.add_assistant_message("Previous answer")

    messages, metadata = await provider.augment(conv, "New question")

    assert len(messages) == 4
    assert messages[0].role == "system"
    assert messages[0].content == "You are helpful."
    assert messages[1].role == "user"
    assert messages[1].content == "Previous question"
    assert messages[2].role == "assistant"
    assert messages[2].content == "Previous answer"
    assert messages[3].role == "user"
    assert messages[3].content == "New question"


@pytest.mark.asyncio
async def test_null_provider_returns_empty_metadata() -> None:
    """Test NullProvider always returns empty metadata."""
    provider = NullProvider()
    conv = Conversation()

    _, metadata = await provider.augment(conv, "Any query")

    assert isinstance(metadata, RAGMetadata)
    assert metadata.sources == []
    assert metadata.retrieval_time_s == 0.0
    assert metadata.chunk_count == 0


@pytest.mark.asyncio
async def test_null_provider_does_not_modify_original_conversation() -> None:
    """Test NullProvider doesn't modify the original conversation."""
    provider = NullProvider()
    conv = Conversation()
    conv.add_user_message("Original")

    original_len = len(conv.messages)
    await provider.augment(conv, "New message")

    # Original conversation should be unchanged
    assert len(conv.messages) == original_len


@pytest.mark.asyncio
async def test_null_provider_message_types() -> None:
    """Test NullProvider returns correct Message types."""
    provider = NullProvider()
    conv = Conversation()

    messages, _ = await provider.augment(conv, "Test")

    assert all(isinstance(m, Message) for m in messages)


def test_null_provider_implements_protocol() -> None:
    """Test NullProvider satisfies RAGProvider protocol."""
    # This is a structural check - NullProvider should have augment method
    provider = NullProvider()
    assert hasattr(provider, "augment")
    assert callable(provider.augment)
