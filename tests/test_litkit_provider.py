"""Tests for LitkitProvider."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chatty.core.conversation import Conversation
from chatty.rag.provider import RAGMetadata, RAGSource


class TestLitkitProviderInit:
    """Tests for LitkitProvider initialization."""

    def test_init_raises_when_litkit_not_installed(self, tmp_path: Path) -> None:
        """LitkitProvider raises LitkitNotInstalledError if litkit not installed."""
        # Create minimal workspace with standard litkit layout
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with (
            patch.dict("sys.modules", {"litkit": None}),
            patch(
                "chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"
            ) as mock_check,
        ):
            from chatty.rag.litkit_provider import LitkitNotInstalledError

            mock_check.side_effect = LitkitNotInstalledError()

            with pytest.raises(LitkitNotInstalledError) as exc_info:
                from chatty.rag.litkit_provider import LitkitProvider

                LitkitProvider(workspace=tmp_path)

            assert "litkit package not installed" in str(exc_info.value)

    def test_init_raises_workspace_not_found(self) -> None:
        """LitkitProvider raises WorkspaceNotFoundError if workspace doesn't exist."""
        from chatty.rag.litkit_provider import WorkspaceNotFoundError

        # Mock litkit being installed
        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            with pytest.raises(WorkspaceNotFoundError) as exc_info:
                LitkitProvider(workspace=Path("/nonexistent/workspace"))

            assert "workspace not found" in str(exc_info.value)

    def test_init_raises_faiss_not_found(self, tmp_path: Path) -> None:
        """LitkitProvider raises FAISSIndexNotFoundError if FAISS indices missing."""
        from chatty.rag.litkit_provider import FAISSIndexNotFoundError

        # Create workspace without indices/ directory
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            with pytest.raises(FAISSIndexNotFoundError) as exc_info:
                LitkitProvider(workspace=tmp_path)

            assert "FAISS index not found" in str(exc_info.value)

    def test_init_raises_database_not_found(self, tmp_path: Path) -> None:
        """LitkitProvider raises DatabaseNotFoundError if SQLite DB missing."""
        from chatty.rag.litkit_provider import DatabaseNotFoundError

        # Create workspace with indices but no sqlite/
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            with pytest.raises(DatabaseNotFoundError) as exc_info:
                LitkitProvider(workspace=tmp_path)

            assert "SQLite database not found" in str(exc_info.value)

    def test_init_success_with_valid_workspace(self, tmp_path: Path) -> None:
        """LitkitProvider initializes successfully with valid workspace."""
        # Create valid workspace with standard litkit layout
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            provider = LitkitProvider(
                workspace=tmp_path,
                top_papers=100,
                top_chunks=10,
            )

            assert provider._workspace == tmp_path
            assert provider._top_papers == 100
            assert provider._top_chunks == 10
            assert provider._db_path == tmp_path / "sqlite" / "litkit.sqlite3"
            assert provider._indices_dir == tmp_path / "indices"

    def test_init_accepts_chunks_only(self, tmp_path: Path) -> None:
        """LitkitProvider accepts workspace with only chunks.faiss."""
        # Create workspace with only chunks.faiss (no papers.faiss)
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "chunks.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            provider = LitkitProvider(workspace=tmp_path)

            assert provider._workspace == tmp_path


class TestLitkitProviderRetrieve:
    """Tests for LitkitProvider retrieval logic.

    Note: _retrieve_chunks now runs in a subprocess for isolation from Textual.
    Tests mock subprocess.run() instead of litkit modules directly.
    """

    @pytest.fixture
    def valid_workspace(self, tmp_path: Path) -> Path:
        """Create a valid workspace directory with standard litkit layout."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()
        return tmp_path

    def test_retrieve_chunks_empty_result(self, valid_workspace: Path) -> None:
        """_retrieve_chunks handles empty paper shortlist (returns empty tuple)."""
        import json

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"chunks": [], "paper_ids": []})
        mock_result.stderr = ""

        with (
            patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"),
            patch("chatty.rag.litkit_provider.subprocess.run", return_value=mock_result),
        ):
            from chatty.rag.litkit_provider import LitkitProvider

            provider = LitkitProvider(workspace=valid_workspace)
            chunks, paper_ids = provider._retrieve_chunks("test query")

            assert chunks == []
            assert paper_ids == []

    def test_retrieve_chunks_success(self, valid_workspace: Path) -> None:
        """_retrieve_chunks returns RetrievedChunk objects with metadata and paper_ids."""
        import json

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": 10,
                        "text": "Chunk 1 text about HIV treatments.",
                        "paper_title": "Paper A",
                        "pmid": "12345",
                        "pmcid": None,
                        "score": 0.9,
                    },
                    {
                        "chunk_id": 20,
                        "text": "Chunk 2 text about side effects.",
                        "paper_title": "Paper B",
                        "pmid": None,
                        "pmcid": "PMC67890",
                        "score": 0.8,
                    },
                ],
                "paper_ids": [100, 200, 300],
            }
        )
        mock_result.stderr = ""

        with (
            patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"),
            patch("chatty.rag.litkit_provider.subprocess.run", return_value=mock_result),
        ):
            from chatty.rag.litkit_provider import LitkitProvider

            provider = LitkitProvider(workspace=valid_workspace)
            chunks, paper_ids = provider._retrieve_chunks("HIV treatments")

            assert len(chunks) == 2
            assert chunks[0].chunk_id == 10
            assert chunks[0].text == "Chunk 1 text about HIV treatments."
            assert chunks[0].paper_title == "Paper A"
            assert chunks[0].pmid == "12345"
            assert chunks[0].pmcid is None
            assert chunks[0].score == 0.9

            assert chunks[1].chunk_id == 20
            assert chunks[1].paper_title == "Paper B"
            assert chunks[1].pmcid == "PMC67890"
            assert chunks[1].score == 0.8

            # Also verify paper_ids returned
            assert paper_ids == [100, 200, 300]

    def test_retrieve_chunks_subprocess_error(self, valid_workspace: Path) -> None:
        """_retrieve_chunks raises LitkitError on subprocess failure."""
        import json

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = json.dumps({"error": "Test error"})

        with (
            patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"),
            patch("chatty.rag.litkit_provider.subprocess.run", return_value=mock_result),
        ):
            from chatty.rag.litkit_provider import LitkitError, LitkitProvider

            provider = LitkitProvider(workspace=valid_workspace)
            with pytest.raises(LitkitError) as exc_info:
                provider._retrieve_chunks("test query")

            assert "Test error" in str(exc_info.value)

    def test_retrieve_chunks_timeout(self, valid_workspace: Path) -> None:
        """_retrieve_chunks raises LitkitError on timeout."""
        import subprocess

        with (
            patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"),
            patch(
                "chatty.rag.litkit_provider.subprocess.run",
                side_effect=subprocess.TimeoutExpired("cmd", 120),
            ),
        ):
            from chatty.rag.litkit_provider import LitkitError, LitkitProvider

            provider = LitkitProvider(workspace=valid_workspace)
            with pytest.raises(LitkitError) as exc_info:
                provider._retrieve_chunks("test query")

            assert "timed out" in str(exc_info.value)

    def test_retrieve_chunks_with_reuse_paper_ids(self, valid_workspace: Path) -> None:
        """_retrieve_chunks passes paper_ids to subprocess for reuse."""
        import json

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"chunks": [], "paper_ids": [1, 2, 3]})
        mock_result.stderr = ""

        with (
            patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"),
            patch(
                "chatty.rag.litkit_provider.subprocess.run", return_value=mock_result
            ) as mock_run,
        ):
            from chatty.rag.litkit_provider import LitkitProvider

            provider = LitkitProvider(workspace=valid_workspace)
            provider._retrieve_chunks("test query", reuse_paper_ids=[1, 2, 3])

            # Check that paper_ids were passed as JSON in subprocess args
            call_args = mock_run.call_args[0][0]
            # Last argument should be JSON-encoded paper_ids
            assert json.loads(call_args[-1]) == [1, 2, 3]


class TestLitkitProviderTokenBudget:
    """Tests for token budget fitting logic."""

    @pytest.fixture
    def provider(self, tmp_path: Path) -> Any:
        """Create a LitkitProvider for testing."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            return LitkitProvider(workspace=tmp_path)

    def test_fit_to_budget_empty_chunks(self, provider: Any) -> None:
        """_fit_to_budget returns empty list for empty input."""
        result = provider._fit_to_budget([], 1000)
        assert result == []

    def test_fit_to_budget_zero_budget(self, provider: Any) -> None:
        """_fit_to_budget returns empty list for zero budget."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [RetrievedChunk(chunk_id=1, text="Test chunk", paper_title="Paper", score=0.9)]
        result = provider._fit_to_budget(chunks, 0)
        assert result == []

    def test_fit_to_budget_negative_budget(self, provider: Any) -> None:
        """_fit_to_budget returns empty list for negative budget."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [RetrievedChunk(chunk_id=1, text="Test chunk", paper_title="Paper", score=0.9)]
        result = provider._fit_to_budget(chunks, -100)
        assert result == []

    def test_fit_to_budget_all_fit(self, provider: Any) -> None:
        """_fit_to_budget returns all chunks when they fit in budget."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(chunk_id=1, text="Short text", paper_title="Paper 1", score=0.9),
            RetrievedChunk(chunk_id=2, text="Another text", paper_title="Paper 2", score=0.8),
        ]
        # Large budget should fit all chunks
        result = provider._fit_to_budget(chunks, 10000)
        assert len(result) == 2

    def test_fit_to_budget_orders_by_score(self, provider: Any) -> None:
        """_fit_to_budget orders chunks by score (highest first)."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(chunk_id=1, text="Low score", paper_title="Paper 1", score=0.5),
            RetrievedChunk(chunk_id=2, text="High score", paper_title="Paper 2", score=0.9),
            RetrievedChunk(chunk_id=3, text="Mid score", paper_title="Paper 3", score=0.7),
        ]
        result = provider._fit_to_budget(chunks, 10000)
        assert result[0].score == 0.9
        assert result[1].score == 0.7
        assert result[2].score == 0.5

    def test_fit_to_budget_truncates_when_needed(self, provider: Any) -> None:
        """_fit_to_budget truncates chunks list when budget exceeded."""
        from chatty.rag.litkit_provider import RetrievedChunk

        # Create chunks with substantial text
        chunks = [
            RetrievedChunk(
                chunk_id=i,
                text="A" * 1000,  # ~250 tokens each
                paper_title=f"Paper {i}",
                score=0.9 - i * 0.1,
            )
            for i in range(10)
        ]
        # Small budget should only fit a few chunks
        result = provider._fit_to_budget(chunks, 500)
        assert len(result) < 10

    def test_estimate_tokens_with_tiktoken(self, provider: Any) -> None:
        """_estimate_tokens uses tiktoken when available."""
        # tiktoken should be available in test environment
        tokens = provider._estimate_tokens("Hello, world!")
        assert isinstance(tokens, int)
        assert tokens > 0
        assert tokens < 100  # Should be small for short text

    def test_estimate_tokens_fallback(self, provider: Any) -> None:
        """_estimate_tokens falls back to character-based estimate."""
        with patch.dict("sys.modules", {"tiktoken": None}):
            # Force ImportError for tiktoken
            text = "Hello, world! This is a test."
            tokens = provider._estimate_tokens(text)
            # Fallback: ~4 chars per token
            assert tokens == len(text) // 4

    def test_truncate_to_tokens(self, provider: Any) -> None:
        """_truncate_to_tokens truncates text to fit token budget."""
        long_text = "This is a very long text. " * 100
        truncated = provider._truncate_to_tokens(long_text, 10)
        assert len(truncated) < len(long_text)
        assert truncated.endswith("...")

    def test_truncate_to_tokens_short_text(self, provider: Any) -> None:
        """_truncate_to_tokens returns original text if under budget."""
        short_text = "Hello"
        truncated = provider._truncate_to_tokens(short_text, 100)
        assert truncated == short_text


class TestLitkitProviderFormatting:
    """Tests for context formatting and metadata building."""

    @pytest.fixture
    def provider(self, tmp_path: Path) -> Any:
        """Create a LitkitProvider for testing."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            return LitkitProvider(workspace=tmp_path)

    def test_format_context_empty(self, provider: Any) -> None:
        """_format_context returns empty string for empty chunks."""
        result = provider._format_context([], turn=1)
        assert result == ""

    def test_format_context_with_chunks(self, provider: Any) -> None:
        """_format_context formats chunks with source info."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(
                chunk_id=1,
                text="Test content about HIV.",
                paper_title="Paper A",
                pmid="12345",
                score=0.9,
            ),
            RetrievedChunk(
                chunk_id=2,
                text="More content about treatments.",
                paper_title="Paper B",
                pmcid="PMC67890",
                score=0.8,
            ),
        ]
        result = provider._format_context(chunks, turn=1)

        # Uses [T.N] format where T is turn number
        assert "[1.1] Paper A (PMID: 12345)" in result
        assert "Test content about HIV." in result
        assert "[1.2] Paper B (PMCID: PMC67890)" in result
        assert "More content about treatments." in result
        assert "[T.N]" in result  # Citation format explanation

    def test_build_metadata(self, provider: Any) -> None:
        """_build_metadata creates RAGMetadata with RAGSource objects."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(
                chunk_id=1,
                text="Short text",
                paper_title="Paper A",
                pmid="12345",
                score=0.9,
            ),
        ]
        metadata = provider._build_metadata(chunks, total_chunks=5, retrieval_time_s=1.2)

        assert isinstance(metadata, RAGMetadata)
        assert len(metadata.sources) == 1
        assert isinstance(metadata.sources[0], RAGSource)
        assert metadata.sources[0].title == "Paper A"
        assert metadata.sources[0].pmid == "12345"
        assert metadata.sources[0].score == 0.9
        assert metadata.sources[0].snippet == "Short text"
        assert metadata.retrieval_time_s == 1.2
        assert metadata.chunk_count == 5

    def test_build_metadata_truncates_long_snippets(self, provider: Any) -> None:
        """_build_metadata truncates long snippets in metadata."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(
                chunk_id=1,
                text="A" * 500,  # Long text
                paper_title="Paper A",
                score=0.9,
            ),
        ]
        metadata = provider._build_metadata(chunks, total_chunks=1, retrieval_time_s=0.5)

        assert len(metadata.sources[0].snippet) <= 203  # 200 + "..."

    def test_build_metadata_with_pmcid(self, provider: Any) -> None:
        """_build_metadata includes PMCID in RAGSource."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(
                chunk_id=1,
                text="Some text",
                paper_title="Paper B",
                pmcid="PMC12345",
                score=0.8,
            ),
        ]
        metadata = provider._build_metadata(chunks, total_chunks=1, retrieval_time_s=0.3)

        assert metadata.sources[0].pmcid == "PMC12345"
        assert metadata.sources[0].pmid is None


class TestLitkitProviderAugment:
    """Tests for the async augment method."""

    @pytest.fixture
    def provider(self, tmp_path: Path) -> Any:
        """Create a LitkitProvider for testing."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            return LitkitProvider(workspace=tmp_path)

    @pytest.fixture
    def conversation(self) -> Conversation:
        """Create a conversation for testing."""
        conv = Conversation(model="gpt-4")
        conv.add_system_message("You are a helpful assistant.")
        conv.add_user_message("Previous question")
        conv.add_assistant_message("Previous answer")
        return conv

    @pytest.mark.asyncio
    async def test_augment_no_chunks_found(self, provider: Any, conversation: Conversation) -> None:
        """augment returns original messages when no chunks found."""
        with patch.object(provider, "_retrieve_chunks", return_value=([], [])):
            messages, metadata = await provider.augment(conversation, "new question")

        # Should have original messages plus new user message
        assert len(messages) == len(conversation.messages) + 1
        assert messages[-1].role == "user"
        assert messages[-1].content == "new question"
        assert metadata.sources == []
        # Should still have retrieval time even with no results
        assert metadata.retrieval_time_s >= 0

    @pytest.mark.asyncio
    async def test_augment_with_chunks(self, provider: Any, conversation: Conversation) -> None:
        """augment injects context into user message."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(
                chunk_id=1,
                text="Relevant content",
                paper_title="Paper A",
                pmid="12345",
                score=0.9,
            ),
        ]
        with (
            patch.object(provider, "_retrieve_chunks", return_value=(chunks, [100])),
            patch.object(provider, "_fit_to_budget", return_value=chunks),
        ):
            messages, metadata = await provider.augment(conversation, "new question")

        # User message should contain context
        assert len(messages) == len(conversation.messages) + 1
        assert messages[-1].role == "user"
        assert "Relevant content" in messages[-1].content
        assert "Question: new question" in messages[-1].content
        assert len(metadata.sources) == 1
        # Check RAGSource structure
        assert isinstance(metadata.sources[0], RAGSource)
        assert metadata.sources[0].title == "Paper A"
        assert metadata.sources[0].pmid == "12345"
        assert metadata.retrieval_time_s >= 0
        assert metadata.chunk_count == 1

    @pytest.mark.asyncio
    async def test_augment_respects_token_budget(
        self, provider: Any, conversation: Conversation
    ) -> None:
        """augment calls _fit_to_budget with calculated available tokens."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(
                chunk_id=1,
                text="Content",
                paper_title="Paper",
                score=0.9,
            ),
        ]

        fit_called_with: list[tuple[Any, ...]] = []

        def mock_fit(chunks_arg: list[Any], available: int) -> list[Any]:
            fit_called_with.append((chunks_arg, available))
            return chunks_arg

        with (
            patch.object(provider, "_retrieve_chunks", return_value=(chunks, [100])),
            patch.object(provider, "_fit_to_budget", side_effect=mock_fit),
        ):
            await provider.augment(conversation, "question")

        assert len(fit_called_with) == 1
        # Should have calculated available budget based on conversation state
        assert fit_called_with[0][1] > 0


class TestLitkitProviderRewriter:
    """Tests for LitkitProvider query rewriting integration."""

    @pytest.fixture
    def provider(self, tmp_path: Path) -> Any:
        """Create a LitkitProvider for testing."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            return LitkitProvider(workspace=tmp_path, rewrite_enabled=True)

    @pytest.fixture
    def provider_rewrite_disabled(self, tmp_path: Path) -> Any:
        """Create a LitkitProvider with rewriting disabled."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            return LitkitProvider(workspace=tmp_path, rewrite_enabled=False)

    @pytest.fixture
    def conversation(self) -> Conversation:
        """Create a conversation with history for testing."""
        conv = Conversation(model="gpt-4")
        conv.add_system_message("You are a helpful assistant.")
        conv.add_user_message("Tell me about HIV treatments")
        conv.add_assistant_message("HIV treatments include various antiretroviral drugs...")
        return conv

    @pytest.mark.asyncio
    async def test_augment_rewrites_short_query(
        self, provider: Any, conversation: Conversation
    ) -> None:
        """augment() rewrites short follow-up queries when client provided."""
        from chatty.client.openai_client import AssistantMessage, OpenAIClient

        # Use spec=OpenAIClient so isinstance check passes
        client = MagicMock(spec=OpenAIClient)
        client.chat = AsyncMock(
            return_value=AssistantMessage(content="What are the side effects of HIV treatments?")
        )

        with patch.object(provider, "_retrieve_chunks", return_value=([], [])) as mock_retrieve:
            await provider.augment(conversation, "What about side effects?", client=client)

            # Rewriter should have been called (client.chat was called)
            client.chat.assert_called_once()
            # Retrieval should use rewritten query (with None for reuse_paper_ids)
            mock_retrieve.assert_called_once()
            call_args = mock_retrieve.call_args
            assert call_args[0][0] == "What are the side effects of HIV treatments?"

    @pytest.mark.asyncio
    async def test_augment_skips_rewrite_without_client(
        self, provider: Any, conversation: Conversation
    ) -> None:
        """augment() skips rewriting when no client provided."""
        with patch.object(provider, "_retrieve_chunks", return_value=([], [])) as mock_retrieve:
            await provider.augment(conversation, "What about side effects?")

            # Retrieval should use original query (no rewriting)
            mock_retrieve.assert_called_once()
            call_args = mock_retrieve.call_args
            assert call_args[0][0] == "What about side effects?"

    @pytest.mark.asyncio
    async def test_augment_skips_rewrite_for_long_query(
        self, provider: Any, conversation: Conversation
    ) -> None:
        """augment() skips rewriting for long queries."""
        client = MagicMock()
        client.chat = AsyncMock()

        long_query = "x" * 900  # > 800 chars threshold

        with patch.object(provider, "_retrieve_chunks", return_value=([], [])) as mock_retrieve:
            await provider.augment(conversation, long_query, client=client)

            # Rewriter should NOT have been called
            client.chat.assert_not_called()
            # Retrieval should use original query
            mock_retrieve.assert_called_once()
            call_args = mock_retrieve.call_args
            assert call_args[0][0] == long_query

    @pytest.mark.asyncio
    async def test_augment_skips_rewrite_when_disabled(
        self, provider_rewrite_disabled: Any, conversation: Conversation
    ) -> None:
        """augment() skips rewriting when rewrite_enabled=False."""
        client = MagicMock()
        client.chat = AsyncMock()

        with patch.object(
            provider_rewrite_disabled, "_retrieve_chunks", return_value=([], [])
        ) as mock_retrieve:
            await provider_rewrite_disabled.augment(conversation, "What about it?", client=client)

            # Rewriter should NOT have been called
            client.chat.assert_not_called()
            # Retrieval should use original query
            mock_retrieve.assert_called_once()
            call_args = mock_retrieve.call_args
            assert call_args[0][0] == "What about it?"

    @pytest.mark.asyncio
    async def test_augment_falls_back_on_rewrite_error(
        self, provider: Any, conversation: Conversation
    ) -> None:
        """augment() uses original query if rewriting fails."""
        from chatty.client.openai_client import OpenAIClient

        # Use spec=OpenAIClient so isinstance check passes
        client = MagicMock(spec=OpenAIClient)
        client.chat = AsyncMock(side_effect=Exception("LLM error"))

        with patch.object(provider, "_retrieve_chunks", return_value=([], [])) as mock_retrieve:
            await provider.augment(conversation, "What about it?", client=client)

            # Rewriter was called but failed
            client.chat.assert_called_once()
            # Retrieval should use original query (fallback)
            mock_retrieve.assert_called_once()
            call_args = mock_retrieve.call_args
            assert call_args[0][0] == "What about it?"

    def test_init_with_rewrite_enabled(self, tmp_path: Path) -> None:
        """LitkitProvider accepts rewrite_enabled parameter."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            provider = LitkitProvider(workspace=tmp_path, rewrite_enabled=True)
            assert provider._rewriter.enabled is True

            provider_disabled = LitkitProvider(workspace=tmp_path, rewrite_enabled=False)
            assert provider_disabled._rewriter.enabled is False


class TestLitkitProviderReranking:
    """Tests for LitkitProvider LLM reranking functionality."""

    @pytest.fixture
    def provider(self, tmp_path: Path) -> Any:
        """Create a LitkitProvider for testing."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            return LitkitProvider(workspace=tmp_path, rerank=True, rerank_top_n=5)

    @pytest.fixture
    def provider_rerank_disabled(self, tmp_path: Path) -> Any:
        """Create a LitkitProvider with reranking disabled."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            return LitkitProvider(workspace=tmp_path, rerank=False)

    def test_parse_rerank_scores_valid_json(self, provider: Any) -> None:
        """_parse_rerank_scores parses valid JSON array."""
        scores = provider._parse_rerank_scores("[5, 4, 3, 2, 1]", 5)
        assert scores == [5, 4, 3, 2, 1]

    def test_parse_rerank_scores_with_whitespace(self, provider: Any) -> None:
        """_parse_rerank_scores handles whitespace in response."""
        scores = provider._parse_rerank_scores("  [5, 4, 3]  \n", 3)
        assert scores == [5, 4, 3]

    def test_parse_rerank_scores_with_markdown_fence(self, provider: Any) -> None:
        """_parse_rerank_scores strips markdown code fences."""
        response = "```json\n[5, 4, 3]\n```"
        scores = provider._parse_rerank_scores(response, 3)
        assert scores == [5, 4, 3]

    def test_parse_rerank_scores_clamps_values(self, provider: Any) -> None:
        """_parse_rerank_scores clamps scores to 1-5 range."""
        scores = provider._parse_rerank_scores("[10, 0, -5, 3]", 4)
        assert scores == [5, 1, 1, 3]  # 10->5, 0->1, -5->1

    def test_parse_rerank_scores_fills_missing(self, provider: Any) -> None:
        """_parse_rerank_scores fills missing scores with default (3)."""
        scores = provider._parse_rerank_scores("[5, 4]", 5)
        assert scores == [5, 4, 3, 3, 3]

    def test_parse_rerank_scores_handles_invalid_entries(self, provider: Any) -> None:
        """_parse_rerank_scores replaces invalid entries with default."""
        scores = provider._parse_rerank_scores('[5, "bad", null, 2]', 4)
        assert scores == [5, 3, 3, 2]

    def test_parse_rerank_scores_invalid_json_raises(self, provider: Any) -> None:
        """_parse_rerank_scores raises ValueError for invalid JSON."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            provider._parse_rerank_scores("not json", 3)

    def test_parse_rerank_scores_non_array_raises(self, provider: Any) -> None:
        """_parse_rerank_scores raises ValueError for non-array JSON."""
        with pytest.raises(ValueError, match="Expected JSON array"):
            provider._parse_rerank_scores('{"scores": [1,2,3]}', 3)

    @pytest.mark.asyncio
    async def test_rerank_chunks_empty_list(self, provider: Any) -> None:
        """_rerank_chunks returns empty list for empty input."""
        client = MagicMock()
        result = await provider._rerank_chunks("query", [], client, top_n=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_chunks_fewer_than_top_n(self, provider: Any) -> None:
        """_rerank_chunks returns all chunks if fewer than top_n."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(chunk_id=1, text="Chunk 1", paper_title="Paper", score=0.5),
            RetrievedChunk(chunk_id=2, text="Chunk 2", paper_title="Paper", score=0.4),
        ]
        client = MagicMock()
        # Should not call client.chat since chunks < top_n
        result = await provider._rerank_chunks("query", chunks, client, top_n=5)
        assert len(result) == 2
        client.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_rerank_chunks_sorts_by_score(self, provider: Any) -> None:
        """_rerank_chunks sorts chunks by LLM-assigned scores."""
        from chatty.client.openai_client import AssistantMessage, OpenAIClient
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(chunk_id=1, text="Chunk 1", paper_title="P1", score=0.9),
            RetrievedChunk(chunk_id=2, text="Chunk 2", paper_title="P2", score=0.8),
            RetrievedChunk(chunk_id=3, text="Chunk 3", paper_title="P3", score=0.7),
            RetrievedChunk(chunk_id=4, text="Chunk 4", paper_title="P4", score=0.6),
            RetrievedChunk(chunk_id=5, text="Chunk 5", paper_title="P5", score=0.5),
            RetrievedChunk(chunk_id=6, text="Chunk 6", paper_title="P6", score=0.4),
        ]

        # LLM scores: chunk 3 is best (5), chunk 1 is worst (1)
        client = MagicMock(spec=OpenAIClient)
        client.chat = AsyncMock(return_value=AssistantMessage(content="[1, 2, 5, 3, 4, 2]"))

        result = await provider._rerank_chunks("query", chunks, client, top_n=3)

        # Should return top 3 by LLM score: chunk 3 (5), chunk 5 (4), chunk 4 (3)
        assert len(result) == 3
        assert result[0].chunk_id == 3  # Score 5
        assert result[1].chunk_id == 5  # Score 4
        assert result[2].chunk_id == 4  # Score 3

    @pytest.mark.asyncio
    async def test_rerank_chunks_fallback_on_error(self, provider: Any) -> None:
        """_rerank_chunks falls back to original order on LLM error."""
        from chatty.client.openai_client import OpenAIClient
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(chunk_id=1, text="Chunk 1", paper_title="P1", score=0.9),
            RetrievedChunk(chunk_id=2, text="Chunk 2", paper_title="P2", score=0.8),
            RetrievedChunk(chunk_id=3, text="Chunk 3", paper_title="P3", score=0.7),
            RetrievedChunk(chunk_id=4, text="Chunk 4", paper_title="P4", score=0.6),
        ]

        client = MagicMock(spec=OpenAIClient)
        client.chat = AsyncMock(side_effect=Exception("LLM error"))

        result = await provider._rerank_chunks("query", chunks, client, top_n=2)

        # Should return first 2 chunks (fallback to original order)
        assert len(result) == 2
        assert result[0].chunk_id == 1
        assert result[1].chunk_id == 2

    @pytest.mark.asyncio
    async def test_rerank_chunks_fallback_on_parse_error(self, provider: Any) -> None:
        """_rerank_chunks falls back on JSON parse error."""
        from chatty.client.openai_client import AssistantMessage, OpenAIClient
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(chunk_id=1, text="Chunk 1", paper_title="P1", score=0.9),
            RetrievedChunk(chunk_id=2, text="Chunk 2", paper_title="P2", score=0.8),
            RetrievedChunk(chunk_id=3, text="Chunk 3", paper_title="P3", score=0.7),
        ]

        client = MagicMock(spec=OpenAIClient)
        client.chat = AsyncMock(return_value=AssistantMessage(content="invalid json response"))

        result = await provider._rerank_chunks("query", chunks, client, top_n=2)

        # Should return first 2 chunks (fallback)
        assert len(result) == 2
        assert result[0].chunk_id == 1
        assert result[1].chunk_id == 2

    def test_init_with_rerank_enabled(self, tmp_path: Path) -> None:
        """LitkitProvider accepts rerank and rerank_top_n parameters."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            provider = LitkitProvider(workspace=tmp_path, rerank=True, rerank_top_n=15)
            assert provider._rerank is True
            assert provider._rerank_top_n == 15

            provider_disabled = LitkitProvider(workspace=tmp_path, rerank=False)
            assert provider_disabled._rerank is False

    @pytest.mark.asyncio
    async def test_augment_applies_reranking_when_enabled(self, provider: Any) -> None:
        """augment() calls _rerank_chunks when rerank=True and client provided."""
        from chatty.client.openai_client import AssistantMessage, OpenAIClient
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(chunk_id=i, text=f"C{i}", paper_title="P", score=0.9 - i * 0.1)
            for i in range(10)
        ]

        conversation = Conversation(model="gpt-4")
        conversation.add_system_message("System")

        client = MagicMock(spec=OpenAIClient)
        # Return scores for reranking
        client.chat = AsyncMock(return_value=AssistantMessage(content="[5,4,3,2,1,5,4,3,2,1]"))

        with (
            patch.object(provider, "_retrieve_chunks", return_value=(chunks, [100])),
            patch.object(provider, "_rerank_chunks", wraps=provider._rerank_chunks),
        ):
            await provider.augment(conversation, "test query", client=client)
            # _rerank_chunks should have been called
            provider._rerank_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_augment_skips_reranking_without_client(self, provider: Any) -> None:
        """augment() skips reranking when no client provided."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(chunk_id=i, text=f"C{i}", paper_title="P", score=0.9 - i * 0.1)
            for i in range(10)
        ]

        conversation = Conversation(model="gpt-4")
        conversation.add_system_message("System")

        with (
            patch.object(provider, "_retrieve_chunks", return_value=(chunks, [100])),
            patch.object(provider, "_rerank_chunks") as mock_rerank,
        ):
            await provider.augment(conversation, "test query", client=None)
            # _rerank_chunks should NOT have been called
            mock_rerank.assert_not_called()

    @pytest.mark.asyncio
    async def test_augment_skips_reranking_when_disabled(
        self, provider_rerank_disabled: Any
    ) -> None:
        """augment() skips reranking when rerank=False."""
        from chatty.client.openai_client import OpenAIClient
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(chunk_id=i, text=f"C{i}", paper_title="P", score=0.9 - i * 0.1)
            for i in range(10)
        ]

        conversation = Conversation(model="gpt-4")
        conversation.add_system_message("System")

        client = MagicMock(spec=OpenAIClient)

        with (
            patch.object(
                provider_rerank_disabled, "_retrieve_chunks", return_value=(chunks, [100])
            ),
            patch.object(provider_rerank_disabled, "_rerank_chunks") as mock_rerank,
        ):
            await provider_rerank_disabled.augment(conversation, "test query", client=client)
            # _rerank_chunks should NOT have been called
            mock_rerank.assert_not_called()


class TestLitkitProviderMultiQuery:
    """Tests for LitkitProvider multi-query retrieval functionality."""

    @pytest.fixture
    def provider(self, tmp_path: Path) -> Any:
        """Create a LitkitProvider with multi-query enabled."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            return LitkitProvider(workspace=tmp_path, multi_query=True, multi_query_count=3)

    @pytest.fixture
    def provider_multi_disabled(self, tmp_path: Path) -> Any:
        """Create a LitkitProvider with multi-query disabled."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            return LitkitProvider(workspace=tmp_path, multi_query=False)

    def test_init_with_multi_query_enabled(self, tmp_path: Path) -> None:
        """LitkitProvider accepts multi_query and multi_query_count."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()

        with patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"):
            from chatty.rag.litkit_provider import LitkitProvider

            provider = LitkitProvider(workspace=tmp_path, multi_query=True, multi_query_count=5)
            assert provider._multi_query is True
            assert provider._multi_query_count == 5

    def test_deduplicate_by_chunk_id(self, provider: Any) -> None:
        """_deduplicate_by_chunk_id removes duplicates by chunk_id."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(chunk_id=1, text="C1", paper_title="P1", score=0.9),
            RetrievedChunk(chunk_id=2, text="C2", paper_title="P2", score=0.8),
            RetrievedChunk(chunk_id=1, text="C1 dup", paper_title="P1", score=0.7),
            RetrievedChunk(chunk_id=3, text="C3", paper_title="P3", score=0.6),
        ]

        result = provider._deduplicate_by_chunk_id(chunks)

        assert len(result) == 3
        chunk_ids = [c.chunk_id for c in result]
        assert 1 in chunk_ids
        assert 2 in chunk_ids
        assert 3 in chunk_ids

    def test_deduplicate_keeps_highest_score(self, provider: Any) -> None:
        """_deduplicate_by_chunk_id keeps highest score per chunk_id."""
        from chatty.rag.litkit_provider import RetrievedChunk

        chunks = [
            RetrievedChunk(chunk_id=1, text="C1 low", paper_title="P1", score=0.5),
            RetrievedChunk(chunk_id=1, text="C1 high", paper_title="P1", score=0.9),
            RetrievedChunk(chunk_id=1, text="C1 mid", paper_title="P1", score=0.7),
        ]

        result = provider._deduplicate_by_chunk_id(chunks)

        assert len(result) == 1
        assert result[0].score == 0.9

    @pytest.mark.asyncio
    async def test_multi_query_retrieve_generates_variants(self, provider: Any) -> None:
        """_multi_query_retrieve generates variants and retrieves for each."""
        from chatty.client.openai_client import AssistantMessage, OpenAIClient
        from chatty.rag.litkit_provider import RetrievedChunk

        conversation = Conversation(model="gpt-4")
        conversation.add_system_message("System")

        # Mock variant generation
        client = MagicMock(spec=OpenAIClient)
        client.chat = AsyncMock(return_value=AssistantMessage(content='["variant 1", "variant 2"]'))

        chunks = [
            RetrievedChunk(chunk_id=i, text=f"C{i}", paper_title="P", score=0.5) for i in range(3)
        ]

        retrieve_calls: list[str] = []

        def mock_retrieve(query: str, _reuse_ids: Any = None) -> tuple[Any, list[int]]:
            retrieve_calls.append(query)
            return chunks, [100]

        with patch.object(provider, "_retrieve_chunks", side_effect=mock_retrieve):
            result_chunks, result_ids = await provider._multi_query_retrieve(
                "original query", conversation, client
            )

            # Should have called retrieve for original + variants
            assert len(retrieve_calls) >= 2
            assert "original query" in retrieve_calls

    @pytest.mark.asyncio
    async def test_multi_query_retrieve_deduplicates(self, provider: Any) -> None:
        """_multi_query_retrieve deduplicates chunks across variants."""
        from chatty.client.openai_client import AssistantMessage, OpenAIClient
        from chatty.rag.litkit_provider import RetrievedChunk

        conversation = Conversation(model="gpt-4")
        conversation.add_system_message("System")

        client = MagicMock(spec=OpenAIClient)
        client.chat = AsyncMock(return_value=AssistantMessage(content='["variant 1"]'))

        call_count = [0]

        def mock_retrieve(_query: str, _reuse_ids: Any = None) -> tuple[Any, list[int]]:
            call_count[0] += 1
            # Return overlapping chunks
            return [
                RetrievedChunk(chunk_id=1, text="C1", paper_title="P", score=0.9),
                RetrievedChunk(chunk_id=2, text="C2", paper_title="P", score=0.8),
            ], [100]

        with patch.object(provider, "_retrieve_chunks", side_effect=mock_retrieve):
            result_chunks, _ = await provider._multi_query_retrieve("query", conversation, client)

            # Should be deduplicated (only 2 unique chunk_ids)
            assert len(result_chunks) == 2

    @pytest.mark.asyncio
    async def test_multi_query_retrieve_fallback_on_error(self, provider: Any) -> None:
        """_multi_query_retrieve falls back on error."""
        from chatty.client.openai_client import OpenAIClient
        from chatty.rag.litkit_provider import RetrievedChunk

        conversation = Conversation(model="gpt-4")
        conversation.add_system_message("System")

        client = MagicMock(spec=OpenAIClient)
        client.chat = AsyncMock(side_effect=Exception("LLM error"))

        chunks = [RetrievedChunk(chunk_id=1, text="C1", paper_title="P", score=0.9)]

        with patch.object(provider, "_retrieve_chunks", return_value=(chunks, [100])):
            result_chunks, _ = await provider._multi_query_retrieve("query", conversation, client)

            # Should fall back to single-query result
            assert len(result_chunks) == 1

    @pytest.mark.asyncio
    async def test_augment_uses_multi_query_when_enabled(self, provider: Any) -> None:
        """augment() uses _multi_query_retrieve when enabled and client provided."""
        from chatty.client.openai_client import AssistantMessage, OpenAIClient
        from chatty.rag.litkit_provider import RetrievedChunk

        conversation = Conversation(model="gpt-4")
        conversation.add_system_message("System")

        client = MagicMock(spec=OpenAIClient)
        client.chat = AsyncMock(return_value=AssistantMessage(content='["variant"]'))

        chunks = [RetrievedChunk(chunk_id=1, text="C1", paper_title="P", score=0.9)]

        with (
            patch.object(
                provider, "_multi_query_retrieve", return_value=(chunks, [100])
            ) as mock_multi,
            patch.object(provider, "_retrieve_chunks"),
        ):
            await provider.augment(conversation, "test query", client=client)
            mock_multi.assert_called_once()

    @pytest.mark.asyncio
    async def test_augment_skips_multi_query_without_client(self, provider: Any) -> None:
        """augment() skips multi-query when no client provided."""
        from chatty.rag.litkit_provider import RetrievedChunk

        conversation = Conversation(model="gpt-4")
        conversation.add_system_message("System")

        chunks = [RetrievedChunk(chunk_id=1, text="C1", paper_title="P", score=0.9)]

        with (
            patch.object(provider, "_multi_query_retrieve") as mock_multi,
            patch.object(provider, "_retrieve_chunks", return_value=(chunks, [100])),
        ):
            await provider.augment(conversation, "test query", client=None)
            mock_multi.assert_not_called()

    @pytest.mark.asyncio
    async def test_augment_skips_multi_query_when_disabled(
        self, provider_multi_disabled: Any
    ) -> None:
        """augment() skips multi-query when multi_query=False."""
        from chatty.client.openai_client import OpenAIClient
        from chatty.rag.litkit_provider import RetrievedChunk

        conversation = Conversation(model="gpt-4")
        conversation.add_system_message("System")

        client = MagicMock(spec=OpenAIClient)
        chunks = [RetrievedChunk(chunk_id=1, text="C1", paper_title="P", score=0.9)]

        with (
            patch.object(provider_multi_disabled, "_multi_query_retrieve") as mock_multi,
            patch.object(provider_multi_disabled, "_retrieve_chunks", return_value=(chunks, [100])),
        ):
            await provider_multi_disabled.augment(conversation, "test query", client=client)
            mock_multi.assert_not_called()


class TestLitkitErrorClasses:
    """Tests for LitkitProvider error classes."""

    def test_litkit_error_base(self) -> None:
        """LitkitError is the base exception class."""
        from chatty.rag.litkit_provider import LitkitError

        err = LitkitError("test message")
        assert str(err) == "test message"
        assert isinstance(err, Exception)

    def test_litkit_not_installed_error_message(self) -> None:
        """LitkitNotInstalledError has helpful message."""
        from chatty.rag.litkit_provider import LitkitNotInstalledError

        err = LitkitNotInstalledError()
        msg = str(err)
        assert "litkit package not installed" in msg
        assert "uv sync --extra litkit" in msg
        assert "rag_provider = 'none'" in msg

    def test_workspace_not_found_error_message(self) -> None:
        """WorkspaceNotFoundError includes workspace path."""
        from chatty.rag.litkit_provider import WorkspaceNotFoundError

        err = WorkspaceNotFoundError(Path("/test/workspace"))
        msg = str(err)
        assert "/test/workspace" in msg
        assert "rag_workspace" in msg

    def test_faiss_index_not_found_error_message(self) -> None:
        """FAISSIndexNotFoundError includes build instructions."""
        from chatty.rag.litkit_provider import FAISSIndexNotFoundError

        err = FAISSIndexNotFoundError(Path("/test/workspace"))
        msg = str(err)
        assert "/test/workspace" in msg
        assert "litkit --build-only" in msg

    def test_database_not_found_error_message(self) -> None:
        """DatabaseNotFoundError includes rebuild instructions."""
        from chatty.rag.litkit_provider import DatabaseNotFoundError

        err = DatabaseNotFoundError(Path("/test/litkit.db"))
        msg = str(err)
        assert "/test/litkit.db" in msg
        assert "Rebuild" in msg
