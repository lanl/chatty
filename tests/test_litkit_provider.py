"""Tests for LitkitProvider."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chatty.core.conversation import Conversation
from chatty.rag.provider import RAGMetadata


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
    """Tests for LitkitProvider retrieval logic."""

    @pytest.fixture
    def mock_litkit(self) -> MagicMock:
        """Create mock litkit module."""
        mock = MagicMock()
        mock.shortlist_papers = MagicMock(return_value=[1, 2, 3])
        mock.search_chunks_constrained = MagicMock(return_value=([10, 20], [0.9, 0.8]))
        mock.get_chunks = MagicMock(
            return_value=[
                {
                    "id": 10,
                    "text": "Chunk 1 text about HIV treatments.",
                    "paper_title": "Paper A",
                    "pmid": "12345",
                    "pmcid": None,
                },
                {
                    "id": 20,
                    "text": "Chunk 2 text about side effects.",
                    "paper_title": "Paper B",
                    "pmid": None,
                    "pmcid": "PMC67890",
                },
            ]
        )
        mock.connect_db = MagicMock(return_value=MagicMock())
        return mock

    @pytest.fixture
    def valid_workspace(self, tmp_path: Path) -> Path:
        """Create a valid workspace directory with standard litkit layout."""
        (tmp_path / "indices").mkdir()
        (tmp_path / "indices" / "papers.faiss").touch()
        (tmp_path / "sqlite").mkdir()
        (tmp_path / "sqlite" / "litkit.sqlite3").touch()
        return tmp_path

    def test_retrieve_chunks_empty_query(
        self, valid_workspace: Path, mock_litkit: MagicMock
    ) -> None:
        """_retrieve_chunks handles empty paper shortlist."""
        mock_litkit.shortlist_papers = MagicMock(return_value=[])

        with (
            patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"),
            patch.dict(
                "sys.modules",
                {
                    "litkit": MagicMock(),
                    "litkit.cli": mock_litkit,
                    "litkit.db": mock_litkit,
                },
            ),
        ):
            from chatty.rag.litkit_provider import LitkitProvider

            provider = LitkitProvider(workspace=valid_workspace)
            chunks = provider._retrieve_chunks("test query")

            assert chunks == []

    def test_retrieve_chunks_no_results(
        self, valid_workspace: Path, mock_litkit: MagicMock
    ) -> None:
        """_retrieve_chunks handles empty chunk results."""
        mock_litkit.search_chunks_constrained = MagicMock(return_value=([], []))

        with (
            patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"),
            patch.dict(
                "sys.modules",
                {
                    "litkit": MagicMock(),
                    "litkit.cli": mock_litkit,
                    "litkit.db": mock_litkit,
                },
            ),
        ):
            from chatty.rag.litkit_provider import LitkitProvider

            provider = LitkitProvider(workspace=valid_workspace)
            chunks = provider._retrieve_chunks("test query")

            assert chunks == []

    def test_retrieve_chunks_success(self, valid_workspace: Path, mock_litkit: MagicMock) -> None:
        """_retrieve_chunks returns RetrievedChunk objects with metadata."""
        with (
            patch("chatty.rag.litkit_provider.LitkitProvider._check_litkit_installed"),
            patch.dict(
                "sys.modules",
                {
                    "litkit": MagicMock(),
                    "litkit.cli": mock_litkit,
                    "litkit.db": mock_litkit,
                },
            ),
        ):
            from chatty.rag.litkit_provider import LitkitProvider

            provider = LitkitProvider(workspace=valid_workspace)
            chunks = provider._retrieve_chunks("HIV treatments")

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
        result = provider._format_context([])
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
        result = provider._format_context(chunks)

        assert "[1] Paper A (PMID: 12345)" in result
        assert "Test content about HIV." in result
        assert "[2] Paper B (PMCID: PMC67890)" in result
        assert "More content about treatments." in result
        assert "Cite sources by number" in result

    def test_build_metadata(self, provider: Any) -> None:
        """_build_metadata creates RAGMetadata with source info."""
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
        metadata = provider._build_metadata(chunks)

        assert isinstance(metadata, RAGMetadata)
        assert len(metadata.sources) == 1
        assert metadata.sources[0]["title"] == "Paper A"
        assert metadata.sources[0]["pmid"] == "12345"
        assert "snippet" in metadata.sources[0]

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
        metadata = provider._build_metadata(chunks)

        assert len(metadata.sources[0]["snippet"]) <= 203  # 200 + "..."


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
        with patch.object(provider, "_retrieve_chunks", return_value=[]):
            messages, metadata = await provider.augment(conversation, "new question")

        # Should have original messages plus new user message
        assert len(messages) == len(conversation.messages) + 1
        assert messages[-1].role == "user"
        assert messages[-1].content == "new question"
        assert metadata.sources == []

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
            patch.object(provider, "_retrieve_chunks", return_value=chunks),
            patch.object(provider, "_fit_to_budget", return_value=chunks),
        ):
            messages, metadata = await provider.augment(conversation, "new question")

        # User message should contain context
        assert len(messages) == len(conversation.messages) + 1
        assert messages[-1].role == "user"
        assert "Relevant content" in messages[-1].content
        assert "Question: new question" in messages[-1].content
        assert len(metadata.sources) == 1

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

        def mock_fit(chunks: list[Any], available: int) -> list[Any]:
            fit_called_with.append((chunks, available))
            return chunks

        with (
            patch.object(provider, "_retrieve_chunks", return_value=chunks),
            patch.object(provider, "_fit_to_budget", side_effect=mock_fit),
        ):
            await provider.augment(conversation, "question")

        assert len(fit_called_with) == 1
        # Should have calculated available budget based on conversation state
        assert fit_called_with[0][1] > 0


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
        assert "uv add litkit" in msg
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
