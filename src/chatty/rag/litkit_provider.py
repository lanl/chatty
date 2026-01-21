"""LitkitProvider for RAG using litkit package."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chatty.client.openai_client import Message
from chatty.core.conversation import Conversation
from chatty.rag.provider import RAGMetadata

if TYPE_CHECKING:
    pass


class LitkitError(Exception):
    """Error related to litkit operations."""

    pass


class LitkitNotInstalledError(LitkitError):
    """Raised when litkit package is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "litkit package not installed. "
            "Run `uv add litkit` or set `rag_provider = 'none'` in config."
        )


class WorkspaceNotFoundError(LitkitError):
    """Raised when the workspace directory doesn't exist."""

    def __init__(self, workspace: Path) -> None:
        super().__init__(
            f"litkit workspace not found at {workspace}. "
            f"Set 'rag_workspace' in config or LITKIT_WORKSPACE environment variable."
        )


class FAISSIndexNotFoundError(LitkitError):
    """Raised when FAISS index files are missing."""

    def __init__(self, workspace: Path) -> None:
        super().__init__(
            f"FAISS index not found in {workspace}. "
            f"Run `litkit --build-only --faiss-writer` to build indices."
        )


class DatabaseNotFoundError(LitkitError):
    """Raised when SQLite database is missing."""

    def __init__(self, db_path: Path) -> None:
        super().__init__(
            f"SQLite database not found at {db_path}. "
            f"Index may be corrupted. Rebuild with `litkit --build-only`."
        )


@dataclass
class RetrievedChunk:
    """A chunk of text retrieved from the corpus."""

    chunk_id: int
    text: str
    paper_title: str
    pmid: str | None = None
    pmcid: str | None = None
    score: float = 0.0


class LitkitProvider:
    """RAG provider using litkit for retrieval.

    This provider uses litkit's two-stage retrieval:
    1. Stage 1: Shortlist papers by abstract/title relevance
    2. Stage 2: Search chunks within shortlisted papers

    litkit must be installed separately: `uv add litkit`
    """

    def __init__(
        self,
        workspace: Path,
        top_papers: int = 500,
        top_chunks: int = 30,
    ) -> None:
        """Initialize the LitkitProvider.

        Args:
            workspace: Path to litkit workspace containing indices and database.
            top_papers: Number of papers to shortlist in stage 1 (default: 500).
            top_chunks: Number of chunks to retrieve in stage 2 (default: 30).

        Raises:
            LitkitNotInstalledError: If litkit package is not installed.
            WorkspaceNotFoundError: If workspace directory doesn't exist.
            FAISSIndexNotFoundError: If FAISS index files are missing.
            DatabaseNotFoundError: If SQLite database is missing.
        """
        self._workspace = workspace
        self._top_papers = top_papers
        self._top_chunks = top_chunks

        # Verify litkit is available
        self._check_litkit_installed()

        # Validate workspace exists and has required files
        self._validate_workspace()

    def _check_litkit_installed(self) -> None:
        """Check if litkit package is installed."""
        try:
            import litkit  # noqa: F401
        except ImportError:
            raise LitkitNotInstalledError() from None

    def _validate_workspace(self) -> None:
        """Validate that the workspace has required files.

        Raises:
            WorkspaceNotFoundError: If workspace directory doesn't exist.
            FAISSIndexNotFoundError: If FAISS index files are missing.
            DatabaseNotFoundError: If SQLite database is missing.
        """
        # Check workspace exists
        if not self._workspace.exists():
            raise WorkspaceNotFoundError(self._workspace)

        # Check for FAISS indices (paper index and chunk index)
        # litkit uses 'papers.index' and 'chunks.index' by default
        papers_index = self._workspace / "papers.index"
        chunks_index = self._workspace / "chunks.index"

        if not papers_index.exists() and not chunks_index.exists():
            # Also check for alternative locations
            faiss_dir = self._workspace / "faiss"
            if faiss_dir.exists():
                papers_index = faiss_dir / "papers.index"
                chunks_index = faiss_dir / "chunks.index"

            if not papers_index.exists() and not chunks_index.exists():
                raise FAISSIndexNotFoundError(self._workspace)

        # Check for SQLite database
        db_path = self._workspace / "litkit.db"
        if not db_path.exists():
            # Try alternative location
            db_path = self._workspace / "corpus.db"
            if not db_path.exists():
                raise DatabaseNotFoundError(self._workspace / "litkit.db")

        # Store validated paths
        self._db_path = db_path

    def _retrieve_chunks(self, query: str) -> list[RetrievedChunk]:
        """Retrieve relevant chunks using litkit (sync).

        This is a synchronous method that should be run in a thread pool
        to avoid blocking the async event loop.

        Uses litkit's two-stage retrieval:
        1. Stage 1: Shortlist papers by abstract/title relevance
        2. Stage 2: Search chunks within shortlisted papers

        Args:
            query: The search query.

        Returns:
            List of retrieved chunks with metadata.

        Raises:
            LitkitError: If retrieval fails.
        """
        try:
            # Import litkit functions here to avoid import errors at module load
            from litkit.cli import get_chunks, search_chunks_constrained, shortlist_papers
            from litkit.db import connect_db
        except ImportError:
            raise LitkitNotInstalledError() from None

        try:
            # Connect to the database
            conn = connect_db(str(self._db_path))

            # Stage 1: Shortlist papers by abstract/title relevance
            paper_ids = shortlist_papers(
                query,
                top_k=self._top_papers,
                workspace=str(self._workspace),
            )

            if not paper_ids:
                return []

            # Stage 2: Search chunks within shortlisted papers
            chunk_ids, scores = search_chunks_constrained(
                query,
                paper_ids,
                top_k=self._top_chunks,
                workspace=str(self._workspace),
            )

            if not chunk_ids:
                return []

            # Get chunk content and metadata from database
            chunks_data = get_chunks(conn, chunk_ids)

            # Convert to RetrievedChunk objects
            retrieved_chunks: list[RetrievedChunk] = []
            for i, chunk_data in enumerate(chunks_data):
                # Handle both dict and object return types from litkit
                if isinstance(chunk_data, dict):
                    chunk_id = chunk_data.get("id", chunk_ids[i] if i < len(chunk_ids) else 0)
                    text = chunk_data.get("text", "")
                    paper_title = chunk_data.get("paper_title", chunk_data.get("title", ""))
                    pmid = chunk_data.get("pmid")
                    pmcid = chunk_data.get("pmcid")
                else:
                    # Object with attributes
                    chunk_id = getattr(chunk_data, "id", chunk_ids[i] if i < len(chunk_ids) else 0)
                    text = getattr(chunk_data, "text", "")
                    paper_title = getattr(
                        chunk_data, "paper_title", getattr(chunk_data, "title", "")
                    )
                    pmid = getattr(chunk_data, "pmid", None)
                    pmcid = getattr(chunk_data, "pmcid", None)

                # Get score for this chunk (if available)
                score = scores[i] if i < len(scores) else 0.0

                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk_id=int(chunk_id) if chunk_id else 0,
                        text=str(text),
                        paper_title=str(paper_title),
                        pmid=str(pmid) if pmid else None,
                        pmcid=str(pmcid) if pmcid else None,
                        score=float(score),
                    )
                )

            return retrieved_chunks

        except Exception as e:
            # Wrap litkit errors in LitkitError for consistent error handling
            if isinstance(e, LitkitError):
                raise
            raise LitkitError(f"Retrieval failed: {e}") from e

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a piece of text.

        Uses tiktoken if available, otherwise falls back to character-based estimate.

        Args:
            text: The text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        try:
            import tiktoken

            # Use cl100k_base encoding (GPT-4, GPT-3.5-turbo)
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            # Fallback: ~4 characters per token is a reasonable estimate
            return len(text) // 4

    def _estimate_conversation_tokens(self, conversation: Conversation) -> int:
        """Estimate total tokens used by conversation messages.

        Uses server-reported tokens if available, otherwise estimates with tiktoken.

        Args:
            conversation: The conversation to estimate tokens for.

        Returns:
            Estimated token count for all messages.
        """
        # Use server-reported tokens if available
        if conversation.server_reported_tokens is not None:
            return conversation.server_reported_tokens

        # Otherwise estimate from message contents
        total = 0
        for msg in conversation.messages:
            total += self._estimate_tokens(msg.content)
            # Add overhead for message structure (role, etc.)
            total += 4  # Approximate overhead per message

        return total

    def _fit_to_budget(
        self,
        chunks: list[RetrievedChunk],
        available_tokens: int,
    ) -> list[RetrievedChunk]:
        """Fit chunks to available token budget.

        Chunks are ordered by score (best first). We include as many chunks
        as possible while staying within the token budget.

        Args:
            chunks: List of retrieved chunks (should be sorted by score).
            available_tokens: Maximum tokens available for context.

        Returns:
            Subset of chunks that fit within the token budget.
        """
        if available_tokens <= 0:
            return []

        # Sort chunks by score (highest first) to prioritize best matches
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

        # Estimate overhead for context formatting
        # "The following excerpts..." + source headers + "---\nPlease use..."
        formatting_overhead = 200  # Conservative estimate

        fitted_chunks: list[RetrievedChunk] = []
        total_tokens = formatting_overhead

        for chunk in sorted_chunks:
            # Estimate tokens for this chunk (text + source header)
            chunk_tokens = self._estimate_tokens(chunk.text)
            # Add overhead for source info line
            source_overhead = 50  # "[1] Paper Title (PMID: xxx)\n"
            chunk_total = chunk_tokens + source_overhead

            # Check if adding this chunk would exceed budget
            if total_tokens + chunk_total > available_tokens:
                # If we haven't added any chunks yet, try to truncate this one
                if not fitted_chunks:
                    # Calculate how much we can fit
                    remaining = available_tokens - total_tokens - source_overhead
                    if remaining > 100:  # Only include if we can fit meaningful content
                        # Truncate text to fit
                        truncated_text = self._truncate_to_tokens(chunk.text, remaining)
                        if truncated_text:
                            truncated_chunk = RetrievedChunk(
                                chunk_id=chunk.chunk_id,
                                text=truncated_text,
                                paper_title=chunk.paper_title,
                                pmid=chunk.pmid,
                                pmcid=chunk.pmcid,
                                score=chunk.score,
                            )
                            fitted_chunks.append(truncated_chunk)
                break

            fitted_chunks.append(chunk)
            total_tokens += chunk_total

        return fitted_chunks

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token budget.

        Args:
            text: The text to truncate.
            max_tokens: Maximum tokens allowed.

        Returns:
            Truncated text with ellipsis if truncated.
        """
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)

            if len(tokens) <= max_tokens:
                return text

            # Truncate tokens and decode
            truncated_tokens = tokens[: max_tokens - 1]  # Leave room for "..."
            truncated_text: str = enc.decode(truncated_tokens)
            return truncated_text.rstrip() + "..."

        except ImportError:
            # Fallback: character-based truncation (~4 chars per token)
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text
            return text[: max_chars - 3].rstrip() + "..."

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks as context for the LLM.

        Args:
            chunks: List of retrieved chunks.

        Returns:
            Formatted context string to prepend to user message.
        """
        if not chunks:
            return ""

        context_parts = [
            "The following excerpts from scientific papers may be relevant to your question:\n"
        ]

        for i, chunk in enumerate(chunks, 1):
            source_info = f"[{i}] {chunk.paper_title}"
            if chunk.pmid:
                source_info += f" (PMID: {chunk.pmid})"
            elif chunk.pmcid:
                source_info += f" (PMCID: {chunk.pmcid})"

            context_parts.append(f"{source_info}\n{chunk.text}\n")

        context_parts.append(
            "---\nPlease use the above context to inform your response. "
            "Cite sources by number when appropriate.\n"
        )

        return "\n".join(context_parts)

    def _build_metadata(self, chunks: list[RetrievedChunk]) -> RAGMetadata:
        """Build RAG metadata from retrieved chunks.

        Args:
            chunks: List of retrieved chunks.

        Returns:
            RAGMetadata with source information.
        """
        sources: list[dict[str, Any]] = []
        for chunk in chunks:
            source: dict[str, Any] = {
                "title": chunk.paper_title,
                "snippet": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            }
            if chunk.pmid:
                source["pmid"] = chunk.pmid
            if chunk.pmcid:
                source["pmcid"] = chunk.pmcid
            sources.append(source)

        return RAGMetadata(sources=sources)

    async def augment(
        self,
        conversation: Conversation,
        user_text: str,
    ) -> tuple[list[Message], RAGMetadata]:
        """Augment messages with retrieved context.

        This method:
        1. Retrieves relevant chunks from litkit (in thread pool)
        2. Fits chunks to available token budget
        3. Injects context into the user message
        4. Returns augmented messages and metadata

        Args:
            conversation: Current conversation history.
            user_text: New user input.

        Returns:
            Tuple of (augmented messages, retrieval metadata).
        """
        # Retrieve chunks in a thread pool to avoid blocking
        chunks = await asyncio.to_thread(self._retrieve_chunks, user_text)

        if not chunks:
            # No relevant chunks found, return messages unchanged
            messages = list(conversation.messages) + [Message(role="user", content=user_text)]
            return messages, RAGMetadata(sources=[])

        # Calculate available token budget
        # Reserve 2000 tokens for response, use remaining for context
        current_tokens = self._estimate_conversation_tokens(conversation)
        context_window = conversation.context_window
        response_buffer = 2000
        available_tokens = context_window - current_tokens - response_buffer

        # Fit chunks to budget
        fitted_chunks = self._fit_to_budget(chunks, available_tokens)

        # Format context and build augmented message
        context = self._format_context(fitted_chunks)
        augmented_text = f"{context}\nQuestion: {user_text}" if context else user_text

        # Build messages list
        messages = list(conversation.messages) + [Message(role="user", content=augmented_text)]

        # Build metadata
        metadata = self._build_metadata(fitted_chunks)

        return messages, metadata
