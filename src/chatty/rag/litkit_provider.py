"""LitkitProvider for RAG using litkit package."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chatty.client.openai_client import Message
from chatty.core.conversation import Conversation
from chatty.rag.provider import RAGMetadata, RAGSource
from chatty.rag.rewriter import QueryRewriter

if TYPE_CHECKING:
    from chatty.client.openai_client import OpenAIClient


# Reranking prompt - scores chunks 1-5 for relevance to query
RERANK_PROMPT = """\
Score each chunk's relevance to the query on a scale of 1-5:
1 = Not relevant at all
2 = Slightly relevant, tangential information
3 = Moderately relevant, some useful information
4 = Highly relevant, directly addresses query
5 = Perfectly relevant, essential information

Query: {query}

Chunks to score:
{chunks}

Output ONLY a JSON array of scores in order, e.g., [4, 2, 5, 3, 1]
No explanation, just the array:"""


# Retrieval script template that runs in a completely isolated subprocess
# This avoids all fd inheritance issues with Textual's terminal I/O
#
# Arguments:
#   1: workspace path
#   2: db_path
#   3: top_papers (for stage-1)
#   4: top_chunks (for stage-2)
#   5: query
#   6: (optional) JSON list of paper_ids to reuse (skips stage-1 if provided)
#
# Output: JSON with chunks and paper_ids used for potential reuse
_RETRIEVAL_SCRIPT = textwrap.dedent("""
import json
import os
import sys

# Set environment before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LITKIT_WORKSPACE"] = sys.argv[1]

try:
    from litkit.cli import deps, get_chunks, search_chunks_constrained, shortlist_papers
    from litkit.db import connect_db

    # Initialize litkit deps
    deps()

    db_path = sys.argv[2]
    top_papers = int(sys.argv[3])
    top_chunks = int(sys.argv[4])
    query = sys.argv[5]

    # Check for optional paper_ids to reuse (skips stage-1)
    reuse_paper_ids = None
    if len(sys.argv) > 6 and sys.argv[6]:
        reuse_paper_ids = json.loads(sys.argv[6])

    conn = connect_db(db_path)

    # Stage 1: Shortlist papers (skip if reusing paper_ids)
    if reuse_paper_ids:
        paper_ids = reuse_paper_ids
    else:
        paper_ids = shortlist_papers(query, top_papers)

    if not paper_ids:
        print(json.dumps({"chunks": [], "paper_ids": []}))
        sys.exit(0)

    # Stage 2: Search chunks within papers
    chunk_ids, scores = search_chunks_constrained(query, paper_ids, top_chunks)

    if not chunk_ids:
        print(json.dumps({"chunks": [], "paper_ids": paper_ids}))
        sys.exit(0)

    chunks_data = get_chunks(conn, chunk_ids)

    chunks = []
    for i, chunk_data in enumerate(chunks_data):
        if isinstance(chunk_data, dict):
            chunk_id = chunk_data.get("id", chunk_ids[i] if i < len(chunk_ids) else 0)
            text = chunk_data.get("text", "")
            paper_title = chunk_data.get("paper_title", chunk_data.get("title", ""))
            pmid = chunk_data.get("pmid")
            pmcid = chunk_data.get("pmcid")
        else:
            chunk_id = getattr(chunk_data, "id", chunk_ids[i] if i < len(chunk_ids) else 0)
            text = getattr(chunk_data, "text", "")
            paper_title = getattr(chunk_data, "paper_title", getattr(chunk_data, "title", ""))
            pmid = getattr(chunk_data, "pmid", None)
            pmcid = getattr(chunk_data, "pmcid", None)

        score = scores[i] if i < len(scores) else 0.0

        chunks.append({
            "chunk_id": int(chunk_id) if chunk_id else 0,
            "text": str(text),
            "paper_title": str(paper_title),
            "pmid": str(pmid) if pmid else None,
            "pmcid": str(pmcid) if pmcid else None,
            "score": float(score),
        })

    # Return chunks AND paper_ids for potential reuse in next turn
    print(json.dumps({"chunks": chunks, "paper_ids": paper_ids}))

except Exception as e:
    # Output error as JSON for parsing
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
""")


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

    Note: Retrieval runs in a completely isolated subprocess (via subprocess.run
    with close_fds=True) to avoid fd inheritance issues with Textual's terminal I/O.

    Citation numbering uses turn-prefixed format [T.N] where T is the turn number
    and N is the chunk number within that turn. This ensures citation references
    remain unambiguous across multi-turn conversations.
    """

    def __init__(
        self,
        workspace: Path,
        top_papers: int = 500,
        top_chunks: int = 30,
        rewrite_enabled: bool = True,
        expand_synonyms: bool = True,
        rerank: bool = False,
        rerank_top_n: int = 10,
        multi_query: bool = False,
        multi_query_count: int = 3,
    ) -> None:
        """Initialize the LitkitProvider.

        Args:
            workspace: Path to litkit workspace containing indices and database.
            top_papers: Number of papers to shortlist in stage 1 (default: 500).
            top_chunks: Number of chunks to retrieve in stage 2 (default: 30).
            rewrite_enabled: Whether to enable LLM-based query rewriting.
            expand_synonyms: Whether to expand queries with domain synonyms
                for improved recall (default: True, v0.4.5+).
            rerank: Whether to use LLM reranking for improved precision
                (default: False, opt-in, v0.4.5+). Adds one LLM call.
            rerank_top_n: Number of top chunks to keep after reranking
                (default: 10, v0.4.5+).
            multi_query: Whether to use multi-query retrieval for improved
                recall (default: False, opt-in, v0.4.5+). Adds one LLM call.
            multi_query_count: Number of query variants to generate
                (default: 3, v0.4.5+).

        Raises:
            LitkitNotInstalledError: If litkit package is not installed.
            WorkspaceNotFoundError: If workspace directory doesn't exist.
            FAISSIndexNotFoundError: If FAISS index files are missing.
            DatabaseNotFoundError: If SQLite database is missing.
        """
        self._workspace = workspace
        self._top_papers = top_papers
        self._top_chunks = top_chunks
        self._rewriter = QueryRewriter(
            enabled=rewrite_enabled,
            expand_synonyms=expand_synonyms,
        )
        self._turn_number = 0  # Track retrieval turn for citation numbering
        self._last_paper_ids: list[int] | None = None  # For shortlist reuse
        self._rerank = rerank
        self._rerank_top_n = rerank_top_n
        self._multi_query = multi_query
        self._multi_query_count = multi_query_count

        # Verify litkit is available
        self._check_litkit_installed()

        # Validate workspace exists and has required files
        self._validate_workspace()

        # Note: We no longer pre-warm litkit in the main process because
        # retrieval runs in a completely isolated subprocess

    def _check_litkit_installed(self) -> None:
        """Check if litkit package is installed."""
        try:
            import litkit  # noqa: F401
        except ImportError:
            raise LitkitNotInstalledError() from None

    def _validate_workspace(self) -> None:
        """Validate that the workspace has required files.

        Standard litkit workspace layout:
            workspace/
            ├── indices/           # FAISS indices
            │   ├── papers.faiss
            │   └── chunks.faiss
            └── sqlite/            # SQLite database
                └── litkit.sqlite3

        Raises:
            WorkspaceNotFoundError: If workspace directory doesn't exist.
            FAISSIndexNotFoundError: If FAISS index files are missing.
            DatabaseNotFoundError: If SQLite database is missing.
        """
        # Check workspace exists
        if not self._workspace.exists():
            raise WorkspaceNotFoundError(self._workspace)

        # Check for FAISS indices in standard litkit location: indices/
        indices_dir = self._workspace / "indices"
        papers_index = indices_dir / "papers.faiss"
        chunks_index = indices_dir / "chunks.faiss"

        if not indices_dir.exists():
            raise FAISSIndexNotFoundError(self._workspace)

        if not papers_index.exists() and not chunks_index.exists():
            raise FAISSIndexNotFoundError(self._workspace)

        # Check for SQLite database in standard litkit location: sqlite/
        sqlite_dir = self._workspace / "sqlite"
        db_path = sqlite_dir / "litkit.sqlite3"

        if not db_path.exists():
            raise DatabaseNotFoundError(db_path)

        # Store validated paths
        self._db_path = db_path
        self._indices_dir = indices_dir

    def reset(self) -> None:
        """Reset provider state for a new session.

        Resets the turn counter and shortlist cache so citation numbers
        start fresh and retrieval doesn't reuse stale paper IDs.
        Called by ChatApp.action_new_session().
        """
        self._turn_number = 0
        self._last_paper_ids = None

    def _retrieve_chunks(
        self,
        query: str,
        reuse_paper_ids: list[int] | None = None,
    ) -> tuple[list[RetrievedChunk], list[int]]:
        """Retrieve relevant chunks using litkit in a subprocess.

        Runs retrieval in a completely isolated subprocess using subprocess.run()
        with close_fds=True. This avoids all fd inheritance issues that occur
        when spawning processes from within Textual's terminal I/O context.

        Args:
            query: The search query.
            reuse_paper_ids: Optional list of paper IDs to reuse (skips stage-1).

        Returns:
            Tuple of (list of retrieved chunks, list of paper IDs used).
            Paper IDs can be cached for shortlist reuse in follow-up queries.

        Raises:
            LitkitError: If retrieval fails.
        """
        try:
            # Build command args
            cmd = [
                sys.executable,
                "-c",
                _RETRIEVAL_SCRIPT,
                str(self._workspace),
                str(self._db_path),
                str(self._top_papers),
                str(self._top_chunks),
                query,
            ]

            # Add optional paper_ids for shortlist reuse
            if reuse_paper_ids:
                cmd.append(json.dumps(reuse_paper_ids))
            else:
                cmd.append("")  # Empty string = run stage-1

            # Run retrieval script in isolated subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                close_fds=True,  # Close all fds - avoids inheritance issues
                timeout=120,  # 2 minute timeout
            )

            if result.returncode != 0:
                # Check for error message in stderr
                if result.stderr:
                    try:
                        error_data = json.loads(result.stderr)
                        if "error" in error_data:
                            raise LitkitError(f"Retrieval failed: {error_data['error']}")
                    except json.JSONDecodeError:
                        pass
                    raise LitkitError(f"Retrieval failed: {result.stderr}")
                raise LitkitError("Retrieval failed with unknown error")

            # Parse JSON output
            if not result.stdout.strip():
                return [], []

            output = json.loads(result.stdout)

            # Handle new output format: {"chunks": [...], "paper_ids": [...]}
            chunk_dicts = output.get("chunks", [])
            paper_ids = output.get("paper_ids", [])

            # Convert to RetrievedChunk objects
            chunks = [
                RetrievedChunk(
                    chunk_id=d["chunk_id"],
                    text=d["text"],
                    paper_title=d["paper_title"],
                    pmid=d["pmid"],
                    pmcid=d["pmcid"],
                    score=d["score"],
                )
                for d in chunk_dicts
            ]

            return chunks, paper_ids

        except subprocess.TimeoutExpired:
            raise LitkitError("Retrieval timed out after 120 seconds") from None
        except json.JSONDecodeError as e:
            raise LitkitError(f"Failed to parse retrieval results: {e}") from e
        except Exception as e:
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

    def _deduplicate_chunks(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Remove duplicate chunks by text content.

        litkit may return the same text passage with different chunk_ids.
        This deduplicates by text content while preserving order (first occurrence wins).

        Args:
            chunks: List of retrieved chunks.

        Returns:
            Deduplicated list preserving original order.
        """
        seen_texts: set[str] = set()
        unique_chunks: list[RetrievedChunk] = []
        for chunk in chunks:
            # Normalize text for comparison (strip whitespace)
            normalized_text = " ".join(chunk.text.split())
            if normalized_text not in seen_texts:
                seen_texts.add(normalized_text)
                unique_chunks.append(chunk)
        return unique_chunks

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

        # Deduplicate chunks first (litkit may return same chunk multiple times)
        chunks = self._deduplicate_chunks(chunks)

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

    def _format_context(self, chunks: list[RetrievedChunk], turn: int) -> str:
        """Format retrieved chunks as context for the LLM.

        Uses turn-prefixed citation format [T.N] where T is turn number
        and N is chunk number within the turn. This ensures citation
        references remain unambiguous across multi-turn conversations.

        Args:
            chunks: List of retrieved chunks.
            turn: Current retrieval turn number.

        Returns:
            Formatted context string to prepend to user message.
        """
        if not chunks:
            return ""

        context_parts = ["The following excerpts from scientific papers may be relevant:\n"]

        for i, chunk in enumerate(chunks, 1):
            # Use [T.N] format for turn-disambiguated citations
            source_info = f"[{turn}.{i}] {chunk.paper_title}"
            if chunk.pmid:
                source_info += f" (PMID: {chunk.pmid})"
            elif chunk.pmcid:
                source_info += f" (PMCID: {chunk.pmcid})"

            context_parts.append(f"{source_info}\n{chunk.text}\n")

        context_parts.append(
            "---\nCite sources using [T.N] format (e.g., [1.3] for turn 1, "
            "source 3). Use the above context to inform your response.\n"
        )

        return "\n".join(context_parts)

    async def _rerank_chunks(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        client: OpenAIClient,
        top_n: int = 10,
    ) -> list[RetrievedChunk]:
        """Rerank chunks using LLM scoring for improved precision.

        Uses a batch scoring approach where all chunks are sent to the LLM
        in a single request. Each chunk is scored 1-5 for relevance:
        1 = Not relevant at all
        5 = Perfectly relevant, essential information

        Args:
            query: The search query to rank against.
            chunks: List of chunks to rerank.
            client: OpenAI client for LLM scoring.
            top_n: Number of top-scoring chunks to return.

        Returns:
            List of top_n chunks sorted by LLM-assigned relevance score.
            Falls back to original chunks (by retrieval score) on error.
        """
        if not chunks:
            return []

        if len(chunks) <= top_n:
            # No need to rerank if we have fewer chunks than top_n
            return chunks

        # Build chunk previews for scoring (truncate to ~500 chars each)
        chunk_previews = []
        for i, chunk in enumerate(chunks):
            preview = chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text
            chunk_previews.append(f"[{i + 1}] {preview}")

        chunks_text = "\n\n".join(chunk_previews)
        prompt = RERANK_PROMPT.format(query=query, chunks=chunks_text)

        try:
            from chatty.client.openai_client import AssistantMessage

            # Single LLM call to score all chunks
            response = await client.chat(
                [Message(role="user", content=prompt)],
                stream=False,
            )

            # Check response type (stream=False returns AssistantMessage)
            if not isinstance(response, AssistantMessage):
                return chunks[:top_n]

            # Parse JSON array of scores from response
            scores = self._parse_rerank_scores(response.content, len(chunks))

            # Pair chunks with scores and sort by score descending
            scored_chunks = list(zip(chunks, scores, strict=False))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)

            # Return top_n chunks, preserving RetrievedChunk objects
            return [chunk for chunk, _score in scored_chunks[:top_n]]

        except Exception:
            # On any error (LLM failure, parse error), fall back to original order
            # Original chunks are already sorted by retrieval score
            return chunks[:top_n]

    def _parse_rerank_scores(self, response: str, expected_count: int) -> list[int]:
        """Parse LLM response into list of integer scores.

        Expects JSON array format: [4, 2, 5, 3, 1]

        Args:
            response: LLM response text (should be JSON array).
            expected_count: Expected number of scores.

        Returns:
            List of integer scores (1-5). Returns default score of 3
            for any missing or invalid entries.

        Raises:
            ValueError: If response is not valid JSON array.
        """
        # Clean response - remove any markdown code blocks
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Remove markdown code fence
            lines = cleaned.split("\n")
            # Find content between fences
            content_lines = []
            in_fence = False
            for line in lines:
                if line.startswith("```"):
                    in_fence = not in_fence
                    continue
                if in_fence or not line.startswith("```"):
                    content_lines.append(line)
            cleaned = "\n".join(content_lines).strip()

        # Parse JSON array
        try:
            scores = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in rerank response: {e}") from e

        if not isinstance(scores, list):
            raise ValueError(f"Expected JSON array, got {type(scores).__name__}")

        # Validate and normalize scores
        result: list[int] = []
        for i in range(expected_count):
            if i < len(scores):
                try:
                    score = int(scores[i])
                    # Clamp to valid range 1-5
                    score = max(1, min(5, score))
                    result.append(score)
                except (ValueError, TypeError):
                    # Invalid score, use default
                    result.append(3)
            else:
                # Missing score, use default
                result.append(3)

        return result

    def _deduplicate_by_chunk_id(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Remove duplicate chunks by chunk_id.

        Used for multi-query retrieval where the same chunk may be
        returned by multiple query variants. Preserves first occurrence
        and keeps highest score for each chunk_id.

        Args:
            chunks: List of retrieved chunks (may contain duplicates).

        Returns:
            Deduplicated list with highest score per chunk_id.
        """
        seen: dict[int, RetrievedChunk] = {}
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                seen[chunk.chunk_id] = chunk
            elif chunk.score > seen[chunk.chunk_id].score:
                # Keep the higher-scoring version
                seen[chunk.chunk_id] = chunk
        return list(seen.values())

    async def _multi_query_retrieve(
        self,
        query: str,
        conversation: Conversation,
        client: OpenAIClient,
        reuse_paper_ids: list[int] | None = None,
    ) -> tuple[list[RetrievedChunk], list[int]]:
        """Retrieve using multiple query variants for improved recall.

        Generates N query variants using LLM, runs retrieval for each,
        and unions the results with deduplication.

        Args:
            query: The primary search query.
            conversation: Conversation history for variant generation.
            client: OpenAI client for generating variants.
            reuse_paper_ids: Optional paper IDs to reuse across all variants.

        Returns:
            Tuple of (deduplicated chunks, paper_ids).
            Falls back to single-query retrieval on error.
        """
        try:
            # Generate query variants
            variants = await self._rewriter.generate_query_variants(
                query, conversation, client, count=self._multi_query_count
            )

            # Collect all chunks from all variants
            all_chunks: list[RetrievedChunk] = []
            all_paper_ids: set[int] = set()

            # Run retrieval for each variant (including original query)
            queries = [query] + [v for v in variants if v != query]

            for q in queries:
                chunks, paper_ids = await asyncio.to_thread(
                    self._retrieve_chunks, q, reuse_paper_ids
                )
                all_chunks.extend(chunks)
                all_paper_ids.update(paper_ids)

            # Deduplicate by chunk_id, keeping highest scores
            unique_chunks = self._deduplicate_by_chunk_id(all_chunks)

            # Sort by score descending
            unique_chunks.sort(key=lambda c: c.score, reverse=True)

            return unique_chunks, list(all_paper_ids)

        except Exception:
            # Fall back to single-query retrieval
            return await asyncio.to_thread(self._retrieve_chunks, query, reuse_paper_ids)

    def _build_metadata(
        self,
        chunks: list[RetrievedChunk],
        total_chunks: int,
        retrieval_time_s: float,
    ) -> RAGMetadata:
        """Build RAG metadata from retrieved chunks.

        Args:
            chunks: List of fitted chunks (after budget fitting).
            total_chunks: Total chunks retrieved before budget fitting.
            retrieval_time_s: Time taken for retrieval in seconds.

        Returns:
            RAGMetadata with source information.
        """
        sources: list[RAGSource] = []
        for chunk in chunks:
            # Create snippet (first 200 chars) for preview
            snippet = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            source = RAGSource(
                title=chunk.paper_title,
                pmid=chunk.pmid,
                pmcid=chunk.pmcid,
                snippet=snippet,
                full_text=chunk.text,  # Store complete chunk text
                score=chunk.score,
            )
            sources.append(source)

        return RAGMetadata(
            sources=sources,
            retrieval_time_s=retrieval_time_s,
            chunk_count=total_chunks,
        )

    async def augment(
        self,
        conversation: Conversation,
        user_text: str,
        client: object | None = None,
    ) -> tuple[list[Message], RAGMetadata]:
        """Augment messages with retrieved context.

        This method:
        1. Optionally rewrites the query for better retrieval (v0.4.2+)
        2. Retrieves relevant chunks from litkit (in isolated subprocess)
        3. Fits chunks to available token budget
        4. Injects context into the user message
        5. Returns augmented messages and metadata

        Retrieval runs via subprocess.run() with close_fds=True to completely
        isolate the subprocess from Textual's terminal I/O, avoiding fd
        inheritance errors that occur with ProcessPoolExecutor/asyncio.to_thread.

        Args:
            conversation: Current conversation history.
            user_text: New user input.
            client: Optional OpenAI client for query rewriting.

        Returns:
            Tuple of (augmented messages, retrieval metadata).
        """
        import time

        # Rewrite query if needed (v0.4.2+)
        retrieval_query = user_text
        rewritten_query: str | None = None
        query_mode: str | None = None
        # Cast client to expected type (protocol allows object for flexibility)
        from chatty.client.openai_client import OpenAIClient

        openai_client = client if isinstance(client, OpenAIClient) else None
        if openai_client and self._rewriter.should_rewrite(user_text, conversation):
            result = await self._rewriter.rewrite_structured(user_text, conversation, openai_client)
            retrieval_query = result.rewritten_query
            query_mode = result.mode
            # Only store rewritten_query if actually different
            if result.was_rewritten and retrieval_query != user_text:
                rewritten_query = retrieval_query
        else:
            # Classify mode even when not rewriting
            query_mode = self._rewriter.classify_mode(user_text, conversation)

        # Determine retrieval strategy based on query mode
        # - NEW_TOPIC: Fresh retrieval (clear cached paper_ids)
        # - FOLLOWUP/REFERENCE: Reuse cached paper_ids if available (skip stage-1)
        reuse_paper_ids: list[int] | None = None
        if query_mode == "NEW_TOPIC":
            # Clear cache for new topic
            self._last_paper_ids = None
        elif self._last_paper_ids and query_mode in ("FOLLOWUP", "REFERENCE"):
            # Reuse previous paper shortlist for follow-up questions
            reuse_paper_ids = self._last_paper_ids

        # Run subprocess retrieval in thread pool to avoid blocking event loop
        # The subprocess itself is completely isolated (close_fds=True)
        start_time = time.monotonic()

        # Optional: Multi-query retrieval for improved recall (v0.4.5+)
        # Generates query variants and unions results
        if self._multi_query and openai_client:
            chunks, paper_ids = await self._multi_query_retrieve(
                retrieval_query,
                conversation,
                openai_client,
                reuse_paper_ids,
            )
        else:
            chunks, paper_ids = await asyncio.to_thread(
                self._retrieve_chunks, retrieval_query, reuse_paper_ids
            )

        retrieval_time_s = time.monotonic() - start_time

        # Cache paper_ids for potential reuse in next turn
        if paper_ids:
            self._last_paper_ids = paper_ids

        if not chunks:
            # No relevant chunks found, return messages unchanged
            messages = list(conversation.messages) + [Message(role="user", content=user_text)]
            return messages, RAGMetadata(
                sources=[],
                retrieval_time_s=retrieval_time_s,
                rewritten_query=rewritten_query,
                query_mode=query_mode,
            )

        total_chunks = len(chunks)

        # Optional: LLM reranking for improved precision (v0.4.5+)
        # Reranking uses an extra LLM call to score chunks by relevance
        if self._rerank and openai_client:
            chunks = await self._rerank_chunks(
                retrieval_query,
                chunks,
                openai_client,
                top_n=self._rerank_top_n,
            )

        # Calculate available token budget
        # Reserve 2000 tokens for response, use remaining for context
        current_tokens = self._estimate_conversation_tokens(conversation)
        context_window = conversation.context_window
        response_buffer = 2000
        available_tokens = context_window - current_tokens - response_buffer

        # Fit chunks to budget
        fitted_chunks = self._fit_to_budget(chunks, available_tokens)

        # Increment turn number for citation disambiguation
        self._turn_number += 1
        turn = self._turn_number

        # Format context and build augmented message
        context = self._format_context(fitted_chunks, turn)
        augmented_text = f"{context}\nQuestion: {user_text}" if context else user_text

        # Build messages list
        messages = list(conversation.messages) + [Message(role="user", content=augmented_text)]

        # Build metadata with rewritten query, mode, and turn number
        metadata = self._build_metadata(fitted_chunks, total_chunks, retrieval_time_s)
        metadata.rewritten_query = rewritten_query
        metadata.query_mode = query_mode
        metadata.turn_number = turn

        return messages, metadata
