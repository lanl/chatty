"""Query rewriter for multi-turn RAG support.

This module provides LLM-based query rewriting to expand short follow-up
questions into standalone queries for better retrieval.

The rewriter solves the problem where litkit performs single-query retrieval
with no conversation memory. Follow-up questions like "What about the side
effects?" fail to retrieve relevant documents because litkit doesn't know
the context refers to treatments discussed earlier.

Example:
    >>> rewriter = QueryRewriter(enabled=True)
    >>> if rewriter.should_rewrite("What about side effects?", conversation):
    ...     query = await rewriter.rewrite(
    ...         "What about side effects?", conversation, client
    ...     )
    ...     # query is now "What are the side effects of HIV treatments?"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chatty.client.openai_client import OpenAIClient
    from chatty.core.conversation import Conversation


logger = logging.getLogger(__name__)


@dataclass
class RewriteResult:
    """Result of query rewriting.

    Contains the rewritten query and metadata about the rewriting process.
    This structured output enables downstream components to make decisions
    based on query mode and rewriting status.

    Attributes:
        rewritten_query: The expanded query for retrieval.
        original_query: User's original input.
        mode: Query intent classification (NEW_TOPIC, FOLLOWUP, REFERENCE).
        was_rewritten: False if fell back to original query.
        entities: Extracted entities for logging/debugging.
    """

    rewritten_query: str
    original_query: str
    mode: str
    was_rewritten: bool = True
    entities: list[str] = field(default_factory=list)


# Length thresholds (in characters)
SHORT_QUERY_THRESHOLD = 200
LONG_QUERY_THRESHOLD = 800

# Query modes for classification
MODE_NEW_TOPIC = "NEW_TOPIC"  # Fresh topic, no prior context
MODE_FOLLOWUP = "FOLLOWUP"  # Follow-up about current topic
MODE_REFERENCE = "REFERENCE"  # Reference to specific cited source

# Follow-up pattern indicators
FOLLOWUP_PATTERNS = [
    r"\b(it|they|them|this|that|these|those)\b",  # Pronouns
    r"\b(the same|mentioned|above|previous|earlier)\b",  # References
    r"\b(compared to|vs\.?|versus|difference between)\b",  # Comparatives
    r"^(why|how|what about|and)\b",  # Short follow-up starters
]

# Reference patterns - questions about specific papers/sources
REFERENCE_PATTERNS = [
    # "paper 1", "source 2", "citation #3"
    r"\b(paper|source|article|citation|reference)\s*" r"(\d+|#?\d+|one|two|three|four|five)\b",
    # "first paper", "second source"
    r"\b(first|second|third|fourth|fifth)\s*" r"(paper|source|article|citation)\b",
    # "in paper 1"
    r"\bin\s+(paper|source|article)\s*\d+\b",
    # "the paper by Smith"
    r"\bthe\s+(paper|article)\s+(by|from|about)\b",
    # "according to paper 1"
    r"\baccording\s+to\s+(paper|source|citation)\s*\d+\b",
]

# New topic indicators
NEW_TOPIC_PATTERNS = [
    r"^(now|let'?s|moving on|changing topic|different question)\b",
    r"^(tell me about|what is|explain|describe)\s+\w+",
    r"^(unrelated|new topic|separate question)\b",
]

# Rewrite prompt template - improved for better domain term handling
# Used when expand_synonyms is False (legacy behavior)
REWRITE_PROMPT = """You are a query rewriter for a scientific literature search.

Your task: Rewrite the follow-up question as a standalone query
suitable for semantic search in a biomedical/scientific database.

CRITICAL RULES:
1. PRESERVE domain-specific terms, acronyms, and terminology
   - Keep: "FDC", "HIV", "mRNA", "SARS-CoV-2", "IL-6", "CD4+", "TNF-α"
   - Do NOT expand acronyms unless the conversation defines them
2. EXPAND abbreviations ONLY when context provides the full form
   - If conversation says "follicular dendritic cells (FDC)",
     then "FDC" → "follicular dendritic cells (FDC)"
   - If abbreviation meaning is unclear, keep the abbreviation as-is
3. DO NOT invent constraints that weren't mentioned:
   - NO date restrictions unless user specified them
   - NO author names unless user mentioned them
   - NO exclusions unless user asked to exclude something
4. Keep scientific terminology precise - do not paraphrase technical terms
5. Incorporate context from the conversation to make the query standalone

Conversation context:
{conversation_summary}

Follow-up question: "{query}"

Output ONLY the rewritten standalone query, no explanation:"""


# Multi-query variant generation prompt (v0.4.5+)
# Generates N diverse query variants for improved recall
MULTI_QUERY_PROMPT = """\
Generate {count} diverse search query variants for a scientific literature search.

Your task: Create {count} different ways to search for the same information,
each emphasizing different aspects, synonyms, or phrasings.

RULES:
1. Each variant should capture the same intent but use different terms
2. Include domain synonyms and related concepts
3. Vary the query structure (e.g., noun-focused vs verb-focused)
4. Keep each variant under 50 words
5. Output ONLY a JSON array of query strings, no explanation

Example for "HIV treatment side effects":
["HIV antiretroviral therapy adverse events toxicity",
 "AIDS medication side effects complications",
 "human immunodeficiency virus drug treatment safety profile"]

Conversation context:
{conversation_summary}

Original query: "{query}"

Output ONLY a JSON array of {count} query variants:"""


# Rewrite prompt with synonym expansion for improved retrieval recall
# Used when expand_synonyms is True (default in v0.4.5+)
REWRITE_PROMPT_WITH_EXPANSION = """\
You are a query rewriter for a scientific literature search.

Your task: Rewrite the follow-up question as a standalone query
suitable for semantic search in a biomedical/scientific database.
EXPAND the query with synonyms and related terms to improve retrieval.

CRITICAL RULES:
1. EXPAND common acronyms with their full forms:
   - "HIV" -> "HIV human immunodeficiency virus"
   - "AIDS" -> "AIDS acquired immunodeficiency syndrome"
   - "mRNA" -> "mRNA messenger RNA"
   - "FDC" -> "FDC follicular dendritic cells" (if context confirms)
2. ADD domain synonyms for key medical/scientific terms:
   - "treatments" -> "treatments therapies medications drugs"
   - "side effects" -> "side effects adverse events toxicity"
   - "efficacy" -> "efficacy effectiveness therapeutic effect"
   - "patients" -> "patients subjects individuals"
3. KEEP the expanded query under 200 words
4. DO NOT invent constraints that weren't mentioned:
   - NO date restrictions unless user specified them
   - NO author names unless user mentioned them
   - NO exclusions unless user asked to exclude something
5. Incorporate context from the conversation to make the query standalone

Example transformations:
- "HIV treatments" -> "HIV human immunodeficiency virus treatments
  therapies antiretroviral medications ART"
- "What are the side effects?" -> "side effects adverse events
  toxicity of [topic from context]"
- "mRNA vaccine efficacy" -> "mRNA messenger RNA vaccine efficacy
  effectiveness immunogenicity"

Conversation context:
{conversation_summary}

Follow-up question: "{query}"

Output ONLY the expanded standalone query, no explanation:"""


class QueryRewriter:
    """LLM-based query rewriter for multi-turn RAG.

    Rewrites short follow-up questions to standalone queries by
    incorporating conversation context. This improves retrieval
    quality for multi-turn conversations.

    Attributes:
        enabled: Whether rewriting is enabled.
        expand_synonyms: Whether to expand queries with domain synonyms.
    """

    def __init__(
        self,
        enabled: bool = True,
        expand_synonyms: bool = True,
    ) -> None:
        """Initialize the rewriter.

        Args:
            enabled: Whether to enable query rewriting.
            expand_synonyms: Whether to expand queries with domain synonyms
                for improved retrieval recall. Default True (v0.4.5+).
        """
        self.enabled = enabled
        self.expand_synonyms = expand_synonyms
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in FOLLOWUP_PATTERNS]
        self._compiled_reference = [re.compile(p, re.IGNORECASE) for p in REFERENCE_PATTERNS]
        self._compiled_new_topic = [re.compile(p, re.IGNORECASE) for p in NEW_TOPIC_PATTERNS]

    def classify_mode(self, query: str, conversation: Conversation) -> str:
        """Classify the query intent mode.

        Modes:
        - NEW_TOPIC: Fresh question, no prior context needed
        - FOLLOWUP: Follow-up about the current discussion topic
        - REFERENCE: Specific reference to a cited paper/source

        Args:
            query: The user's query text.
            conversation: Current conversation history.

        Returns:
            One of MODE_NEW_TOPIC, MODE_FOLLOWUP, or MODE_REFERENCE.
        """
        # No user/assistant history means it's always a new topic
        # (system prompts don't count as conversation context)
        non_system = [m for m in conversation.messages if m.role != "system"]
        if not non_system:
            return MODE_NEW_TOPIC

        # Check for reference patterns first (highest priority)
        for pattern in self._compiled_reference:
            if pattern.search(query):
                return MODE_REFERENCE

        # Check for explicit new topic indicators
        for pattern in self._compiled_new_topic:
            if pattern.search(query):
                return MODE_NEW_TOPIC

        # Check for follow-up patterns
        if self._has_followup_patterns(query):
            return MODE_FOLLOWUP

        # Default: short queries are follow-ups, long are new topics
        if len(query) < SHORT_QUERY_THRESHOLD:
            return MODE_FOLLOWUP

        return MODE_NEW_TOPIC

    def should_rewrite(self, query: str, conversation: Conversation) -> bool:
        """Determine if a query should be rewritten.

        Decision logic:
        1. If disabled, return False
        2. If no user/assistant history, return False (nothing to reference)
        3. If query > 800 chars, return False (self-contained)
        4. If query < 200 chars, return True (likely follow-up)
        5. Check for follow-up patterns

        Args:
            query: The user's query text.
            conversation: Current conversation history.

        Returns:
            True if query should be rewritten.
        """
        if not self.enabled:
            return False

        # No user/assistant history = nothing to reference
        # (system prompts don't count as conversation context)
        non_system = [m for m in conversation.messages if m.role != "system"]
        if not non_system:
            return False

        query_len = len(query)

        # Long queries are assumed self-contained
        if query_len > LONG_QUERY_THRESHOLD:
            return False

        # Short queries are likely follow-ups
        if query_len < SHORT_QUERY_THRESHOLD:
            return True

        # Medium length: check for follow-up patterns
        return self._has_followup_patterns(query)

    def _has_followup_patterns(self, query: str) -> bool:
        """Check if query contains follow-up indicators.

        Args:
            query: The query to check.

        Returns:
            True if follow-up patterns detected.
        """
        return any(pattern.search(query) for pattern in self._compiled_patterns)

    async def rewrite(
        self,
        query: str,
        conversation: Conversation,
        client: OpenAIClient,
    ) -> str:
        """Rewrite a follow-up query to standalone form.

        Uses the LLM to expand the query with conversation context.
        Falls back to original query on error.

        Args:
            query: The original query to rewrite.
            conversation: Conversation history for context.
            client: OpenAI client for LLM call.

        Returns:
            Rewritten query, or original if rewrite fails.
        """
        result = await self.rewrite_structured(query, conversation, client)
        return result.rewritten_query

    async def rewrite_structured(
        self,
        query: str,
        conversation: Conversation,
        client: OpenAIClient,
    ) -> RewriteResult:
        """Rewrite a follow-up query and return structured result.

        Uses the LLM to expand the query with conversation context.
        Falls back to original query on error.

        Args:
            query: The original query to rewrite.
            conversation: Conversation history for context.
            client: OpenAI client for LLM call.

        Returns:
            RewriteResult with rewritten query and metadata.
        """
        from chatty.client.openai_client import AssistantMessage, Message

        # Classify mode and log for debugging
        mode = self.classify_mode(query, conversation)
        logger.debug("Query mode: %s for query: %.50s...", mode, query)

        try:
            # Build conversation summary (last 2-3 exchanges)
            summary = self._build_conversation_summary(conversation)

            # Select prompt based on expand_synonyms setting
            prompt_template = (
                REWRITE_PROMPT_WITH_EXPANSION if self.expand_synonyms else REWRITE_PROMPT
            )
            prompt = prompt_template.format(
                conversation_summary=summary,
                query=query,
            )

            # Make non-streaming LLM call
            messages = [Message(role="user", content=prompt)]
            response = await client.chat(messages, stream=False)

            # stream=False returns AssistantMessage, not AsyncIterator
            if not isinstance(response, AssistantMessage):
                logger.debug("Unexpected response type, falling back")
                return RewriteResult(
                    rewritten_query=query,
                    original_query=query,
                    mode=mode,
                    was_rewritten=False,
                )

            rewritten = response.content.strip()

            # Validate response (not empty, not too long)
            if rewritten and len(rewritten) < 1000:
                logger.debug(
                    "Rewritten [%s]: '%.50s' -> '%.50s'",
                    mode,
                    query[:50],
                    rewritten[:50],
                )
                return RewriteResult(
                    rewritten_query=rewritten,
                    original_query=query,
                    mode=mode,
                    was_rewritten=True,
                )

            logger.debug("Invalid rewrite response, falling back")
            return RewriteResult(
                rewritten_query=query,
                original_query=query,
                mode=mode,
                was_rewritten=False,
            )

        except Exception:
            # Any error → use original query
            logger.debug(
                "Rewrite failed, falling back to original",
                exc_info=True,
            )
            return RewriteResult(
                rewritten_query=query,
                original_query=query,
                mode=mode,
                was_rewritten=False,
            )

    def _build_conversation_summary(
        self,
        conversation: Conversation,
        max_exchanges: int = 3,
    ) -> str:
        """Build a summary of recent conversation for context.

        Args:
            conversation: Full conversation history.
            max_exchanges: Maximum user/assistant pairs to include.

        Returns:
            Formatted conversation summary.
        """
        messages = conversation.messages
        if not messages:
            return "(no prior conversation)"

        # Take last N exchanges (user + assistant pairs)
        recent = messages[-(max_exchanges * 2) :]

        lines = []
        for msg in recent:
            role = msg.role.capitalize()
            # Truncate long messages
            content = msg.content
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    async def generate_query_variants(
        self,
        query: str,
        conversation: Conversation,
        client: OpenAIClient,
        count: int = 3,
    ) -> list[str]:
        """Generate multiple query variants for multi-query retrieval.

        Creates diverse search queries capturing the same intent with
        different terms, synonyms, and phrasings. This improves recall
        by retrieving documents that match any variant.

        Args:
            query: The original query to generate variants for.
            conversation: Conversation history for context.
            client: OpenAI client for LLM call.
            count: Number of variants to generate (default: 3).

        Returns:
            List of query variants. Falls back to [query] on error.
        """

        from chatty.client.openai_client import AssistantMessage, Message

        try:
            # Build conversation summary for context
            summary = self._build_conversation_summary(conversation)

            prompt = MULTI_QUERY_PROMPT.format(
                count=count,
                conversation_summary=summary,
                query=query,
            )

            # Make non-streaming LLM call
            messages = [Message(role="user", content=prompt)]
            response = await client.chat(messages, stream=False)

            if not isinstance(response, AssistantMessage):
                logger.debug("Unexpected response type for variants")
                return [query]

            # Parse JSON array from response
            variants = self._parse_query_variants(response.content, count)

            if variants:
                logger.debug(
                    "Generated %d variants for: %.50s",
                    len(variants),
                    query[:50],
                )
                return variants

            logger.debug("No valid variants parsed, using original")
            return [query]

        except Exception:
            logger.debug(
                "Variant generation failed, using original",
                exc_info=True,
            )
            return [query]

    def _parse_query_variants(
        self,
        response: str,
        expected_count: int,
    ) -> list[str]:
        """Parse LLM response into list of query variants.

        Expects JSON array format: ["variant 1", "variant 2", ...]

        Args:
            response: LLM response text (should be JSON array).
            expected_count: Expected number of variants.

        Returns:
            List of query strings. Empty list if parsing fails.
        """
        import json

        # Clean response - remove markdown code blocks if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            content_lines = []
            in_fence = False
            for line in lines:
                if line.startswith("```"):
                    in_fence = not in_fence
                    continue
                if in_fence or not line.startswith("```"):
                    content_lines.append(line)
            cleaned = "\n".join(content_lines).strip()

        try:
            variants = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.debug("Failed to parse variants JSON: %.100s", cleaned)
            return []

        if not isinstance(variants, list):
            logger.debug("Variants response is not a list")
            return []

        # Filter to valid non-empty strings
        valid_variants = [str(v).strip() for v in variants if isinstance(v, str) and v.strip()]

        # Limit to expected count
        return valid_variants[:expected_count]
