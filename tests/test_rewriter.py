"""Tests for query rewriter module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from chatty.rag.rewriter import (
    LONG_QUERY_THRESHOLD,
    MODE_FOLLOWUP,
    MODE_NEW_TOPIC,
    MODE_REFERENCE,
    SHORT_QUERY_THRESHOLD,
    QueryRewriter,
)


class TestShouldRewrite:
    """Tests for should_rewrite() decision logic."""

    def test_disabled_returns_false(self) -> None:
        """Disabled rewriter never rewrites."""
        rewriter = QueryRewriter(enabled=False)
        conv = MagicMock()
        conv.messages = [MagicMock()]
        assert rewriter.should_rewrite("What about side effects?", conv) is False

    def test_no_history_returns_false(self) -> None:
        """No conversation history means nothing to reference."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = []
        assert rewriter.should_rewrite("What about side effects?", conv) is False

    def test_system_prompt_only_returns_false(self) -> None:
        """System prompt only (no user/assistant) means nothing to reference."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        # Conversation has system prompt but no user/assistant messages
        conv.messages = [MagicMock(role="system", content="You are a helpful assistant")]
        assert rewriter.should_rewrite("What about side effects?", conv) is False

    def test_short_query_returns_true(self) -> None:
        """Queries under threshold are rewritten."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = [MagicMock()]

        short_query = "What about it?"  # < 200 chars
        assert len(short_query) < SHORT_QUERY_THRESHOLD
        assert rewriter.should_rewrite(short_query, conv) is True

    def test_long_query_returns_false(self) -> None:
        """Long queries (>800 chars) are assumed self-contained."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = [MagicMock()]

        long_query = "x" * (LONG_QUERY_THRESHOLD + 1)
        assert rewriter.should_rewrite(long_query, conv) is False

    def test_medium_query_with_pronoun_returns_true(self) -> None:
        """Medium queries with pronouns are rewritten."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = [MagicMock()]

        # Build a query with pronoun "it" that's between 200-800 chars
        query = (
            "I would like more information about it and how it affects "
            "patient outcomes in clinical settings. " * 3
        )
        assert SHORT_QUERY_THRESHOLD < len(query) < LONG_QUERY_THRESHOLD
        assert rewriter.should_rewrite(query, conv) is True

    def test_medium_query_without_patterns_returns_false(self) -> None:
        """Medium queries without follow-up patterns are not rewritten."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = [MagicMock()]

        # 250 chars, no pronouns or references
        query = (
            "HIV treatment options for patients with drug resistance have evolved. "
            "Please describe current protocols for managing multi-drug resistant infections."
            + " "
            * 80
        )
        assert SHORT_QUERY_THRESHOLD < len(query) < LONG_QUERY_THRESHOLD
        assert rewriter.should_rewrite(query, conv) is False

    def test_exact_threshold_boundaries(self) -> None:
        """Test behavior at exact threshold boundaries."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = [MagicMock()]

        # Exactly at SHORT_QUERY_THRESHOLD - should NOT rewrite (>= 200 needs patterns)
        query_at_short = "x" * SHORT_QUERY_THRESHOLD
        assert len(query_at_short) == SHORT_QUERY_THRESHOLD
        # This has no patterns, so should return False
        assert rewriter.should_rewrite(query_at_short, conv) is False

        # Just under SHORT_QUERY_THRESHOLD - should rewrite
        query_under_short = "x" * (SHORT_QUERY_THRESHOLD - 1)
        assert len(query_under_short) < SHORT_QUERY_THRESHOLD
        assert rewriter.should_rewrite(query_under_short, conv) is True

        # Exactly at LONG_QUERY_THRESHOLD - should check patterns
        query_at_long = "x" * LONG_QUERY_THRESHOLD
        assert len(query_at_long) == LONG_QUERY_THRESHOLD
        # No patterns, so should return False
        assert rewriter.should_rewrite(query_at_long, conv) is False


class TestPatternDetection:
    """Tests for follow-up pattern detection."""

    @pytest.mark.parametrize(
        "query",
        [
            "What about it?",
            "Tell me more about them",
            "How does this work?",
            "Compare that to other options",
            "What do they recommend?",
            "Explain these findings",
            "Those results are interesting",
        ],
    )
    def test_detects_pronouns(self, query: str) -> None:
        """Detects pronoun patterns."""
        rewriter = QueryRewriter(enabled=True)
        assert rewriter._has_followup_patterns(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "As mentioned above, what are the alternatives?",
            "The previous treatment had issues",
            "Earlier you said something about dosage",
            "The same approach could work here",
            "You mentioned something interesting",
        ],
    )
    def test_detects_references(self, query: str) -> None:
        """Detects reference patterns."""
        rewriter = QueryRewriter(enabled=True)
        assert rewriter._has_followup_patterns(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "Why?",
            "How?",
            "What about side effects?",
            "And the dosage?",
        ],
    )
    def test_detects_short_followups(self, query: str) -> None:
        """Detects short follow-up starters."""
        rewriter = QueryRewriter(enabled=True)
        assert rewriter._has_followup_patterns(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "How does lopinavir compare to ritonavir?",
            "What is the difference between these drugs?",
            "Lopinavir vs ritonavir efficacy",
            "Compare the outcomes versus placebo",
        ],
    )
    def test_detects_comparatives(self, query: str) -> None:
        """Detects comparative patterns."""
        rewriter = QueryRewriter(enabled=True)
        assert rewriter._has_followup_patterns(query) is True

    def test_no_patterns_in_standalone(self) -> None:
        """Standalone queries don't trigger patterns."""
        rewriter = QueryRewriter(enabled=True)
        query = "Describe the mechanism of action for lopinavir in HIV treatment"
        assert rewriter._has_followup_patterns(query) is False

    def test_case_insensitive(self) -> None:
        """Pattern detection is case insensitive."""
        rewriter = QueryRewriter(enabled=True)
        assert rewriter._has_followup_patterns("WHAT ABOUT IT?") is True
        assert rewriter._has_followup_patterns("The PREVIOUS study") is True


class TestRewrite:
    """Tests for rewrite() LLM call."""

    @pytest.mark.asyncio
    async def test_successful_rewrite(self) -> None:
        """Successfully rewrites query via LLM."""
        from chatty.client.openai_client import AssistantMessage

        rewriter = QueryRewriter(enabled=True)

        conv = MagicMock()
        conv.messages = [
            MagicMock(role="user", content="Tell me about HIV treatments"),
            MagicMock(role="assistant", content="HIV treatments include..."),
        ]

        client = MagicMock()
        client.chat = AsyncMock(
            return_value=AssistantMessage(content="What are the side effects of HIV treatments?")
        )

        result = await rewriter.rewrite("What about side effects?", conv, client)

        assert result == "What are the side effects of HIV treatments?"
        client.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_on_error(self) -> None:
        """Falls back to original query on LLM error."""
        rewriter = QueryRewriter(enabled=True)

        conv = MagicMock()
        conv.messages = [MagicMock(role="user", content="Test")]

        client = MagicMock()
        client.chat = AsyncMock(side_effect=Exception("LLM error"))

        result = await rewriter.rewrite("What about it?", conv, client)

        assert result == "What about it?"  # Original query

    @pytest.mark.asyncio
    async def test_fallback_on_empty_response(self) -> None:
        """Falls back if LLM returns empty string."""
        from chatty.client.openai_client import AssistantMessage

        rewriter = QueryRewriter(enabled=True)

        conv = MagicMock()
        conv.messages = [MagicMock(role="user", content="Test")]

        client = MagicMock()
        client.chat = AsyncMock(return_value=AssistantMessage(content=""))

        result = await rewriter.rewrite("What about it?", conv, client)

        assert result == "What about it?"

    @pytest.mark.asyncio
    async def test_fallback_on_too_long_response(self) -> None:
        """Falls back if LLM returns overly long response."""
        from chatty.client.openai_client import AssistantMessage

        rewriter = QueryRewriter(enabled=True)

        conv = MagicMock()
        conv.messages = [MagicMock(role="user", content="Test")]

        client = MagicMock()
        # Response over 1000 chars should be rejected
        client.chat = AsyncMock(return_value=AssistantMessage(content="x" * 1001))

        result = await rewriter.rewrite("What about it?", conv, client)

        assert result == "What about it?"

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_response(self) -> None:
        """Strips whitespace from LLM response."""
        from chatty.client.openai_client import AssistantMessage

        rewriter = QueryRewriter(enabled=True)

        conv = MagicMock()
        conv.messages = [MagicMock(role="user", content="Test")]

        client = MagicMock()
        client.chat = AsyncMock(
            return_value=AssistantMessage(content="  Rewritten query with spaces  \n")
        )

        result = await rewriter.rewrite("What about it?", conv, client)

        assert result == "Rewritten query with spaces"


class TestConversationSummary:
    """Tests for conversation summary building."""

    def test_empty_conversation(self) -> None:
        """Handles empty conversation."""
        rewriter = QueryRewriter(enabled=True)

        conv = MagicMock()
        conv.messages = []

        summary = rewriter._build_conversation_summary(conv)
        assert "no prior conversation" in summary.lower()

    def test_truncates_long_messages(self) -> None:
        """Truncates messages over 500 chars."""
        rewriter = QueryRewriter(enabled=True)

        conv = MagicMock()
        conv.messages = [
            MagicMock(role="user", content="x" * 600),
        ]

        summary = rewriter._build_conversation_summary(conv)
        assert "..." in summary
        assert len(summary) < 600

    def test_limits_exchanges(self) -> None:
        """Limits to max_exchanges pairs."""
        rewriter = QueryRewriter(enabled=True)

        conv = MagicMock()
        conv.messages = [MagicMock(role="user", content=f"User message {i}") for i in range(10)]

        summary = rewriter._build_conversation_summary(conv, max_exchanges=2)
        # Should only have last 4 messages (2 exchanges × 2 messages)
        assert "User message 5" not in summary
        assert "User message 8" in summary or "User message 9" in summary

    def test_formats_role_correctly(self) -> None:
        """Formats role names with capitalization."""
        rewriter = QueryRewriter(enabled=True)

        conv = MagicMock()
        conv.messages = [
            MagicMock(role="user", content="Hello"),
            MagicMock(role="assistant", content="Hi there"),
        ]

        summary = rewriter._build_conversation_summary(conv)
        assert "User: Hello" in summary
        assert "Assistant: Hi there" in summary

    def test_single_message(self) -> None:
        """Handles single message conversation."""
        rewriter = QueryRewriter(enabled=True)

        conv = MagicMock()
        conv.messages = [MagicMock(role="user", content="First question")]

        summary = rewriter._build_conversation_summary(conv)
        assert "User: First question" in summary


class TestQueryRewriterInit:
    """Tests for QueryRewriter initialization."""

    def test_default_enabled(self) -> None:
        """Rewriter is enabled by default."""
        rewriter = QueryRewriter()
        assert rewriter.enabled is True

    def test_can_disable(self) -> None:
        """Rewriter can be disabled."""
        rewriter = QueryRewriter(enabled=False)
        assert rewriter.enabled is False

    def test_patterns_compiled(self) -> None:
        """Patterns are compiled on init."""
        rewriter = QueryRewriter()
        assert len(rewriter._compiled_patterns) > 0
        # Should be compiled regex patterns
        for pattern in rewriter._compiled_patterns:
            assert hasattr(pattern, "search")

    def test_reference_patterns_compiled(self) -> None:
        """Reference patterns are compiled on init."""
        rewriter = QueryRewriter()
        assert len(rewriter._compiled_reference) > 0

    def test_new_topic_patterns_compiled(self) -> None:
        """New topic patterns are compiled on init."""
        rewriter = QueryRewriter()
        assert len(rewriter._compiled_new_topic) > 0


class TestRewriteStructured:
    """Tests for rewrite_structured() returning RewriteResult."""

    @pytest.mark.asyncio
    async def test_returns_rewrite_result(self) -> None:
        """Returns RewriteResult with correct fields."""
        from chatty.client.openai_client import AssistantMessage
        from chatty.rag.rewriter import RewriteResult

        rewriter = QueryRewriter(enabled=True)

        conv = MagicMock()
        conv.messages = [
            MagicMock(role="user", content="Tell me about HIV"),
            MagicMock(role="assistant", content="HIV is..."),
        ]

        client = MagicMock()
        client.chat = AsyncMock(
            return_value=AssistantMessage(content="What are the side effects of HIV treatments?")
        )

        result = await rewriter.rewrite_structured("What about side effects?", conv, client)

        assert isinstance(result, RewriteResult)
        assert result.rewritten_query == "What are the side effects of HIV treatments?"
        assert result.original_query == "What about side effects?"
        assert result.mode == MODE_FOLLOWUP
        assert result.was_rewritten is True

    @pytest.mark.asyncio
    async def test_fallback_sets_was_rewritten_false(self) -> None:
        """Sets was_rewritten=False when fallback to original."""
        from chatty.rag.rewriter import RewriteResult

        rewriter = QueryRewriter(enabled=True)

        conv = MagicMock()
        conv.messages = [MagicMock(role="user", content="Test")]

        client = MagicMock()
        client.chat = AsyncMock(side_effect=Exception("Error"))

        result = await rewriter.rewrite_structured("What about it?", conv, client)

        assert isinstance(result, RewriteResult)
        assert result.rewritten_query == "What about it?"
        assert result.original_query == "What about it?"
        assert result.was_rewritten is False

    @pytest.mark.asyncio
    async def test_mode_is_reference_for_paper_mention(self) -> None:
        """Classifies REFERENCE mode for paper mentions."""
        from chatty.client.openai_client import AssistantMessage
        from chatty.rag.rewriter import RewriteResult

        rewriter = QueryRewriter(enabled=True)

        conv = MagicMock()
        conv.messages = [MagicMock(role="user", content="Test")]

        client = MagicMock()
        client.chat = AsyncMock(return_value=AssistantMessage(content="Details about paper 1"))

        result = await rewriter.rewrite_structured("What does paper 1 say?", conv, client)

        assert isinstance(result, RewriteResult)
        assert result.mode == MODE_REFERENCE


class TestModeClassification:
    """Tests for classify_mode() query intent classification."""

    def test_no_history_returns_new_topic(self) -> None:
        """No conversation history means NEW_TOPIC."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = []
        assert rewriter.classify_mode("Any query", conv) == MODE_NEW_TOPIC

    @pytest.mark.parametrize(
        "query",
        [
            "What does paper 1 say about this?",
            "In paper 2, what are the assumptions?",
            "Can you explain source 3?",
            "According to citation 1, what happened?",
            "The first paper mentions something",
            "What about the second article?",
            "Reference 5 discusses this topic",
            "In the article by Smith, what method was used?",
        ],
    )
    def test_detects_reference_mode(self, query: str) -> None:
        """Detects REFERENCE mode for paper/source citations."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = [MagicMock()]
        assert rewriter.classify_mode(query, conv) == MODE_REFERENCE

    @pytest.mark.parametrize(
        "query",
        [
            "Now let's talk about cancer",
            "Let's discuss a different topic",
            "Moving on, what about diabetes?",
            "Tell me about HIV treatments",
            "What is the mechanism of action?",
            "Explain how mRNA vaccines work",
            "Describe the clinical trial process",
        ],
    )
    def test_detects_new_topic_mode(self, query: str) -> None:
        """Detects NEW_TOPIC mode for fresh questions."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = [MagicMock()]
        assert rewriter.classify_mode(query, conv) == MODE_NEW_TOPIC

    @pytest.mark.parametrize(
        "query",
        [
            "What about side effects?",
            "Tell me more about it",
            "How does that compare?",
            "And the dosage?",
            "Why?",
        ],
    )
    def test_detects_followup_mode(self, query: str) -> None:
        """Detects FOLLOWUP mode for follow-up questions."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = [MagicMock()]
        assert rewriter.classify_mode(query, conv) == MODE_FOLLOWUP

    def test_short_query_defaults_to_followup(self) -> None:
        """Short queries without patterns default to FOLLOWUP."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = [MagicMock()]
        # Short query with no patterns
        assert rewriter.classify_mode("FDC role?", conv) == MODE_FOLLOWUP

    def test_long_query_defaults_to_new_topic(self) -> None:
        """Long queries without patterns default to NEW_TOPIC."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = [MagicMock()]
        # Query over 200 chars with no patterns
        long_q = "Mechanism of action for lopinavir " * 10
        assert len(long_q) > SHORT_QUERY_THRESHOLD
        assert rewriter.classify_mode(long_q, conv) == MODE_NEW_TOPIC

    def test_reference_takes_priority_over_followup(self) -> None:
        """REFERENCE mode takes priority over FOLLOWUP patterns."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = [MagicMock()]
        # Has both "it" (followup) and "paper 1" (reference)
        query = "In paper 1, what does it say about side effects?"
        assert rewriter.classify_mode(query, conv) == MODE_REFERENCE

    def test_case_insensitive_reference(self) -> None:
        """Reference detection is case insensitive."""
        rewriter = QueryRewriter(enabled=True)
        conv = MagicMock()
        conv.messages = [MagicMock()]
        assert rewriter.classify_mode("PAPER 1 says what?", conv) == MODE_REFERENCE
        assert rewriter.classify_mode("The FIRST article", conv) == MODE_REFERENCE
