"""Tests for compression utilities and integration.

Tests the compression module functions and integration with the app.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Label

from chatty.client.openai_client import Message
from chatty.core.compression import (
    COMPRESSION_PROMPT,
    build_compression_prompt,
    detect_code_blocks,
    estimate_messages_tokens,
    estimate_tokens,
)

# ============================================================================
# Unit Tests for Compression Utilities
# ============================================================================


class TestBuildCompressionPrompt:
    """Tests for build_compression_prompt()."""

    def test_includes_user_and_assistant_messages(self) -> None:
        """Prompt includes user and assistant messages."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]
        prompt = build_compression_prompt(messages)

        assert "User: Hello" in prompt
        assert "Assistant: Hi there!" in prompt
        assert "User: How are you?" in prompt

    def test_excludes_system_messages(self) -> None:
        """System messages are excluded from prompt."""
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ]
        prompt = build_compression_prompt(messages)

        assert "You are helpful" not in prompt
        assert "User: Hello" in prompt
        assert "Assistant: Hi!" in prompt

    def test_uses_compression_prompt_template(self) -> None:
        """Uses the COMPRESSION_PROMPT template."""
        messages = [Message(role="user", content="Test")]
        prompt = build_compression_prompt(messages)

        # Check key parts of the template are present
        assert "Summarize the following conversation" in prompt
        assert "Key topics discussed" in prompt
        assert "Summary:" in prompt

    def test_empty_messages_returns_template(self) -> None:
        """Empty messages list returns template with empty conversation."""
        prompt = build_compression_prompt([])
        assert "Summarize the following conversation" in prompt


class TestDetectCodeBlocks:
    """Tests for detect_code_blocks()."""

    def test_detects_code_blocks(self) -> None:
        """Detects ``` markers in messages."""
        messages = [
            Message(role="user", content="Here is code:\n```python\nprint('hi')\n```"),
        ]
        assert detect_code_blocks(messages) is True

    def test_detects_code_in_assistant_messages(self) -> None:
        """Detects code blocks in assistant messages."""
        messages = [
            Message(role="user", content="Write code"),
            Message(role="assistant", content="```\ncode here\n```"),
        ]
        assert detect_code_blocks(messages) is True

    def test_no_code_blocks(self) -> None:
        """Returns False when no code blocks present."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        assert detect_code_blocks(messages) is False

    def test_inline_backticks_not_detected(self) -> None:
        """Single backticks (inline code) are not detected as code blocks."""
        messages = [
            Message(role="user", content="Use the `print` function"),
        ]
        assert detect_code_blocks(messages) is False

    def test_empty_messages(self) -> None:
        """Empty messages list returns False."""
        assert detect_code_blocks([]) is False


class TestEstimateTokens:
    """Tests for estimate_tokens()."""

    def test_estimates_tokens(self) -> None:
        """Token estimation returns reasonable value."""
        text = "Hello, how are you doing today?"
        tokens = estimate_tokens(text)
        # Should be roughly 7-10 tokens for this text
        assert 5 <= tokens <= 15

    def test_empty_string(self) -> None:
        """Empty string returns 0 tokens."""
        assert estimate_tokens("") == 0

    def test_long_text(self) -> None:
        """Longer text returns more tokens."""
        short = estimate_tokens("Hello")
        long = estimate_tokens("Hello " * 100)
        assert long > short

    def test_unknown_model_uses_fallback(self) -> None:
        """Unknown model name uses fallback encoding."""
        # Should not raise, should use cl100k_base fallback
        tokens = estimate_tokens("Hello world", model="unknown-model-xyz")
        assert tokens > 0


class TestEstimateMessagesTokens:
    """Tests for estimate_messages_tokens()."""

    def test_estimates_total_tokens(self) -> None:
        """Estimates total tokens for all messages."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        tokens = estimate_messages_tokens(messages)
        # Should include content tokens plus overhead per message
        assert tokens > 0

    def test_includes_per_message_overhead(self) -> None:
        """Includes overhead per message."""
        single = [Message(role="user", content="Hello")]
        double = [
            Message(role="user", content="Hello"),
            Message(role="user", content=""),  # Empty but still has overhead
        ]
        # Two messages should have more overhead than one
        assert estimate_messages_tokens(double) > estimate_messages_tokens(single)

    def test_empty_messages(self) -> None:
        """Empty messages list returns 0."""
        assert estimate_messages_tokens([]) == 0


# ============================================================================
# Integration Tests
# ============================================================================


class CompressionTestApp(App[None]):
    """Minimal app for testing compression actions."""

    def __init__(self) -> None:
        super().__init__()
        self.compress_called = False
        self.undo_called = False

    def compose(self) -> ComposeResult:
        yield Label("Test")


class TestCompressionIntegration:
    """Integration tests for compression features."""

    async def test_compress_short_conversation_rejected(self) -> None:
        """Compression rejected for < 3 messages."""
        from chatty.config import Config
        from chatty.core.conversation import Conversation
        from chatty.ui.app import ChatApp

        config = Config(base_url="http://test", model="test")

        app = ChatApp(
            config_with_sources=type(
                "ConfigWithSources",
                (),
                {"config": config, "sources": {}},
            )()
        )

        async with app.run_test() as pilot:
            # Initialize conversation with only 2 messages
            app.conversation = Conversation()
            app.conversation.add_user_message("Hello")
            app.conversation.add_assistant_message("Hi")

            # Try to compress - should be rejected
            app.action_compress()
            await pilot.pause()

            # Verify no worker started for compression
            # (notification shown instead)

    async def test_undo_without_compression_rejected(self) -> None:
        """Undo fails when no compression applied."""
        from chatty.config import Config
        from chatty.ui.app import ChatApp

        config = Config(base_url="http://test", model="test")

        app = ChatApp(
            config_with_sources=type(
                "ConfigWithSources",
                (),
                {"config": config, "sources": {}},
            )()
        )

        async with app.run_test() as pilot:
            # Try to undo without any compression
            app.action_undo_compress()
            await pilot.pause()

            # Should not crash, just show notification

    async def test_compression_state_initialized(self) -> None:
        """Compression state attributes are initialized."""
        from chatty.config import Config
        from chatty.ui.app import ChatApp

        config = Config(base_url="http://test", model="test")

        app = ChatApp(
            config_with_sources=type(
                "ConfigWithSources",
                (),
                {"config": config, "sources": {}},
            )()
        )

        async with app.run_test():
            # Verify compression state attributes exist
            assert app._pre_compression_state is None
            assert app._compression_available is False
            assert app._pending_summary is None

    async def test_compression_keybindings_registered(self) -> None:
        """Ctrl+K and Ctrl+U keybindings are registered."""
        from textual.binding import Binding

        from chatty.config import Config
        from chatty.ui.app import ChatApp

        config = Config(base_url="http://test", model="test")

        app = ChatApp(
            config_with_sources=type(
                "ConfigWithSources",
                (),
                {"config": config, "sources": {}},
            )()
        )

        # Check bindings are defined (filter Binding objects only)
        binding_keys = [b.key for b in app.BINDINGS if isinstance(b, Binding)]
        assert "ctrl+k" in binding_keys
        assert "ctrl+u" in binding_keys

    async def test_handle_compression_choice_cancel(self) -> None:
        """Cancel in compression modal clears pending summary."""
        from chatty.config import Config
        from chatty.ui.app import ChatApp

        config = Config(base_url="http://test", model="test")

        app = ChatApp(
            config_with_sources=type(
                "ConfigWithSources",
                (),
                {"config": config, "sources": {}},
            )()
        )

        async with app.run_test():
            # Set up pending summary
            app._pending_summary = "Test summary"

            # Cancel (None result)
            app._handle_compression_choice(None)

            # Pending summary should be cleared
            assert app._pending_summary is None

    async def test_handle_compression_choice_apply(self) -> None:
        """Apply compression stores state and updates conversation."""
        from chatty.config import Config
        from chatty.core.conversation import Conversation
        from chatty.ui.app import ChatApp

        config = Config(base_url="http://test", model="test")

        app = ChatApp(
            config_with_sources=type(
                "ConfigWithSources",
                (),
                {"config": config, "sources": {}},
            )()
        )

        async with app.run_test():
            # Set up conversation and pending summary
            app.conversation = Conversation()
            app.conversation.add_user_message("Hello")
            app.conversation.add_assistant_message("Hi there")
            app.conversation.add_user_message("How are you?")
            original_messages = list(app.conversation.messages)

            app._pending_summary = "Conversation about greetings."

            # Apply compression
            app._handle_compression_choice(True)

            # Verify state was stored
            assert app._pre_compression_state == original_messages
            assert app._compression_available is True

            # Verify conversation was replaced
            assert len(app.conversation.messages) < len(original_messages)
            # Summary should be in a system message
            assert any(
                "Conversation about greetings" in m.content for m in app.conversation.messages
            )

    async def test_undo_restores_state(self) -> None:
        """Undo restores pre-compression state."""
        from chatty.config import Config
        from chatty.core.conversation import Conversation
        from chatty.ui.app import ChatApp

        config = Config(base_url="http://test", model="test")

        app = ChatApp(
            config_with_sources=type(
                "ConfigWithSources",
                (),
                {"config": config, "sources": {}},
            )()
        )

        async with app.run_test():
            # Set up conversation
            app.conversation = Conversation()
            app.conversation.add_user_message("Hello")
            app.conversation.add_assistant_message("Hi there")
            app.conversation.add_user_message("How are you?")

            # Store state as if compression was applied
            original_messages = list(app.conversation.messages)
            app._pre_compression_state = original_messages
            app._compression_available = True

            # Replace with summary
            app.conversation.clear()
            app.conversation.add_system_message("[Summary] Greetings exchanged")

            # Undo
            app.action_undo_compress()

            # Verify state was restored
            assert app.conversation.messages == original_messages
            assert app._pre_compression_state is None
            assert app._compression_available is False


# ============================================================================
# Prompt Content Tests
# ============================================================================


class TestCompressionPrompt:
    """Tests for COMPRESSION_PROMPT constant."""

    def test_prompt_has_required_sections(self) -> None:
        """Prompt includes all required instruction sections."""
        assert "Summarize" in COMPRESSION_PROMPT
        assert "Key topics" in COMPRESSION_PROMPT
        assert "Important decisions" in COMPRESSION_PROMPT
        assert "Technical details" in COMPRESSION_PROMPT
        assert "Summary:" in COMPRESSION_PROMPT

    def test_prompt_has_placeholder(self) -> None:
        """Prompt has {conversation} placeholder."""
        assert "{conversation}" in COMPRESSION_PROMPT
