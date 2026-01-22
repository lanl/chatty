"""Tests for workers module.

Tests the async worker functions extracted from ChatApp:
- fetch_context_window
- fetch_and_show_models
- send_message
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from chatty.client.openai_client import AssistantMessage, ChattyClientError
from chatty.ui.workers import (
    fetch_and_show_models,
    fetch_context_window,
    send_message,
)


@pytest.fixture
def mock_app() -> MagicMock:
    """Create a mock ChatApp for testing workers."""
    app = MagicMock()
    app.client = MagicMock()
    app.conversation = MagicMock()
    app.conversation.messages = []
    app.conversation.get_token_display.return_value = "100/128000"
    app.conversation.server_reported_tokens = None
    app.config = MagicMock()
    app.config.model = "gpt-4"
    app.rag_provider = MagicMock()
    app.transcript = MagicMock()
    app._pending_user_text = "test message"
    app._current_model = "gpt-4"
    app.streaming = True
    app.current_worker = MagicMock()
    app.last_rag_metadata = None

    # Mock query_one to return mock widgets
    mock_chat_log = MagicMock()
    mock_status_bar = MagicMock()
    mock_status_bar.last_response_time = 1.5

    def query_one_side_effect(selector: str, _widget_type: type | None = None) -> MagicMock:
        if "chat-log" in selector:
            return mock_chat_log
        if "status-bar" in selector:
            return mock_status_bar
        return MagicMock()

    app.query_one = MagicMock(side_effect=query_one_side_effect)

    return app


# ============================================================================
# fetch_context_window Tests
# ============================================================================


class TestFetchContextWindow:
    """Tests for fetch_context_window function."""

    @pytest.mark.asyncio
    async def test_fetch_context_window_no_client(self, mock_app: MagicMock) -> None:
        """Does nothing when client is None."""
        mock_app.client = None
        await fetch_context_window(mock_app)
        # Should not raise error

    @pytest.mark.asyncio
    async def test_fetch_context_window_no_conversation(self, mock_app: MagicMock) -> None:
        """Does nothing when conversation is None."""
        mock_app.conversation = None
        await fetch_context_window(mock_app)
        # Should not raise error

    @pytest.mark.asyncio
    async def test_fetch_context_window_success(self, mock_app: MagicMock) -> None:
        """Successfully fetches and sets context window."""
        mock_app.client.get_model_context_length = AsyncMock(return_value=128000)

        await fetch_context_window(mock_app)

        mock_app.conversation.set_context_window.assert_called_once_with(128000)
        # Status bar should be updated
        status_bar = mock_app.query_one("#status-bar")
        status_bar.update_status.assert_called()

    @pytest.mark.asyncio
    async def test_fetch_context_window_no_context_length(self, mock_app: MagicMock) -> None:
        """Shows error when endpoint returns None for context_length."""
        mock_app.client.get_model_context_length = AsyncMock(return_value=None)

        await fetch_context_window(mock_app)

        # Should show error message
        chat_log = mock_app.query_one("#chat-log")
        chat_log.add_message.assert_called()
        call_args = chat_log.add_message.call_args[0]
        assert call_args[0] == "error"
        assert "context_window" in call_args[1]

    @pytest.mark.asyncio
    async def test_fetch_context_window_client_error(self, mock_app: MagicMock) -> None:
        """Shows error when client raises ChattyClientError."""
        mock_app.client.get_model_context_length = AsyncMock(
            side_effect=ChattyClientError("Connection failed")
        )

        await fetch_context_window(mock_app)

        chat_log = mock_app.query_one("#chat-log")
        chat_log.add_message.assert_called()
        call_args = chat_log.add_message.call_args[0]
        assert call_args[0] == "error"
        assert "failed to fetch" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_fetch_context_window_generic_error(self, mock_app: MagicMock) -> None:
        """Shows error for unexpected exceptions."""
        mock_app.client.get_model_context_length = AsyncMock(
            side_effect=RuntimeError("Unexpected error")
        )

        await fetch_context_window(mock_app)

        chat_log = mock_app.query_one("#chat-log")
        chat_log.add_message.assert_called()
        call_args = chat_log.add_message.call_args[0]
        assert call_args[0] == "error"
        assert "failed to determine" in call_args[1].lower()


# ============================================================================
# fetch_and_show_models Tests
# ============================================================================


class TestFetchAndShowModels:
    """Tests for fetch_and_show_models function."""

    @pytest.mark.asyncio
    async def test_fetch_models_no_client(self, mock_app: MagicMock) -> None:
        """Does nothing when client is None."""
        mock_app.client = None
        await fetch_and_show_models(mock_app)
        # Should not raise error

    @pytest.mark.asyncio
    async def test_fetch_models_success(self, mock_app: MagicMock) -> None:
        """Successfully fetches models and shows picker."""
        mock_app.client.models = AsyncMock(return_value=["gpt-4", "gpt-3.5-turbo"])

        await fetch_and_show_models(mock_app)

        # Should push ModelPickerModal
        mock_app.push_screen.assert_called_once()
        call_args = mock_app.push_screen.call_args[0]
        from chatty.ui.modals import ModelPickerModal

        assert isinstance(call_args[0], ModelPickerModal)

    @pytest.mark.asyncio
    async def test_fetch_models_empty_list(self, mock_app: MagicMock) -> None:
        """Shows manual input modal when models list is empty."""
        mock_app.client.models = AsyncMock(return_value=[])

        await fetch_and_show_models(mock_app)

        # Should push ModelInputModal instead
        mock_app.push_screen.assert_called_once()
        call_args = mock_app.push_screen.call_args[0]
        from chatty.ui.modals import ModelInputModal

        assert isinstance(call_args[0], ModelInputModal)

    @pytest.mark.asyncio
    async def test_fetch_models_client_error(self, mock_app: MagicMock) -> None:
        """Shows manual input modal on ChattyClientError."""
        mock_app.client.models = AsyncMock(side_effect=ChattyClientError("/models not supported"))

        await fetch_and_show_models(mock_app)

        # Should push ModelInputModal
        mock_app.push_screen.assert_called_once()
        call_args = mock_app.push_screen.call_args[0]
        from chatty.ui.modals import ModelInputModal

        assert isinstance(call_args[0], ModelInputModal)

    @pytest.mark.asyncio
    async def test_fetch_models_generic_error(self, mock_app: MagicMock) -> None:
        """Shows error message on unexpected exception."""
        mock_app.client.models = AsyncMock(side_effect=RuntimeError("Network timeout"))

        await fetch_and_show_models(mock_app)

        chat_log = mock_app.query_one("#chat-log")
        chat_log.add_message.assert_called()
        call_args = chat_log.add_message.call_args[0]
        assert call_args[0] == "error"
        assert "failed to fetch" in call_args[1].lower()


# ============================================================================
# send_message Tests
# ============================================================================


class TestSendMessage:
    """Tests for send_message function."""

    @pytest.mark.asyncio
    async def test_send_message_no_client(self, mock_app: MagicMock) -> None:
        """Does nothing when client is None."""
        mock_app.client = None
        await send_message(mock_app)
        # Should not raise error

    @pytest.mark.asyncio
    async def test_send_message_no_conversation(self, mock_app: MagicMock) -> None:
        """Does nothing when conversation is None."""
        mock_app.conversation = None
        await send_message(mock_app)
        # Should not raise error

    @pytest.mark.asyncio
    async def test_send_message_no_pending_text(self, mock_app: MagicMock) -> None:
        """Does nothing when no pending text."""
        mock_app._pending_user_text = ""
        await send_message(mock_app)
        # Client should not be called
        mock_app.client.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_message_streaming_success(self, mock_app: MagicMock) -> None:
        """Successfully sends message in streaming mode."""
        mock_app.streaming = True
        mock_app._pending_user_text = "Hello"

        # Mock RAG provider
        mock_app.rag_provider.augment = AsyncMock(
            return_value=([{"role": "user", "content": "Hello"}], None)
        )

        # Mock streaming response
        async def mock_stream() -> AsyncIterator[str]:
            for token in ["Hello", " there", "!"]:
                yield token

        mock_app.client.chat = AsyncMock(return_value=mock_stream())

        # Mock chat log add_message to return a mock widget
        mock_msg_widget = MagicMock()
        mock_msg_widget.is_streaming = False
        chat_log = mock_app.query_one("#chat-log")
        chat_log.add_message.return_value = mock_msg_widget

        await send_message(mock_app)

        # Verify streaming was done
        chat_log.append_to_last.assert_called()
        mock_msg_widget.finish_streaming.assert_called_once()

        # Verify conversation was updated
        mock_app.conversation.add_user_message.assert_called_with("Hello")
        mock_app.conversation.add_assistant_message.assert_called()

        # Verify transcript was logged
        mock_app.transcript.log_message.assert_called()

    @pytest.mark.asyncio
    async def test_send_message_non_streaming_success(self, mock_app: MagicMock) -> None:
        """Successfully sends message in non-streaming mode."""
        mock_app.streaming = False
        mock_app._pending_user_text = "Hello"

        # Mock RAG provider
        mock_app.rag_provider.augment = AsyncMock(
            return_value=([{"role": "user", "content": "Hello"}], None)
        )

        # Mock non-streaming response
        response = AssistantMessage(
            content="Hi there!",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        mock_app.client.chat = AsyncMock(return_value=response)

        chat_log = mock_app.query_one("#chat-log")

        await send_message(mock_app)

        # Verify message was added
        chat_log.add_message.assert_called()
        # Find the assistant message call
        assistant_calls = [c for c in chat_log.add_message.call_args_list if c[0][0] == "assistant"]
        assert len(assistant_calls) == 1

        # Verify conversation was updated
        mock_app.conversation.add_user_message.assert_called_with("Hello")
        mock_app.conversation.add_assistant_message.assert_called()

    @pytest.mark.asyncio
    async def test_send_message_client_error(self, mock_app: MagicMock) -> None:
        """Shows error on ChattyClientError."""
        mock_app._pending_user_text = "Hello"
        mock_app.rag_provider.augment = AsyncMock(
            return_value=([{"role": "user", "content": "Hello"}], None)
        )
        mock_app.client.chat = AsyncMock(side_effect=ChattyClientError("API error"))

        await send_message(mock_app)

        chat_log = mock_app.query_one("#chat-log")
        error_calls = [c for c in chat_log.add_message.call_args_list if c[0][0] == "error"]
        assert len(error_calls) == 1

        # Status bar should show error
        status_bar = mock_app.query_one("#status-bar")
        status_bar.update_status.assert_called_with(status="Error")

        # Transcript should log error
        mock_app.transcript.log_message.assert_called()

    @pytest.mark.asyncio
    async def test_send_message_generic_error(self, mock_app: MagicMock) -> None:
        """Shows error on unexpected exception."""
        mock_app._pending_user_text = "Hello"
        mock_app.rag_provider.augment = AsyncMock(
            return_value=([{"role": "user", "content": "Hello"}], None)
        )
        mock_app.client.chat = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        await send_message(mock_app)

        chat_log = mock_app.query_one("#chat-log")
        error_calls = [c for c in chat_log.add_message.call_args_list if c[0][0] == "error"]
        assert len(error_calls) == 1
        assert "unexpected" in error_calls[0][0][1].lower()

    @pytest.mark.asyncio
    async def test_send_message_clears_worker(self, mock_app: MagicMock) -> None:
        """Worker is cleared after completion."""
        mock_app._pending_user_text = "Hello"
        mock_app.streaming = False
        mock_app.rag_provider.augment = AsyncMock(
            return_value=([{"role": "user", "content": "Hello"}], None)
        )

        response = AssistantMessage(content="Hi!", usage=None)
        mock_app.client.chat = AsyncMock(return_value=response)

        await send_message(mock_app)

        assert mock_app.current_worker is None

    @pytest.mark.asyncio
    async def test_send_message_clears_worker_on_error(self, mock_app: MagicMock) -> None:
        """Worker is cleared even on error."""
        mock_app._pending_user_text = "Hello"
        mock_app.rag_provider.augment = AsyncMock(
            return_value=([{"role": "user", "content": "Hello"}], None)
        )
        mock_app.client.chat = AsyncMock(side_effect=ChattyClientError("Failed"))

        await send_message(mock_app)

        assert mock_app.current_worker is None

    @pytest.mark.asyncio
    async def test_send_message_updates_status_thinking(self, mock_app: MagicMock) -> None:
        """Status bar shows 'Thinking...' during processing with NullProvider."""
        from chatty.rag.provider import NullProvider

        mock_app._pending_user_text = "Hello"
        mock_app.streaming = False
        # Use NullProvider to get "Thinking..." status (not "Searching corpus...")
        mock_app.rag_provider = NullProvider()

        response = AssistantMessage(content="Hi!", usage=None)
        mock_app.client.chat = AsyncMock(return_value=response)

        await send_message(mock_app)

        status_bar = mock_app.query_one("#status-bar")
        # First call should be "Thinking..." for NullProvider
        first_call = status_bar.update_status.call_args_list[0]
        assert first_call[1].get("status") == "Thinking..."

    @pytest.mark.asyncio
    async def test_send_message_updates_status_searching_corpus(self, mock_app: MagicMock) -> None:
        """Status bar shows 'Searching corpus...' during RAG retrieval."""
        mock_app._pending_user_text = "Hello"
        mock_app.streaming = False
        # Non-NullProvider shows "Searching corpus..." first
        mock_app.rag_provider.augment = AsyncMock(
            return_value=([{"role": "user", "content": "Hello"}], None)
        )

        response = AssistantMessage(content="Hi!", usage=None)
        mock_app.client.chat = AsyncMock(return_value=response)

        await send_message(mock_app)

        status_bar = mock_app.query_one("#status-bar")
        # First call should be "Searching corpus..." for RAG providers
        first_call = status_bar.update_status.call_args_list[0]
        assert first_call[1].get("status") == "Searching corpus..."
        # Second call should be "Thinking..." after RAG completes
        second_call = status_bar.update_status.call_args_list[1]
        assert second_call[1].get("status") == "Thinking..."

    @pytest.mark.asyncio
    async def test_send_message_stores_rag_metadata(self, mock_app: MagicMock) -> None:
        """RAG metadata is stored on app."""
        mock_app._pending_user_text = "Hello"
        mock_app.streaming = False

        rag_metadata = {"sources": ["doc1.txt"]}
        mock_app.rag_provider.augment = AsyncMock(
            return_value=([{"role": "user", "content": "Hello"}], rag_metadata)
        )

        response = AssistantMessage(content="Hi!", usage=None)
        mock_app.client.chat = AsyncMock(return_value=response)

        await send_message(mock_app)

        assert mock_app.last_rag_metadata == rag_metadata


# ============================================================================
# extract_cited_indices Tests
# ============================================================================


class TestExtractCitedIndices:
    """Tests for extract_cited_indices function."""

    def test_empty_text(self) -> None:
        """Returns empty set for empty text."""
        from chatty.ui.workers import extract_cited_indices

        assert extract_cited_indices("") == set()

    def test_no_citations(self) -> None:
        """Returns empty set when no citations present."""
        from chatty.ui.workers import extract_cited_indices

        text = "This is a response with no citations."
        assert extract_cited_indices(text) == set()

    def test_single_citation(self) -> None:
        """Extracts single citation."""
        from chatty.ui.workers import extract_cited_indices

        text = "This claim is supported by evidence [1]."
        assert extract_cited_indices(text) == {1}

    def test_multiple_citations(self) -> None:
        """Extracts multiple separate citations."""
        from chatty.ui.workers import extract_cited_indices

        text = "Claim A [1] and claim B [2] and claim C [3]."
        assert extract_cited_indices(text) == {1, 2, 3}

    def test_comma_separated_citations(self) -> None:
        """Extracts comma-separated citations like [1, 3, 7]."""
        from chatty.ui.workers import extract_cited_indices

        text = "This is supported by multiple sources [1, 3, 7]."
        assert extract_cited_indices(text) == {1, 3, 7}

    def test_no_spaces_in_comma_list(self) -> None:
        """Extracts citations without spaces like [1,3,7]."""
        from chatty.ui.workers import extract_cited_indices

        text = "Multiple sources support this [1,3,7]."
        assert extract_cited_indices(text) == {1, 3, 7}

    def test_duplicate_citations(self) -> None:
        """Deduplicates repeated citations."""
        from chatty.ui.workers import extract_cited_indices

        text = "First claim [1], second claim [1], third claim [2]."
        assert extract_cited_indices(text) == {1, 2}

    def test_mixed_single_and_multi(self) -> None:
        """Handles mix of single and multi-citation brackets."""
        from chatty.ui.workers import extract_cited_indices

        text = "Claim A [1]. Claim B [2, 3]. Claim C [4]."
        assert extract_cited_indices(text) == {1, 2, 3, 4}

    def test_ignores_non_citation_brackets(self) -> None:
        """Ignores brackets that aren't citations."""
        from chatty.ui.workers import extract_cited_indices

        text = "The array [x, y, z] is not a citation but [1] is."
        assert extract_cited_indices(text) == {1}

    def test_double_digit_citations(self) -> None:
        """Handles double-digit citation numbers."""
        from chatty.ui.workers import extract_cited_indices

        text = "Sources [10], [15], and [20, 25] support this."
        assert extract_cited_indices(text) == {10, 15, 20, 25}
