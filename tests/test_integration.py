"""Integration tests for chatty.

Tests the full ChatApp flow including:
- App launch and initial state
- Action handlers (new_session, toggle_stream, etc.)
- Widget interactions
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from chatty.config import Config, ConfigWithSources
from chatty.ui.app import ChatApp
from chatty.ui.widgets import ChatInput, ChatLog, StatusBar


@pytest.fixture
def mock_config_with_sources(tmp_path: Path) -> ConfigWithSources:
    """Create a mock ConfigWithSources for testing."""
    config = Config(
        base_url="http://test.local/v1",
        api_key=SecretStr("test-key"),
        model="test-model",
        session_path=str(tmp_path / "sessions"),
        export_path=str(tmp_path / "exports"),
        copy_fallback_path=str(tmp_path / "copies"),
        transcript_enabled=False,
    )
    return ConfigWithSources(config=config, sources={})


# ============================================================================
# App Launch Tests
# ============================================================================


class TestAppLaunch:
    """Tests for app launch and initialization."""

    async def test_app_launches(self, mock_config_with_sources: ConfigWithSources) -> None:
        """App launches without error."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test():
            # App should be running
            assert app.is_running

    async def test_app_has_chat_log(self, mock_config_with_sources: ConfigWithSources) -> None:
        """App has chat log widget."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test():
            chat_log = app.query_one("#chat-log", ChatLog)
            assert chat_log is not None

    async def test_app_has_input(self, mock_config_with_sources: ConfigWithSources) -> None:
        """App has input widget."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test():
            input_widget = app.query_one("#input", ChatInput)
            assert input_widget is not None

    async def test_app_has_status_bar(self, mock_config_with_sources: ConfigWithSources) -> None:
        """App has status bar widget."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test():
            status_bar = app.query_one("#status-bar", StatusBar)
            assert status_bar is not None

    async def test_initial_model_in_status(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """Status bar shows configured model."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test():
            status_bar = app.query_one("#status-bar", StatusBar)
            assert status_bar._model == "test-model"


# ============================================================================
# Action Tests
# ============================================================================


class TestActions:
    """Tests for app actions."""

    async def test_new_session_clears_chat(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """action_new_session clears chat log."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            chat_log = app.query_one("#chat-log", ChatLog)

            # Add some messages
            chat_log.add_message("user", "Hello")
            chat_log.add_message("assistant", "Hi there")
            assert chat_log.get_last_message() is not None

            # Clear with new session
            app.action_new_session()
            await pilot.pause()

            # Chat log should be empty
            assert chat_log.get_last_message() is None

    async def test_toggle_stream(self, mock_config_with_sources: ConfigWithSources) -> None:
        """action_toggle_stream toggles streaming mode."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test():
            initial = app.streaming

            app.action_toggle_stream()
            assert app.streaming != initial

            app.action_toggle_stream()
            assert app.streaming == initial

    async def test_cancel_when_not_running(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """action_cancel does nothing when no worker running."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            # No worker running
            assert app.current_worker is None

            # Cancel should not crash
            app.action_cancel()
            await pilot.pause()

    async def test_copy_empty_conversation(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """action_copy shows warning when no messages."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            # Conversation is empty
            app.action_copy()
            await pilot.pause()
            # Should not crash - notification shown

    async def test_save_empty_conversation(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """action_save shows message when conversation empty."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            chat_log = app.query_one("#chat-log", ChatLog)

            app.action_save()
            await pilot.pause()

            # Should show a system message about empty conversation
            last = chat_log.get_last_message()
            assert last is not None
            assert "empty" in last.message_content.lower()

    async def test_export_empty_conversation(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """action_export shows warning when conversation empty."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            app.action_export()
            await pilot.pause()
            # Should not crash - notification shown


# ============================================================================
# Conversation Flow Tests
# ============================================================================


class TestConversationFlow:
    """Tests for conversation state management."""

    async def test_conversation_initialized(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """Conversation object is initialized on mount."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test():
            assert app.conversation is not None

    async def test_client_initialized(self, mock_config_with_sources: ConfigWithSources) -> None:
        """OpenAI client is initialized on mount."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test():
            assert app.client is not None

    async def test_system_prompt_added(self, mock_config_with_sources: ConfigWithSources) -> None:
        """System prompt added to conversation if configured."""
        # Add system prompt to config
        mock_config_with_sources.config.system_prompt = "You are helpful"
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test():
            assert app.conversation is not None
            assert len(app.conversation.messages) > 0
            assert app.conversation.messages[0].role == "system"


# ============================================================================
# Input Widget Tests
# ============================================================================


class TestInputWidget:
    """Tests for input widget behavior."""

    async def test_input_focused_on_launch(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """Input widget is focused on app launch."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test():
            input_widget = app.query_one("#input", ChatInput)
            # Input should have focus
            assert input_widget.has_focus

    async def test_empty_submit_ignored(self, mock_config_with_sources: ConfigWithSources) -> None:
        """Empty submit (Ctrl+P with no text) is ignored."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            chat_log = app.query_one("#chat-log", ChatLog)

            # Get message count before (may have config warnings)
            from chatty.ui.widgets import MessageWidget

            initial_count = len(list(chat_log.query(MessageWidget)))

            # Submit empty
            app.action_submit()
            await pilot.pause()

            # No new message added
            final_count = len(list(chat_log.query(MessageWidget)))
            assert final_count == initial_count
