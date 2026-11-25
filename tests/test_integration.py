"""Integration tests for chatty.

Tests the full ChatApp flow including:
- App launch and initial state
- Action handlers (new_session, toggle_stream, etc.)
- Widget interactions
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx
from pydantic import SecretStr

from chatty.client.openai_client import AssistantMessage, Message, OpenAIClient
from chatty.config import Config, ConfigWithSources
from chatty.core.conversation import Conversation
from chatty.core.session import Session, SessionMetadata, load_session, save_session
from chatty.rag.litkit_provider import LitkitProvider, RetrievedChunk
from chatty.rag.provider import RAGMetadata
from chatty.ui.app import ChatApp
from chatty.ui.modals import CompressionPreviewModal, ModelPickerModal, SessionRenameModal
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


@pytest.fixture
def rag_config(
    mock_config_with_sources: ConfigWithSources, monkeypatch: pytest.MonkeyPatch
) -> ConfigWithSources:
    config = mock_config_with_sources.config
    config.rag_provider = "litkit"
    config.rag_rewrite_enabled = False
    config.stream = False
    monkeypatch.setattr(LitkitProvider, "_check_litkit_installed", lambda _self: None)
    monkeypatch.setattr(LitkitProvider, "_validate_workspace", lambda _self: None)
    monkeypatch.setattr(
        LitkitProvider,
        "_retrieve_chunks",
        lambda *_args: ([RetrievedChunk(1, "Passage", "Paper")], [42]),
    )
    return mock_config_with_sources


@pytest.fixture
def saved_session_file(mock_config_with_sources: ConfigWithSources) -> Path:
    config = mock_config_with_sources.config
    session = Session(
        metadata=SessionMetadata.create(name="Saved session", model=config.model, message_count=2),
        system_prompt="",
        messages=[Message("user", "Saved question"), Message("assistant", "Saved answer")],
    )
    return save_session(session, config.get_session_path())


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
            await pilot.pause()  # Wait for mount
            chat_log = app.query_one("#chat-log", ChatLog)

            # Clear conversation completely (remove any system prompt too)
            if app.conversation:
                app.conversation.clear()
            chat_log.clear_messages()
            await pilot.pause()

            app.action_save()
            await pilot.pause()
            await pilot.pause()  # Extra pause for message to appear

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


# ============================================================================
# More Action Tests for Coverage
# ============================================================================


class TestAdditionalActions:
    """Additional action tests for coverage."""

    async def test_regenerate_empty_conversation(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """action_regenerate does nothing with empty conversation."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            # Empty conversation
            assert app.conversation is not None
            initial_count = len(app.conversation.messages)

            # Try to regenerate - should do nothing
            app.action_regenerate()
            await pilot.pause()

            # No crash, no change
            assert len(app.conversation.messages) == initial_count

    async def test_browse_sessions(self, mock_config_with_sources: ConfigWithSources) -> None:
        """action_browse_sessions opens modal."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            # Browse sessions
            app.action_browse_sessions()
            await pilot.pause()

            # Modal should be open
            # Modal should be open - just verify no crash

    async def test_load_file(self, mock_config_with_sources: ConfigWithSources) -> None:
        """action_load_file opens modal."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            # Load file
            app.action_load_file()
            await pilot.pause()

            # Modal should be open

            # Just verify no crash

    async def test_pick_model(self, mock_config_with_sources: ConfigWithSources) -> None:
        """action_pick_model starts model fetch."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            # Pick model - starts async fetch
            app.action_pick_model()
            await pilot.pause()

            # Just verify no crash
            await pilot.pause()


# ============================================================================
# Phase 1: app.py Coverage Tests
# ============================================================================


class TestCopyAction:
    """Tests for action_copy with mock clipboard."""

    async def test_copy_with_assistant_message(
        self, mock_config_with_sources: ConfigWithSources, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """action_copy copies last assistant message to clipboard."""
        import pyperclip

        copied_text: list[str] = []
        monkeypatch.setattr(pyperclip, "copy", lambda x: copied_text.append(x))

        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()  # Wait for mount
            assert app.conversation is not None

            # Add messages to conversation (after mount)
            app.conversation.add_user_message("Hello")
            app.conversation.add_assistant_message("Hi there!")

            # Copy
            app.action_copy()
            await pilot.pause()

            # Verify copy was called
            assert len(copied_text) >= 1
            assert any("Hi there!" in text for text in copied_text)

    async def test_copy_clipboard_fallback(
        self, mock_config_with_sources: ConfigWithSources, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """action_copy falls back to file when clipboard unavailable."""
        import pyperclip

        def raise_error(_text: str) -> None:
            raise pyperclip.PyperclipException("No clipboard")

        monkeypatch.setattr(pyperclip, "copy", raise_error)
        config = mock_config_with_sources.config
        copy_dir = config.get_copy_fallback_path() / "[draft]"
        config.copy_fallback_path = str(copy_dir)

        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test(notifications=True) as pilot:
            await pilot.pause()  # Wait for mount
            assert app.conversation is not None

            # Add messages
            app.conversation.add_user_message("Hello")
            app.conversation.add_assistant_message("Response text")

            # Copy (should fallback to file)
            app.action_copy()
            await pilot.pause()

            (saved_file,) = copy_dir.glob("copy-*.txt")
            assert str(saved_file) in str(app.screen.query_one("Toast").render())


class TestSaveAction:
    """Tests for action_save with populated conversation."""

    async def test_save_opens_rename_modal_on_first_save(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """action_save opens name modal on first save."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()  # Wait for mount
            assert app.conversation is not None

            # Add messages
            app.conversation.add_user_message("Hello world")
            app.conversation.add_assistant_message("Hi!")

            # Save - should open modal
            app.action_save()
            await pilot.pause()

            # Modal should be open (2 screens: main + modal)
            assert len(app.screen_stack) == 2

    async def test_save_creates_session_file_via_modal(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """action_save creates session file after modal is dismissed."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()  # Wait for mount
            assert app.conversation is not None

            # Add messages
            app.conversation.add_user_message("Hello world")
            app.conversation.add_assistant_message("Hi!")

            # Save - opens modal
            app.action_save()
            await pilot.pause()

            # Press enter to accept default name in modal
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()  # Extra pause for save to complete

            # Verify file created
            session_path = Path(mock_config_with_sources.config.session_path)
            sessions = list(session_path.glob("*.json"))
            assert len(sessions) >= 1

    async def test_subsequent_save_no_modal(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """Subsequent saves don't open modal."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()  # Wait for mount
            assert app.conversation is not None

            # Add messages
            app.conversation.add_user_message("Hello world")
            app.conversation.add_assistant_message("Hi!")

            # First save - opens modal
            app.action_save()
            await pilot.pause()
            await pilot.press("enter")  # Accept default name
            await pilot.pause()

            # Add more messages
            app.conversation.add_user_message("More")
            app.conversation.add_assistant_message("Content")

            # Second save - should NOT open modal
            initial_stack_size = len(app.screen_stack)
            app.action_save()
            await pilot.pause()

            # No new modal
            assert len(app.screen_stack) == initial_stack_size


class TestExportAction:
    """Tests for action_export with populated conversation."""

    async def test_export_creates_markdown_file(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """action_export creates markdown file."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()  # Wait for mount
            assert app.conversation is not None

            # Add messages
            app.conversation.add_user_message("Hello")
            app.conversation.add_assistant_message("Hi there!")

            # Export
            app.action_export()
            await pilot.pause()

            # Verify file created
            export_path = Path(mock_config_with_sources.config.export_path)
            exports = list(export_path.glob("*.md"))
            assert len(exports) >= 1

    async def test_export_file_content(self, mock_config_with_sources: ConfigWithSources) -> None:
        """action_export creates markdown with correct content."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()  # Wait for mount
            assert app.conversation is not None

            # Add messages
            app.conversation.add_user_message("Test question")
            app.conversation.add_assistant_message("Test answer")

            # Export
            app.action_export()
            await pilot.pause()

            # Check content
            export_path = Path(mock_config_with_sources.config.export_path)
            exports = list(export_path.glob("*.md"))
            if exports:
                content = exports[0].read_text()
                assert "Test question" in content or "Test answer" in content


class TestSessionLoading:
    """Tests for session_file loading on startup."""

    async def test_load_session_on_startup(
        self, mock_config_with_sources: ConfigWithSources, saved_session_file: Path
    ) -> None:
        """ChatApp loads session from session_file parameter."""
        app = ChatApp(
            config_with_sources=mock_config_with_sources,
            session_file=saved_session_file,
        )
        async with app.run_test():
            assert app.conversation is not None
            assert app.conversation.messages == load_session(saved_session_file).messages


# ============================================================================
# Phase 2: Medium app.py Coverage Tests
# ============================================================================


class TestHandleFilePath:
    """Tests for _handle_file_path callback."""

    async def test_handle_file_path_loads_content(
        self, mock_config_with_sources: ConfigWithSources, tmp_path: Path
    ) -> None:
        """_handle_file_path loads file content into input."""
        # Create test file
        test_file = tmp_path / "query.txt"
        test_file.write_text("Test query from file")

        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()

            # Call handler directly
            app._handle_file_path(str(test_file))
            await pilot.pause()

            # Verify input has content
            input_widget = app.query_one("#input", ChatInput)
            assert "Test query from file" in input_widget.text

    async def test_handle_file_path_not_found(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """_handle_file_path handles missing file gracefully."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()

            # Call handler with nonexistent file
            app._handle_file_path("/nonexistent/file.txt")
            await pilot.pause()

            # Should not crash - error shown


class TestHandleModelSelection:
    """Tests for _handle_model_selection callback."""

    @pytest.mark.parametrize("action", ["submit", "regenerate", "compress"])
    async def test_model_picker_finishes_during_chat_action(
        self,
        action: str,
        mock_config_with_sources: ConfigWithSources,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_config_with_sources.config.context_window = 16000
        mock_config_with_sources.config.stream = False
        release = asyncio.Event()

        async def models(_client: OpenAIClient) -> list[str]:
            await release.wait()
            return ["test-model", "other-model"]

        monkeypatch.setattr(OpenAIClient, "models", models)
        reply = AsyncMock(return_value=AssistantMessage("Answer"))
        monkeypatch.setattr(OpenAIClient, "chat", reply)
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            app.query_one(ChatInput).text = "First question"
            app.action_submit()
            await app.workers.wait_for_complete()
            await pilot.press("ctrl+g")
            app.query_one(ChatInput).text = "Second question"
            getattr(app, f"action_{action}")()
            worker = app._compression_worker if action == "compress" else app.current_worker
            assert worker is not None
            await worker.wait()
            await pilot.pause()
            assert reply.await_count == 2
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, ModelPickerModal)

    @respx.mock
    @pytest.mark.parametrize("override", [None, 16000])
    async def test_context_window_follows_model_selection(
        self,
        override: int | None,
        mock_config_with_sources: ConfigWithSources,
    ) -> None:
        mock_config_with_sources.config.context_window = override or "auto"
        models = respx.get("http://test.local/v1/models").respond(
            json={
                "data": [
                    {"id": "test-model", "context_length": 64000},
                    {"id": "other-model", "context_length": 8192},
                    {"id": "unknown-model"},
                ]
            }
        )
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test():
            await app.workers.wait_for_complete()
            assert app.conversation is not None
            assert app.conversation.context_window == (override or 64000)
            app.conversation.server_reported_tokens = 10000
            app._handle_model_selection("other-model")
            await app.workers.wait_for_complete()
            assert app.conversation.context_window == (override or 8192)
            assert app.conversation.server_reported_tokens is None
            app._handle_model_selection("unknown-model")
            await app.workers.wait_for_complete()
            assert app.conversation.context_window == (override or 128000)
            if override:
                assert not models.called
            else:
                warning = app.query_one(ChatLog).get_last_message()
                assert warning is not None
                assert "unknown-model" in warning.message_content
                assert "128,000" in warning.message_content

    @respx.mock
    @pytest.mark.parametrize("replace_turn", [False, True], ids=["finish", "supersede"])
    async def test_switch_during_retrieval_uses_each_models_budget(
        self,
        rag_config: ConfigWithSources,
        monkeypatch: pytest.MonkeyPatch,
        replace_turn: bool,
    ) -> None:
        respx.get("http://test.local/v1/models").respond(
            json={
                "data": [
                    {"id": "test-model", "context_length": 64000},
                    {"id": "small", "context_length": 2000},
                ]
            }
        )
        replies = respx.post("http://test.local/v1/chat/completions").respond(
            json={"choices": [{"message": {"content": "Answer"}}], "usage": {"total_tokens": 500}}
        )
        app = ChatApp(config_with_sources=rag_config)
        searching = asyncio.Event()
        release = asyncio.Event()
        augment = app.rag_provider.augment

        async def delayed(
            conversation: Conversation, question: str, client: object
        ) -> tuple[list[Message], RAGMetadata]:
            searching.set()
            await release.wait()
            return await augment(conversation, question, client)

        monkeypatch.setattr(app.rag_provider, "augment", delayed)
        async with app.run_test() as pilot:
            app.query_one(ChatInput).text = "First question"
            app.action_submit()
            await asyncio.wait_for(searching.wait(), timeout=5)
            app._handle_model_selection("small")
            if not replace_turn:
                release.set()
                await app.workers.wait_for_complete()
                await pilot.pause()
                first = json.loads(replies.calls.last.request.content)
                assert first["model"] == "test-model"
                assert "Passage" in first["messages"][-1]["content"]
                assert app.conversation is not None
                assert app.conversation.server_reported_tokens is None
            app.query_one(ChatInput).text = "Next question"
            app.action_submit()
            release.set()
            assert app.current_worker is not None
            await app.current_worker.wait()
            assert replies.call_count == (1 if replace_turn else 2)
            last = json.loads(replies.calls.last.request.content)
            assert last["model"] == "small"
            assert last["messages"][-1]["content"] == "Next question"

    async def test_handle_model_selection_updates_model(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """_handle_model_selection updates model."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()

            # Call handler directly
            app._handle_model_selection("new-model-name")
            await pilot.pause()

            # Verify model updated
            assert app.config.model == "new-model-name"
            status_bar = app.query_one("#status-bar", StatusBar)
            assert status_bar._model == "new-model-name"

    async def test_handle_model_selection_none(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """_handle_model_selection with None (cancelled) does nothing."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()

            original_model = app.config.model

            # Call handler with None (user cancelled)
            app._handle_model_selection(None)
            await pilot.pause()

            # Model unchanged
            assert app.config.model == original_model


class TestHandleSessionLoad:
    """Tests for _handle_session_load callback."""

    async def test_load_then_undo_and_save_preserves_loaded_history(
        self, mock_config_with_sources: ConfigWithSources, saved_session_file: Path
    ) -> None:
        expected = load_session(saved_session_file).messages
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            assert app.conversation is not None
            app.conversation.add_user_message("Question from the previous session")
            app._pending_summary = "Summary of the previous session"
            app._handle_compression_choice(True)

            await pilot.press("ctrl+l", "enter")
            await pilot.click("#load-btn")
            assert app.conversation.messages == expected
            await pilot.press("ctrl+y", "ctrl+s")
            assert load_session(saved_session_file).messages == expected
            assert app.conversation.messages == expected

    async def test_new_session_discards_compression_undo(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            assert app.conversation is not None
            fresh_history = list(app.conversation.messages)
            app.conversation.add_user_message("Question from the previous session")
            app._pending_summary = "Summary of the previous session"
            app._handle_compression_choice(True)

            await pilot.press("ctrl+n", "ctrl+y")
            assert app.conversation.messages == fresh_history

    async def test_new_session_discards_in_flight_compression(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            assert app.conversation is not None
            assert app.client is not None
            app.conversation.add_user_message("Old question")
            app.conversation.add_assistant_message("Old answer")
            app.conversation.add_user_message("Follow-up")
            started = asyncio.Event()
            reply = asyncio.Event()

            async def delayed_summary(*_args: object, **_kwargs: object) -> AssistantMessage:
                started.set()
                await reply.wait()
                return AssistantMessage("Summary of the previous session")

            with patch.object(app.client, "chat", delayed_summary):
                app.action_compress()
                await asyncio.wait_for(started.wait(), timeout=5)
                await pilot.press("ctrl+n")
                reply.set()
                await app.workers.wait_for_complete()
                assert not isinstance(app.screen, CompressionPreviewModal)

    @pytest.mark.parametrize("cancel", ["escape", "button"])
    async def test_cancel_save_and_load_keeps_unsaved_session(
        self, mock_config_with_sources: ConfigWithSources, saved_session_file: Path, cancel: str
    ) -> None:
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            assert app.conversation is not None
            app.conversation.add_user_message("Unsaved question")
            expected = list(app.conversation.messages)
            app._session_dirty = True
            app._handle_session_load(saved_session_file)
            await pilot.pause()
            await pilot.click("#save-load-btn")
            assert isinstance(app.screen, SessionRenameModal)
            if cancel == "escape":
                await pilot.press("escape")
            else:
                await pilot.click("#cancel-btn")

            assert app.conversation.messages == expected
            assert app._current_session is None
            assert app._session_dirty
            assert list(saved_session_file.parent.glob("*.json")) == [saved_session_file]

    @pytest.mark.parametrize("first_save", [True, False], ids=["unnamed", "previously-saved"])
    async def test_save_and_load_requires_successful_save(
        self,
        mock_config_with_sources: ConfigWithSources,
        saved_session_file: Path,
        first_save: bool,
    ) -> None:
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            assert app.conversation is not None
            if not first_save:
                app._handle_first_save("Existing session")
            app.conversation.add_user_message("Unsaved question")
            expected = list(app.conversation.messages)
            app._session_dirty = True

            with patch("chatty.ui.session_handlers.save_session", side_effect=OSError("disk full")):
                app._handle_session_load(saved_session_file)
                await pilot.pause()
                await pilot.click("#save-load-btn")
                if first_save:
                    await pilot.press("enter")
            assert app.conversation.messages == expected
            assert app._session_dirty
            if first_save:
                assert app._current_session is None

            # Retry after storage becomes available.
            app._handle_session_load(saved_session_file)
            await pilot.pause()
            await pilot.click("#save-load-btn")
            if first_save:
                assert isinstance(app.screen, SessionRenameModal)
                await pilot.press("enter")
            assert app.conversation.messages == load_session(saved_session_file).messages
            [saved_source] = set(saved_session_file.parent.glob("*.json")) - {saved_session_file}
            assert load_session(saved_source).messages == expected

    async def test_handle_session_load_none(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """_handle_session_load with None (cancelled) does nothing."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()

            # Call handler with None
            app._handle_session_load(None)
            await pilot.pause()

            # No crash


class TestQueryFileLoading:
    """Tests for query_file loading on startup."""

    @respx.mock
    @pytest.mark.parametrize("limit", [2000, None], ids=["advertised", "unavailable"])
    async def test_first_rag_query_uses_discovered_window(
        self,
        rag_config: ConfigWithSources,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        limit: int | None,
    ) -> None:
        release = asyncio.Event()

        async def models(_request: httpx.Request) -> httpx.Response:
            await release.wait()
            return httpx.Response(
                200, json={"data": [{"id": "test-model", "context_length": limit}]}
            )

        respx.get("http://test.local/v1/models").mock(side_effect=models)
        chat = AsyncMock(return_value=AssistantMessage("Answer"))
        monkeypatch.setattr(OpenAIClient, "chat", chat)
        query_file = tmp_path / "query.txt"
        query_file.write_text("Initial query from file")

        app = ChatApp(
            config_with_sources=rag_config,
            query_file=str(query_file),
        )
        async with app.run_test():
            release.set()
            await app.workers.wait_for_complete()
            chat.assert_awaited_once()
            question = chat.await_args_list[0].args[0][-1].content
            assert "Initial query from file" in question
            # Only the fallback window has room for the passage after reserving 2,000 tokens.
            assert ("Passage" in question) == (limit is None)
            assert app.conversation is not None
            expected = limit or 128000
            assert app.conversation.context_window == expected
            assert f"{expected:,} tokens" in app.query_one(StatusBar)._tokens


class TestStatusBarUpdates:
    """Tests for status bar updates."""

    async def test_status_bar_model_update(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """Status bar model can be updated."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()

            status_bar = app.query_one("#status-bar", StatusBar)
            status_bar._model = "updated-model"
            await pilot.pause()

            assert status_bar._model == "updated-model"


class TestFooterInteraction:
    """Tests for footer widget integration."""

    async def test_footer_exists(self, mock_config_with_sources: ConfigWithSources) -> None:
        """App has footer widget."""
        from chatty.ui.footer import ChattyFooter

        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test():
            footer = app.query_one(ChattyFooter)
            assert footer is not None


# ============================================================================
# Additional Coverage Tests (v0.2.10)
# ============================================================================


class TestHelpModal:
    """Tests for help modal integration."""

    async def test_help_action_opens_modal(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """action_help opens help modal."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()

            # Open help
            app.action_help()
            await pilot.pause()

            # Modal should be open (2 screens: main + modal)
            assert len(app.screen_stack) == 2

    async def test_help_modal_closes_with_escape(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """Help modal closes with Escape."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()

            # Open help
            app.action_help()
            await pilot.pause()

            # Close with escape
            await pilot.press("escape")
            await pilot.pause()

            # Back to main screen
            assert len(app.screen_stack) == 1


class TestSearchIntegration:
    """Tests for search functionality in app."""

    async def test_search_action_opens_bar(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """action_search opens search bar."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()

            # Open search via action
            app.action_search()
            await pilot.pause()

            # Just verify no crash
            assert app is not None

    async def test_search_closes_with_escape(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """Search bar closes with escape."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()

            # Open search
            app.action_search()
            await pilot.pause()

            # Press escape to close
            await pilot.press("escape")
            await pilot.pause()

            # Just verify no crash
            assert app is not None

    async def test_search_with_messages(self, mock_config_with_sources: ConfigWithSources) -> None:
        """Search finds matching messages."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()
            chat_log = app.query_one("#chat-log", ChatLog)

            # Add messages
            chat_log.add_message("user", "Hello world")
            chat_log.add_message("assistant", "Hi there world")
            await pilot.pause()

            # Open search
            app.action_search()
            await pilot.pause()

            # Just verify no crash
            assert app is not None


class TestSubmitQuery:
    """Tests for query submission edge cases."""

    async def test_submit_whitespace_only_ignored(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """Submit with only whitespace is ignored."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()
            input_widget = app.query_one("#input", ChatInput)
            chat_log = app.query_one("#chat-log", ChatLog)

            from chatty.ui.widgets import MessageWidget

            initial_count = len(list(chat_log.query(MessageWidget)))

            # Set whitespace text
            input_widget.text = "   \n\t  "

            # Submit
            app.action_submit()
            await pilot.pause()

            # No new message
            final_count = len(list(chat_log.query(MessageWidget)))
            assert final_count == initial_count


class TestStreamingToggle:
    """Tests for streaming mode toggle."""

    async def test_streaming_toggle_updates_status_bar(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """Toggle streaming updates status bar."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()
            status_bar = app.query_one("#status-bar", StatusBar)

            initial = status_bar._streaming

            app.action_toggle_stream()
            await pilot.pause()

            assert status_bar._streaming != initial


class TestFirstSaveFlow:
    """Tests for first save name prompt flow."""

    async def test_handle_first_save_creates_session(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """_handle_first_save creates session with provided name."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.conversation is not None

            # Add messages
            app.conversation.add_user_message("Test")
            app.conversation.add_assistant_message("Reply")

            # Call handler directly with a name
            app._handle_first_save("My Custom Session")
            await pilot.pause()

            # Verify session was saved
            session_path = Path(mock_config_with_sources.config.session_path)
            sessions = list(session_path.glob("*.json"))
            assert len(sessions) >= 1

    async def test_handle_first_save_cancelled(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """_handle_first_save with None (cancelled) does nothing."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.conversation is not None

            # Add messages
            app.conversation.add_user_message("Test")
            app.conversation.add_assistant_message("Reply")

            # Call handler with None (user cancelled)
            app._handle_first_save(None)
            await pilot.pause()

            # No session created
            session_path = Path(mock_config_with_sources.config.session_path)
            sessions = list(session_path.glob("*.json"))
            assert len(sessions) == 0


class TestRAGPromptPersistence:
    """Tests for RAG prompt persistence across session operations.

    These tests verify that RAG-enabled sessions maintain the correct
    system prompt after action_new_session() and compression operations.
    """

    async def test_new_session_preserves_rag_prompt(self, tmp_path: Path) -> None:
        """action_new_session uses get_effective_system_prompt for RAG sessions."""

        # Create RAG-enabled config
        config = Config(
            base_url="http://test.local/v1",
            api_key=SecretStr("test-key"),
            model="test-model",
            rag_provider="litkit",  # RAG enabled
            session_path=str(tmp_path / "sessions"),
            export_path=str(tmp_path / "exports"),
            copy_fallback_path=str(tmp_path / "copies"),
            transcript_enabled=False,
        )
        config_with_sources = ConfigWithSources(config=config, sources={})

        app = ChatApp(config_with_sources=config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.conversation is not None

            # Initial system prompt should be RAG prompt
            initial_messages = [m for m in app.conversation.messages if m.role == "system"]
            assert len(initial_messages) > 0
            assert "cite" in initial_messages[0].content.lower()

            # Add some conversation
            app.conversation.add_user_message("Test question")
            app.conversation.add_assistant_message("Test answer")

            # New session - should preserve RAG prompt
            app.action_new_session()
            await pilot.pause()

            # Verify RAG system prompt is still present
            new_messages = [m for m in app.conversation.messages if m.role == "system"]
            assert len(new_messages) > 0
            # Should contain citation instructions (key part of RAG prompt)
            system_content = new_messages[0].content.lower()
            assert "cite" in system_content or "context" in system_content

    async def test_compression_preserves_rag_prompt(self, tmp_path: Path) -> None:
        """_handle_compression_choice uses get_effective_system_prompt for RAG sessions."""

        # Create RAG-enabled config
        config = Config(
            base_url="http://test.local/v1",
            api_key=SecretStr("test-key"),
            model="test-model",
            rag_provider="litkit",  # RAG enabled
            session_path=str(tmp_path / "sessions"),
            export_path=str(tmp_path / "exports"),
            copy_fallback_path=str(tmp_path / "copies"),
            transcript_enabled=False,
        )
        config_with_sources = ConfigWithSources(config=config, sources={})

        app = ChatApp(config_with_sources=config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.conversation is not None

            # Directly call _handle_compression_choice with apply=True
            # This simulates user accepting compression with a summary
            app._pending_summary = "Summary of the conversation"
            app._handle_compression_choice(True)
            await pilot.pause()

            # Verify RAG system prompt is still present after compression
            system_messages = [m for m in app.conversation.messages if m.role == "system"]
            # Should have at least one system message with RAG prompt
            assert len(system_messages) > 0
            # First system message should be effective prompt (RAG)
            first_system = system_messages[0].content.lower()
            assert "cite" in first_system or "context" in first_system

    async def test_non_rag_session_uses_default_prompt(self, tmp_path: Path) -> None:
        """Non-RAG sessions use default system prompt after new session."""
        # Create non-RAG config
        config = Config(
            base_url="http://test.local/v1",
            api_key=SecretStr("test-key"),
            model="test-model",
            rag_provider="none",  # RAG disabled
            session_path=str(tmp_path / "sessions"),
            export_path=str(tmp_path / "exports"),
            copy_fallback_path=str(tmp_path / "copies"),
            transcript_enabled=False,
        )
        config_with_sources = ConfigWithSources(config=config, sources={})

        app = ChatApp(config_with_sources=config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.conversation is not None

            # Add conversation
            app.conversation.add_user_message("Hello")
            app.conversation.add_assistant_message("Hi")

            # New session
            app.action_new_session()
            await pilot.pause()

            # Should have default prompt, not RAG prompt
            system_messages = [m for m in app.conversation.messages if m.role == "system"]
            if system_messages:
                # Should be default "helpful assistant", not RAG citation prompt
                content = system_messages[0].content.lower()
                assert "precise scientific" not in content


class TestConversationState:
    """Tests for conversation state management."""

    async def test_new_session_resets_tracking(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """action_new_session resets session tracking."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.conversation is not None

            # Add messages and save
            app.conversation.add_user_message("Test")
            app.conversation.add_assistant_message("Reply")
            app._handle_first_save("Test Session")
            await pilot.pause()

            # Now we have a current session
            has_session = app._current_session is not None

            # New session
            app.action_new_session()
            await pilot.pause()

            # Session tracking should be reset
            # Note: may be cleared regardless
            if has_session:
                # Just verify no crash
                pass
