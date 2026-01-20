"""Tests for session_handlers module.

Tests the session management functions extracted from ChatApp:
- load_session_file
- load_and_submit_query_file
- handle_session_load
- handle_first_save
- save_current_session
- handle_file_path
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from chatty.client.openai_client import Message
from chatty.core.session import Session, SessionMetadata
from chatty.ui.session_handlers import (
    handle_file_path,
    handle_first_save,
    handle_session_load,
    load_and_submit_query_file,
    load_session_file,
    save_current_session,
)


@pytest.fixture
def mock_app() -> Generator[MagicMock, None, None]:
    """Create a mock ChatApp for testing handlers."""
    app = MagicMock()
    app.session_file = None
    app.query_file = None
    app.conversation = MagicMock()
    app.conversation.messages = []
    app.config = MagicMock()
    app.config.model = "gpt-4"
    app.config.system_prompt = "You are a helpful assistant."
    app.config.get_session_path.return_value = Path("/tmp/sessions")
    app.transcript = MagicMock()
    app._pending_user_text = "test message"
    app._current_session = None

    # Mock query_one to return mock widgets
    mock_chat_log = MagicMock()
    mock_input = MagicMock()
    mock_status_bar = MagicMock()

    def query_one_side_effect(selector: str, _widget_type: type | None = None) -> MagicMock:
        if "chat-log" in selector:
            return mock_chat_log
        if "input" in selector:
            return mock_input
        if "status-bar" in selector:
            return mock_status_bar
        return MagicMock()

    app.query_one = MagicMock(side_effect=query_one_side_effect)

    yield app


@pytest.fixture
def sample_session() -> Generator[Session, None, None]:
    """Create a sample session for testing."""
    metadata = SessionMetadata.create(
        name="Test Session",
        model="gpt-4",
        message_count=2,
    )
    yield Session(
        metadata=metadata,
        system_prompt="You are helpful.",
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ],
    )


# ============================================================================
# load_session_file Tests
# ============================================================================


class TestLoadSessionFile:
    """Tests for load_session_file function."""

    def test_load_session_file_no_file(self, mock_app: MagicMock) -> None:
        """Does nothing when session_file is None."""
        mock_app.session_file = None
        load_session_file(mock_app)
        # Should not call query_one since we exit early
        # No error should occur

    def test_load_session_file_success(
        self, mock_app: MagicMock, sample_session: Session, tmp_path: Path
    ) -> None:
        """Successfully loads session from file."""
        # Save a session file
        from chatty.core.session import save_session

        filepath = save_session(sample_session, tmp_path)
        mock_app.session_file = filepath

        load_session_file(mock_app)

        # Verify session was loaded
        assert mock_app._current_session is not None
        assert mock_app._current_session.metadata.name == "Test Session"
        # Verify conversation was updated
        mock_app.conversation.clear.assert_called_once()

    def test_load_session_file_error(self, mock_app: MagicMock, tmp_path: Path) -> None:
        """Handles error when loading invalid session file."""
        # Create an invalid session file
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json {")
        mock_app.session_file = invalid_file

        load_session_file(mock_app)

        # Should show error message
        chat_log = mock_app.query_one("#chat-log")
        chat_log.add_message.assert_called()
        # Check last call was an error
        last_call = chat_log.add_message.call_args_list[-1]
        assert last_call[0][0] == "error"

    def test_load_session_file_restores_messages(self, mock_app: MagicMock, tmp_path: Path) -> None:
        """Restores all message types correctly."""
        from chatty.core.session import save_session

        metadata = SessionMetadata.create(
            name="Full Session",
            model="gpt-4",
            message_count=3,
        )
        session = Session(
            metadata=metadata,
            system_prompt="System prompt",
            messages=[
                Message(role="system", content="System prompt"),
                Message(role="user", content="User message"),
                Message(role="assistant", content="Assistant message"),
            ],
        )
        filepath = save_session(session, tmp_path)
        mock_app.session_file = filepath

        load_session_file(mock_app)

        # Verify all message types were added to conversation
        assert mock_app.conversation.add_system_message.called
        assert mock_app.conversation.add_user_message.called
        assert mock_app.conversation.add_assistant_message.called


# ============================================================================
# load_and_submit_query_file Tests
# ============================================================================


class TestLoadAndSubmitQueryFile:
    """Tests for load_and_submit_query_file function."""

    def test_load_query_file_no_file(self, mock_app: MagicMock) -> None:
        """Does nothing when query_file is None."""
        mock_app.query_file = None
        load_and_submit_query_file(mock_app)
        # Should not call any methods

    def test_load_query_file_success(self, mock_app: MagicMock, tmp_path: Path) -> None:
        """Successfully loads and submits query from file."""
        query_file = tmp_path / "query.txt"
        query_file.write_text("What is Python?")
        mock_app.query_file = query_file

        load_and_submit_query_file(mock_app)

        # Verify input was set and submit called
        input_widget = mock_app.query_one("#input")
        assert input_widget.text == "What is Python?"
        mock_app.action_submit.assert_called_once()

    def test_load_query_file_empty(self, mock_app: MagicMock, tmp_path: Path) -> None:
        """Does not submit empty query file."""
        query_file = tmp_path / "empty.txt"
        query_file.write_text("   \n  ")  # whitespace only
        mock_app.query_file = query_file

        load_and_submit_query_file(mock_app)

        # Should not call submit since content is empty after strip
        mock_app.action_submit.assert_not_called()

    def test_load_query_file_error(self, mock_app: MagicMock) -> None:
        """Handles error when query file doesn't exist."""
        mock_app.query_file = Path("/nonexistent/query.txt")

        load_and_submit_query_file(mock_app)

        # Should show error message
        chat_log = mock_app.query_one("#chat-log")
        chat_log.add_message.assert_called()
        last_call = chat_log.add_message.call_args_list[-1]
        assert last_call[0][0] == "error"


# ============================================================================
# handle_session_load Tests
# ============================================================================


class TestHandleSessionLoad:
    """Tests for handle_session_load function."""

    def test_handle_session_load_cancelled(self, mock_app: MagicMock) -> None:
        """Does nothing when filepath is None (cancelled)."""
        handle_session_load(mock_app, None)
        # Should not call any methods
        mock_app.conversation.clear.assert_not_called()

    def test_handle_session_load_success(
        self, mock_app: MagicMock, sample_session: Session, tmp_path: Path
    ) -> None:
        """Successfully loads session from browser selection."""
        from chatty.core.session import save_session

        filepath = save_session(sample_session, tmp_path)

        handle_session_load(mock_app, filepath)

        # Verify session was loaded
        assert mock_app._current_session is not None
        assert mock_app._current_session.metadata.name == "Test Session"
        # Verify chat log was cleared and populated
        chat_log = mock_app.query_one("#chat-log")
        chat_log.clear_messages.assert_called_once()

    def test_handle_session_load_error(self, mock_app: MagicMock, tmp_path: Path) -> None:
        """Handles error when loading invalid session."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not json")

        handle_session_load(mock_app, invalid_file)

        # Should show error message
        chat_log = mock_app.query_one("#chat-log")
        # Find error call
        error_calls = [c for c in chat_log.add_message.call_args_list if c[0][0] == "error"]
        assert len(error_calls) > 0

    def test_handle_session_load_updates_status_bar(
        self, mock_app: MagicMock, sample_session: Session, tmp_path: Path
    ) -> None:
        """Updates status bar after loading session."""
        from chatty.core.session import save_session

        filepath = save_session(sample_session, tmp_path)

        handle_session_load(mock_app, filepath)

        # Verify status bar was updated
        status_bar = mock_app.query_one("#status-bar")
        status_bar.update_status.assert_called()


# ============================================================================
# handle_first_save Tests
# ============================================================================


class TestHandleFirstSave:
    """Tests for handle_first_save function."""

    def test_handle_first_save_cancelled(self, mock_app: MagicMock) -> None:
        """Does nothing when name is None (cancelled)."""
        handle_first_save(mock_app, None)
        assert mock_app._current_session is None

    def test_handle_first_save_no_conversation(self, mock_app: MagicMock) -> None:
        """Does nothing when conversation is None."""
        mock_app.conversation = None
        handle_first_save(mock_app, "My Session")
        assert mock_app._current_session is None

    def test_handle_first_save_empty_name(self, mock_app: MagicMock) -> None:
        """Does nothing when name is empty string."""
        handle_first_save(mock_app, "")
        assert mock_app._current_session is None

    def test_handle_first_save_success(self, mock_app: MagicMock, tmp_path: Path) -> None:
        """Creates session with provided name."""
        mock_app.conversation.messages = [
            Message(role="user", content="Hello"),
        ]
        mock_app.config.get_session_path.return_value = tmp_path

        # Patch save_current_session to avoid actual file ops
        with patch("chatty.ui.session_handlers.save_current_session") as mock_save:
            handle_first_save(mock_app, "New Session")

        # Verify session was created
        assert mock_app._current_session is not None
        assert mock_app._current_session.metadata.name == "New Session"
        mock_save.assert_called_once_with(mock_app)


# ============================================================================
# save_current_session Tests
# ============================================================================


class TestSaveCurrentSession:
    """Tests for save_current_session function."""

    def test_save_current_session_no_session(self, mock_app: MagicMock) -> None:
        """Does nothing when no current session."""
        mock_app._current_session = None
        save_current_session(mock_app)
        # Should not raise error

    def test_save_current_session_no_conversation(
        self, mock_app: MagicMock, sample_session: Session
    ) -> None:
        """Does nothing when no conversation."""
        mock_app._current_session = sample_session
        mock_app.conversation = None
        save_current_session(mock_app)
        # Should not raise error

    def test_save_current_session_success(
        self, mock_app: MagicMock, sample_session: Session, tmp_path: Path
    ) -> None:
        """Successfully saves session to disk."""
        mock_app._current_session = sample_session
        mock_app.conversation.messages = sample_session.messages
        mock_app.config.get_session_path.return_value = tmp_path

        save_current_session(mock_app)

        # Verify success message was shown
        chat_log = mock_app.query_one("#chat-log")
        success_calls = [c for c in chat_log.add_message.call_args_list if c[0][0] == "system"]
        assert len(success_calls) > 0

        # Verify file was created
        session_files = list(tmp_path.glob("*.json"))
        assert len(session_files) == 1

    def test_save_current_session_error(self, mock_app: MagicMock, sample_session: Session) -> None:
        """Handles error when saving fails."""
        mock_app._current_session = sample_session
        mock_app.conversation.messages = sample_session.messages
        # Invalid path that can't be created
        mock_app.config.get_session_path.return_value = Path("/nonexistent/path/sessions")

        save_current_session(mock_app)

        # Should show error message
        chat_log = mock_app.query_one("#chat-log")
        error_calls = [c for c in chat_log.add_message.call_args_list if c[0][0] == "error"]
        assert len(error_calls) > 0


# ============================================================================
# handle_file_path Tests
# ============================================================================


class TestHandleFilePath:
    """Tests for handle_file_path function."""

    def test_handle_file_path_cancelled(self, mock_app: MagicMock) -> None:
        """Does nothing when path is None (cancelled)."""
        handle_file_path(mock_app, None)
        # Should not call any methods

    def test_handle_file_path_success(self, mock_app: MagicMock, tmp_path: Path) -> None:
        """Successfully loads file content into input."""
        query_file = tmp_path / "query.txt"
        query_file.write_text("What is AI?")

        handle_file_path(mock_app, str(query_file))

        # Verify input was set
        input_widget = mock_app.query_one("#input")
        assert input_widget.text == "What is AI?"
        input_widget.focus.assert_called_once()

    def test_handle_file_path_not_found(self, mock_app: MagicMock) -> None:
        """Shows error when file not found."""
        handle_file_path(mock_app, "/nonexistent/file.txt")

        chat_log = mock_app.query_one("#chat-log")
        error_calls = [c for c in chat_log.add_message.call_args_list if c[0][0] == "error"]
        assert len(error_calls) == 1
        assert "not found" in error_calls[0][0][1].lower()

    def test_handle_file_path_permission_error(self, mock_app: MagicMock, tmp_path: Path) -> None:
        """Shows error when permission denied."""
        # Create a file and make it unreadable
        restricted_file = tmp_path / "restricted.txt"
        restricted_file.write_text("secret")
        restricted_file.chmod(0o000)

        try:
            handle_file_path(mock_app, str(restricted_file))

            chat_log = mock_app.query_one("#chat-log")
            error_calls = [c for c in chat_log.add_message.call_args_list if c[0][0] == "error"]
            assert len(error_calls) == 1
            assert "permission" in error_calls[0][0][1].lower()
        finally:
            # Restore permissions so tmp_path can be cleaned up
            restricted_file.chmod(0o644)

    def test_handle_file_path_empty_file(self, mock_app: MagicMock, tmp_path: Path) -> None:
        """Shows error when file is empty."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        handle_file_path(mock_app, str(empty_file))

        chat_log = mock_app.query_one("#chat-log")
        error_calls = [c for c in chat_log.add_message.call_args_list if c[0][0] == "error"]
        assert len(error_calls) == 1
        assert "empty" in error_calls[0][0][1].lower()

    def test_handle_file_path_whitespace_only(self, mock_app: MagicMock, tmp_path: Path) -> None:
        """Shows error when file contains only whitespace."""
        whitespace_file = tmp_path / "whitespace.txt"
        whitespace_file.write_text("   \n\t  \n")

        handle_file_path(mock_app, str(whitespace_file))

        chat_log = mock_app.query_one("#chat-log")
        error_calls = [c for c in chat_log.add_message.call_args_list if c[0][0] == "error"]
        assert len(error_calls) == 1
        assert "empty" in error_calls[0][0][1].lower()

    def test_handle_file_path_generic_error(self, mock_app: MagicMock) -> None:
        """Shows generic error for unexpected exceptions."""
        # Use mock_open to simulate a generic exception
        with patch("builtins.open", mock_open()) as m:
            m.side_effect = OSError("Disk full")
            handle_file_path(mock_app, "/some/file.txt")

        chat_log = mock_app.query_one("#chat-log")
        error_calls = [c for c in chat_log.add_message.call_args_list if c[0][0] == "error"]
        assert len(error_calls) == 1
        assert "failed to load" in error_calls[0][0][1].lower()


# ============================================================================
# Session Dirty Flag Tests (v0.2.15)
# ============================================================================


class TestSessionDirtyFlag:
    """Tests for session dirty flag tracking."""

    def test_dirty_flag_cleared_on_save(
        self, mock_app: MagicMock, sample_session: Session, tmp_path: Path
    ) -> None:
        """Dirty flag is cleared after successful save."""
        mock_app._current_session = sample_session
        mock_app.conversation.messages = sample_session.messages
        mock_app.config.get_session_path.return_value = tmp_path
        mock_app._session_dirty = True

        save_current_session(mock_app)

        # Verify dirty flag was cleared
        assert mock_app._session_dirty is False

    def test_dirty_flag_cleared_on_load(
        self, mock_app: MagicMock, sample_session: Session, tmp_path: Path
    ) -> None:
        """Dirty flag is cleared after loading session."""
        from chatty.core.session import save_session

        filepath = save_session(sample_session, tmp_path)
        mock_app._session_dirty = True

        handle_session_load(mock_app, filepath)

        # Verify dirty flag was cleared
        assert mock_app._session_dirty is False

    def test_dirty_flag_not_cleared_on_save_error(
        self, mock_app: MagicMock, sample_session: Session
    ) -> None:
        """Dirty flag remains set when save fails."""
        mock_app._current_session = sample_session
        mock_app.conversation.messages = sample_session.messages
        mock_app._session_dirty = True
        # Invalid path that can't be created
        mock_app.config.get_session_path.return_value = Path("/nonexistent/path")

        save_current_session(mock_app)

        # Dirty flag should still be set (save failed)
        assert mock_app._session_dirty is True

    def test_dirty_flag_not_cleared_on_load_error(
        self, mock_app: MagicMock, tmp_path: Path
    ) -> None:
        """Dirty flag remains set when load fails."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not json")
        mock_app._session_dirty = True

        handle_session_load(mock_app, invalid_file)

        # Dirty flag should still be set (load failed)
        assert mock_app._session_dirty is True

    def test_dirty_flag_not_changed_on_cancel(self, mock_app: MagicMock) -> None:
        """Dirty flag unchanged when load cancelled."""
        mock_app._session_dirty = True

        handle_session_load(mock_app, None)  # Cancelled

        # Dirty flag should still be set
        assert mock_app._session_dirty is True
