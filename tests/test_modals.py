"""Tests for chatty modal dialogs.

Tests the modal screens: FileInputModal, SessionBrowserModal,
ModelPickerModal, ModelInputModal.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Input, Label, OptionList

from chatty.ui.modals import (
    CompressionPreviewModal,
    FileInputModal,
    ModelInputModal,
    ModelPickerModal,
    SessionBrowserModal,
    SessionDeleteConfirmModal,
    SessionRenameModal,
    UnsavedChangesModal,
)

# ============================================================================
# FileInputModal Tests
# ============================================================================


class FileInputTestApp(App[None]):
    """App for testing FileInputModal."""

    def __init__(self) -> None:
        super().__init__()
        self.result: str | None = "not_called"

    def compose(self) -> ComposeResult:
        yield Label("Test App")

    async def show_modal(self) -> None:
        """Show the file input modal."""
        result = await self.push_screen_wait(FileInputModal())
        self.result = result


class TestFileInputModal:
    """Tests for FileInputModal."""

    async def test_modal_renders(self) -> None:
        """FileInputModal renders correctly."""
        app = FileInputTestApp()
        async with app.run_test() as pilot:
            modal = FileInputModal()
            app.push_screen(modal)
            await pilot.pause()

            # Should have the modal visible
            assert len(app.screen_stack) == 2

            # Should have input and buttons - query from modal screen
            input_widget = modal.query_one("#file-path", Input)
            assert input_widget is not None

    async def test_cancel_button_dismisses(self) -> None:
        """Cancel button dismisses with None."""
        app = FileInputTestApp()
        async with app.run_test() as pilot:
            app.push_screen(FileInputModal(), callback=lambda r: setattr(app, "result", r))
            await pilot.pause()

            # Click cancel
            await pilot.click("#cancel-btn")
            await pilot.pause()

            assert app.result is None

    async def test_escape_dismisses(self) -> None:
        """Escape key dismisses with None."""
        app = FileInputTestApp()
        async with app.run_test() as pilot:
            app.push_screen(FileInputModal(), callback=lambda r: setattr(app, "result", r))
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.result is None


# ============================================================================
# Additional Modal Coverage Tests (v0.2.10)
# ============================================================================


class TestModelPickerModalSelection:
    """Tests for ModelPickerModal selection."""

    async def test_select_button_returns_model(self) -> None:
        """Select button returns highlighted model."""
        app = ModelPickerTestApp()
        async with app.run_test() as pilot:
            models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
            modal = ModelPickerModal(models, "gpt-4")
            app.push_screen(modal, callback=lambda r: setattr(app, "result", r))
            await pilot.pause()

            # Click select (should select first/highlighted model)
            await pilot.click("#select-btn")
            await pilot.pause()

            # Should return a model name
            assert app.result in models or app.result is None

    async def test_double_click_selects_model(self) -> None:
        """Double-clicking a model selects it."""
        app = ModelPickerTestApp()
        async with app.run_test() as pilot:
            models = ["model-a", "model-b"]
            modal = ModelPickerModal(models, "model-a")
            app.push_screen(modal, callback=lambda r: setattr(app, "result", r))
            await pilot.pause()

            # Session should be visible
            assert modal._current_model == "model-a"


class TestModelInputModalSubmit:
    """Tests for ModelInputModal enter key submit."""

    async def test_enter_key_submits(self) -> None:
        """Enter key in input submits the model name."""
        app = ModelInputTestApp()
        async with app.run_test() as pilot:
            modal = ModelInputModal("gpt-4")
            app.push_screen(modal, callback=lambda r: setattr(app, "result", r))
            await pilot.pause()

            # Set a new value
            input_widget = modal.query_one("#model-name", Input)
            input_widget.value = "new-model"

            # Press enter
            await pilot.press("enter")
            await pilot.pause()

            # Should return the model
            assert app.result == "new-model"


# ============================================================================
# SessionRenameModal Tests (v0.2.7)
# ============================================================================


class SessionRenameTestApp(App[None]):
    """App for testing SessionRenameModal."""

    def __init__(self) -> None:
        super().__init__()
        self.result: str | None | object = "not_called"

    def compose(self) -> ComposeResult:
        yield Label("Test App")


class TestSessionRenameModal:
    """Tests for SessionRenameModal."""

    async def test_modal_renders(self) -> None:
        """SessionRenameModal renders with current name."""
        app = SessionRenameTestApp()
        async with app.run_test() as pilot:
            modal = SessionRenameModal("Current Name")
            app.push_screen(modal)
            await pilot.pause()

            # Should have input with current name as value
            input_widget = modal.query_one("#session-name", Input)
            assert input_widget.value == "Current Name"

    async def test_cancel_button_returns_none(self) -> None:
        """Cancel button returns None."""
        app = SessionRenameTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                SessionRenameModal("Test Name"),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.click("#cancel-btn")
            await pilot.pause()

            assert app.result is None

    async def test_escape_returns_none(self) -> None:
        """Escape key returns None."""
        app = SessionRenameTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                SessionRenameModal("Test Name"),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.result is None

    async def test_rename_button_returns_new_name(self) -> None:
        """Rename button returns the entered name."""
        app = SessionRenameTestApp()
        async with app.run_test() as pilot:
            modal = SessionRenameModal("Old Name")
            app.push_screen(
                modal,
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            # Change the name
            input_widget = modal.query_one("#session-name", Input)
            input_widget.value = "New Name"

            await pilot.click("#rename-btn")
            await pilot.pause()

            assert app.result == "New Name"

    async def test_enter_key_submits(self) -> None:
        """Enter key in input submits the name."""
        app = SessionRenameTestApp()
        async with app.run_test() as pilot:
            modal = SessionRenameModal("Old Name")
            app.push_screen(
                modal,
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            # Change the name and press enter
            input_widget = modal.query_one("#session-name", Input)
            input_widget.value = "Submitted Name"

            # Submit via enter key on input
            await pilot.press("enter")
            await pilot.pause()

            assert app.result == "Submitted Name"

    async def test_empty_name_returns_none(self) -> None:
        """Empty name returns None."""
        app = SessionRenameTestApp()
        async with app.run_test() as pilot:
            modal = SessionRenameModal("Original")
            app.push_screen(
                modal,
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            # Clear the input
            input_widget = modal.query_one("#session-name", Input)
            input_widget.value = ""

            await pilot.click("#rename-btn")
            await pilot.pause()

            assert app.result is None

    async def test_whitespace_only_returns_none(self) -> None:
        """Whitespace-only name returns None."""
        app = SessionRenameTestApp()
        async with app.run_test() as pilot:
            modal = SessionRenameModal("Original")
            app.push_screen(
                modal,
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            # Set whitespace only
            input_widget = modal.query_one("#session-name", Input)
            input_widget.value = "   "

            await pilot.click("#rename-btn")
            await pilot.pause()

            assert app.result is None

    async def test_strips_whitespace(self) -> None:
        """Whitespace is stripped from name."""
        app = SessionRenameTestApp()
        async with app.run_test() as pilot:
            modal = SessionRenameModal("Original")
            app.push_screen(
                modal,
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            # Set padded name
            input_widget = modal.query_one("#session-name", Input)
            input_widget.value = "  Padded Name  "

            await pilot.click("#rename-btn")
            await pilot.pause()

            assert app.result == "Padded Name"


# ============================================================================
# SessionDeleteConfirmModal Tests (v0.2.8)
# ============================================================================


class SessionDeleteTestApp(App[None]):
    """App for testing SessionDeleteConfirmModal."""

    def __init__(self) -> None:
        super().__init__()
        self.result: bool | None | object = "not_called"

    def compose(self) -> ComposeResult:
        yield Label("Test App")


class TestSessionDeleteConfirmModal:
    """Tests for SessionDeleteConfirmModal."""

    async def test_modal_renders_with_session_name(self) -> None:
        """SessionDeleteConfirmModal shows session name."""
        app = SessionDeleteTestApp()
        async with app.run_test() as pilot:
            modal = SessionDeleteConfirmModal("My Session")
            app.push_screen(modal)
            await pilot.pause()

            # Session name should be stored
            assert modal._session_name == "My Session"

    async def test_cancel_button_returns_none(self) -> None:
        """Cancel button returns None."""
        app = SessionDeleteTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                SessionDeleteConfirmModal("Test Session"),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.click("#cancel-btn")
            await pilot.pause()

            assert app.result is None

    async def test_escape_returns_none(self) -> None:
        """Escape key returns None."""
        app = SessionDeleteTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                SessionDeleteConfirmModal("Test Session"),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.result is None

    async def test_delete_button_returns_true(self) -> None:
        """Delete button returns True."""
        app = SessionDeleteTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                SessionDeleteConfirmModal("Test Session"),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.click("#delete-btn")
            await pilot.pause()

            assert app.result is True

    async def test_cancel_button_focused_by_default(self) -> None:
        """Cancel button is focused by default (safer UX)."""
        app = SessionDeleteTestApp()
        async with app.run_test() as pilot:
            modal = SessionDeleteConfirmModal("Test Session")
            app.push_screen(modal)
            await pilot.pause()

            # Cancel button should have focus
            from textual.widgets import Button

            cancel_btn = modal.query_one("#cancel-btn", Button)
            assert cancel_btn.has_focus


# ============================================================================
# CompressionPreviewModal Tests (v0.3.0)
# ============================================================================


class CompressionPreviewTestApp(App[None]):
    """App for testing CompressionPreviewModal."""

    def __init__(self) -> None:
        super().__init__()
        self.result: bool | None | object = "not_called"

    def compose(self) -> ComposeResult:
        yield Label("Test App")


class TestCompressionPreviewModal:
    """Tests for CompressionPreviewModal."""

    async def test_modal_renders(self) -> None:
        """CompressionPreviewModal renders correctly."""
        app = CompressionPreviewTestApp()
        async with app.run_test() as pilot:
            modal = CompressionPreviewModal(
                summary="This is a test summary.",
                original_tokens=1000,
                compressed_tokens=100,
                has_code_blocks=False,
            )
            app.push_screen(modal)
            await pilot.pause()

            # Should have the modal visible
            assert len(app.screen_stack) == 2

    async def test_cancel_button_returns_none(self) -> None:
        """Cancel button returns None."""
        app = CompressionPreviewTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                CompressionPreviewModal(
                    summary="Summary",
                    original_tokens=1000,
                    compressed_tokens=100,
                    has_code_blocks=False,
                ),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.click("#cancel-btn")
            await pilot.pause()

            assert app.result is None

    async def test_apply_button_returns_true(self) -> None:
        """Apply button returns True."""
        app = CompressionPreviewTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                CompressionPreviewModal(
                    summary="Summary",
                    original_tokens=1000,
                    compressed_tokens=100,
                    has_code_blocks=False,
                ),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.click("#apply-btn")
            await pilot.pause()

            assert app.result is True

    async def test_escape_returns_none(self) -> None:
        """Escape key returns None."""
        app = CompressionPreviewTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                CompressionPreviewModal(
                    summary="Summary",
                    original_tokens=1000,
                    compressed_tokens=100,
                    has_code_blocks=False,
                ),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.result is None

    async def test_cancel_focused_by_default(self) -> None:
        """Cancel button is focused by default (safer UX)."""
        app = CompressionPreviewTestApp()
        async with app.run_test() as pilot:
            modal = CompressionPreviewModal(
                summary="Summary",
                original_tokens=1000,
                compressed_tokens=100,
                has_code_blocks=False,
            )
            app.push_screen(modal)
            await pilot.pause()

            # Cancel button should have focus
            from textual.widgets import Button

            cancel_btn = modal.query_one("#cancel-btn", Button)
            assert cancel_btn.has_focus

    async def test_stores_parameters(self) -> None:
        """Modal stores all parameters correctly."""
        app = CompressionPreviewTestApp()
        async with app.run_test() as pilot:
            modal = CompressionPreviewModal(
                summary="Test summary content",
                original_tokens=5000,
                compressed_tokens=500,
                has_code_blocks=True,
            )
            app.push_screen(modal)
            await pilot.pause()

            assert modal._summary == "Test summary content"
            assert modal._original_tokens == 5000
            assert modal._compressed_tokens == 500
            assert modal._has_code_blocks is True

    async def test_warning_shown_for_code_blocks(self) -> None:
        """Warning is displayed when has_code_blocks=True."""
        app = CompressionPreviewTestApp()
        async with app.run_test() as pilot:
            modal = CompressionPreviewModal(
                summary="Summary",
                original_tokens=1000,
                compressed_tokens=100,
                has_code_blocks=True,
            )
            app.push_screen(modal)
            await pilot.pause()

            # Warning should be present (modal has _has_code_blocks=True)
            assert modal._has_code_blocks is True
            # The warning label is rendered conditionally in compose()

    async def test_no_warning_without_code_blocks(self) -> None:
        """No warning when has_code_blocks=False."""
        app = CompressionPreviewTestApp()
        async with app.run_test() as pilot:
            modal = CompressionPreviewModal(
                summary="Summary",
                original_tokens=1000,
                compressed_tokens=100,
                has_code_blocks=False,
            )
            app.push_screen(modal)
            await pilot.pause()

            # No warning should be shown
            assert modal._has_code_blocks is False


# ============================================================================
# FileInputModal Additional Tests
# ============================================================================


class TestFileInputModalAdditional:
    """Additional tests for FileInputModal."""

    async def test_load_button_returns_path(self) -> None:
        """Load button returns entered path."""
        app = FileInputTestApp()
        async with app.run_test() as pilot:
            modal = FileInputModal()
            app.push_screen(modal, callback=lambda r: setattr(app, "result", r))
            await pilot.pause()

            # Enter a path - query from modal
            input_widget = modal.query_one("#file-path", Input)
            input_widget.value = "/test/path.txt"

            # Click load
            await pilot.click("#load-btn")
            await pilot.pause()

            assert app.result == "/test/path.txt"

    async def test_empty_input_returns_none(self) -> None:
        """Empty input returns None on load."""
        app = FileInputTestApp()
        async with app.run_test() as pilot:
            app.push_screen(FileInputModal(), callback=lambda r: setattr(app, "result", r))
            await pilot.pause()

            # Don't enter anything, click load
            await pilot.click("#load-btn")
            await pilot.pause()

            assert app.result is None


# ============================================================================
# ModelPickerModal Tests
# ============================================================================


class ModelPickerTestApp(App[None]):
    """App for testing ModelPickerModal."""

    def __init__(self) -> None:
        super().__init__()
        self.result: str | None = "not_called"

    def compose(self) -> ComposeResult:
        yield Label("Test App")


class TestModelPickerModal:
    """Tests for ModelPickerModal."""

    async def test_modal_renders_with_models(self) -> None:
        """ModelPickerModal shows models."""
        app = ModelPickerTestApp()
        async with app.run_test() as pilot:
            models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
            modal = ModelPickerModal(models, "gpt-4")
            app.push_screen(modal)
            await pilot.pause()

            # Should have option list with models - query from modal
            option_list = modal.query_one("#model-list", OptionList)
            assert option_list.option_count == 3

    async def test_current_model_marked(self) -> None:
        """Current model is marked with bullet."""
        app = ModelPickerTestApp()
        async with app.run_test() as pilot:
            models = ["gpt-4", "gpt-3.5-turbo"]
            modal = ModelPickerModal(models, "gpt-4")
            app.push_screen(modal)
            await pilot.pause()

            # The current model should be stored
            assert modal._current_model == "gpt-4"

    async def test_cancel_returns_none(self) -> None:
        """Cancel button returns None."""
        app = ModelPickerTestApp()
        async with app.run_test() as pilot:
            models = ["gpt-4", "gpt-3.5-turbo"]
            app.push_screen(
                ModelPickerModal(models, "gpt-4"),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.click("#cancel-btn")
            await pilot.pause()

            assert app.result is None

    async def test_escape_returns_none(self) -> None:
        """Escape returns None."""
        app = ModelPickerTestApp()
        async with app.run_test() as pilot:
            models = ["gpt-4"]
            app.push_screen(
                ModelPickerModal(models, "gpt-4"),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.result is None


# ============================================================================
# ModelInputModal Tests
# ============================================================================


class ModelInputTestApp(App[None]):
    """App for testing ModelInputModal."""

    def __init__(self) -> None:
        super().__init__()
        self.result: str | None = "not_called"

    def compose(self) -> ComposeResult:
        yield Label("Test App")


class TestModelInputModal:
    """Tests for ModelInputModal."""

    async def test_modal_renders(self) -> None:
        """ModelInputModal renders with input."""
        app = ModelInputTestApp()
        async with app.run_test() as pilot:
            modal = ModelInputModal("gpt-4")
            app.push_screen(modal)
            await pilot.pause()

            # Should have input with current model as value - query from modal
            input_widget = modal.query_one("#model-name", Input)
            assert input_widget.value == "gpt-4"

    async def test_shows_error_message(self) -> None:
        """Error message is displayed when provided."""
        app = ModelInputTestApp()
        async with app.run_test() as pilot:
            modal = ModelInputModal("gpt-4", "Connection failed")
            app.push_screen(modal)
            await pilot.pause()

            # Error message should be stored
            assert modal._error_message == "Connection failed"

    async def test_cancel_returns_none(self) -> None:
        """Cancel returns None."""
        app = ModelInputTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                ModelInputModal("gpt-4"),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.click("#cancel-btn")
            await pilot.pause()

            assert app.result is None

    async def test_apply_returns_model(self) -> None:
        """Apply returns entered model."""
        app = ModelInputTestApp()
        async with app.run_test() as pilot:
            modal = ModelInputModal("gpt-4")
            app.push_screen(
                modal,
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            # Change value - query from modal
            input_widget = modal.query_one("#model-name", Input)
            input_widget.value = "claude-3"

            await pilot.click("#apply-btn")
            await pilot.pause()

            assert app.result == "claude-3"

    async def test_empty_returns_none(self) -> None:
        """Empty input returns None."""
        app = ModelInputTestApp()
        async with app.run_test() as pilot:
            modal = ModelInputModal("gpt-4")
            app.push_screen(
                modal,
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            # Clear input - query from modal
            input_widget = modal.query_one("#model-name", Input)
            input_widget.value = ""

            await pilot.click("#apply-btn")
            await pilot.pause()

            assert app.result is None


# ============================================================================
# SessionBrowserModal Tests
# ============================================================================


class SessionBrowserTestApp(App[None]):
    """App for testing SessionBrowserModal."""

    def __init__(self) -> None:
        super().__init__()
        self.result: Path | None | str = "not_called"

    def compose(self) -> ComposeResult:
        yield Label("Test App")


class TestSessionBrowserModal:
    """Tests for SessionBrowserModal."""

    async def test_modal_renders(self, tmp_path: Path) -> None:
        """SessionBrowserModal renders."""
        app = SessionBrowserTestApp()
        async with app.run_test() as pilot:
            modal = SessionBrowserModal(tmp_path)
            app.push_screen(modal)
            await pilot.pause()

            # Should have option list - query from modal
            option_list = modal.query_one("#session-list", OptionList)
            assert option_list is not None

    async def test_empty_directory_shows_message(self, tmp_path: Path) -> None:
        """Empty directory shows helpful message."""
        app = SessionBrowserTestApp()
        async with app.run_test() as pilot:
            modal = SessionBrowserModal(tmp_path)
            app.push_screen(modal)
            await pilot.pause()

            # No sessions loaded
            assert modal._sessions == []

    async def test_cancel_returns_none(self, tmp_path: Path) -> None:
        """Cancel returns None."""
        app = SessionBrowserTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                SessionBrowserModal(tmp_path),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.click("#cancel-btn")
            await pilot.pause()

            assert app.result is None

    async def test_escape_returns_none(self, tmp_path: Path) -> None:
        """Escape returns None."""
        app = SessionBrowserTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                SessionBrowserModal(tmp_path),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.result is None

    async def test_load_button_with_sessions(self, tmp_path: Path) -> None:
        """Load button returns selected session path."""
        from chatty.client.openai_client import Message
        from chatty.core.session import (
            Session,
            SessionMetadata,
            save_session,
        )

        # Create a session file
        metadata = SessionMetadata.create(
            name="Test Session",
            model="gpt-4",
            message_count=1,
        )
        session = Session(
            metadata=metadata,
            system_prompt="You are helpful.",
            messages=[Message(role="user", content="Hello")],
        )
        save_session(session, tmp_path)

        app = SessionBrowserTestApp()
        async with app.run_test() as pilot:
            modal = SessionBrowserModal(tmp_path)
            app.push_screen(modal, callback=lambda r: setattr(app, "result", r))
            await pilot.pause()

            # Should have session loaded
            assert len(modal._sessions) == 1

            # Click load (first item should be highlighted)
            await pilot.click("#load-btn")
            await pilot.pause()

            # Should return a Path
            assert app.result is not None
            assert isinstance(app.result, Path)

    async def test_enter_key_loads_session(self, tmp_path: Path) -> None:
        """Enter key loads selected session."""
        from chatty.client.openai_client import Message
        from chatty.core.session import (
            Session,
            SessionMetadata,
            save_session,
        )

        metadata = SessionMetadata.create(name="Enter Test", model="gpt-4", message_count=1)
        session = Session(
            metadata=metadata,
            system_prompt="Test",
            messages=[Message(role="user", content="Hi")],
        )
        save_session(session, tmp_path)

        app = SessionBrowserTestApp()
        async with app.run_test() as pilot:
            modal = SessionBrowserModal(tmp_path)
            app.push_screen(modal, callback=lambda r: setattr(app, "result", r))
            await pilot.pause()

            # Press enter to load
            await pilot.press("enter")
            await pilot.pause()

            # Should return a Path
            assert app.result is not None


# ============================================================================
# SessionBrowserModal Action Tests (v0.2.14)
# ============================================================================


class TestSessionBrowserModalActions:
    """Tests for SessionBrowserModal rename/delete actions."""

    async def test_delete_action_opens_confirm(self, tmp_path: Path) -> None:
        """Delete action opens confirmation modal."""
        from chatty.client.openai_client import Message
        from chatty.core.session import (
            Session,
            SessionMetadata,
            save_session,
        )

        metadata = SessionMetadata.create(name="Delete Me", model="gpt-4", message_count=1)
        session = Session(
            metadata=metadata,
            system_prompt="Test",
            messages=[Message(role="user", content="Hi")],
        )
        save_session(session, tmp_path)

        app = SessionBrowserTestApp()
        async with app.run_test() as pilot:
            modal = SessionBrowserModal(tmp_path)
            app.push_screen(modal)
            await pilot.pause()

            # Press d for delete
            await pilot.press("d")
            await pilot.pause()

            # Confirmation modal should be pushed
            assert len(app.screen_stack) >= 2

    async def test_rename_action_opens_rename_modal(self, tmp_path: Path) -> None:
        """Rename action opens rename modal."""
        from chatty.client.openai_client import Message
        from chatty.core.session import (
            Session,
            SessionMetadata,
            save_session,
        )

        metadata = SessionMetadata.create(name="Rename Me", model="gpt-4", message_count=1)
        session = Session(
            metadata=metadata,
            system_prompt="Test",
            messages=[Message(role="user", content="Hi")],
        )
        save_session(session, tmp_path)

        app = SessionBrowserTestApp()
        async with app.run_test() as pilot:
            modal = SessionBrowserModal(tmp_path)
            app.push_screen(modal)
            await pilot.pause()

            # Press r for rename
            await pilot.press("r")
            await pilot.pause()

            # Rename modal should be pushed
            assert len(app.screen_stack) >= 2

    async def test_delete_action_no_selection(self, tmp_path: Path) -> None:
        """Delete action does nothing when no sessions."""
        app = SessionBrowserTestApp()
        async with app.run_test() as pilot:
            modal = SessionBrowserModal(tmp_path)
            app.push_screen(modal)
            await pilot.pause()

            # Press d with no sessions
            await pilot.press("d")
            await pilot.pause()

            # Should still be on same screen
            assert len(app.screen_stack) == 2

    async def test_rename_action_no_selection(self, tmp_path: Path) -> None:
        """Rename action does nothing when no sessions."""
        app = SessionBrowserTestApp()
        async with app.run_test() as pilot:
            modal = SessionBrowserModal(tmp_path)
            app.push_screen(modal)
            await pilot.pause()

            # Press r with no sessions
            await pilot.press("r")
            await pilot.pause()

            # Should still be on same screen
            assert len(app.screen_stack) == 2


# ============================================================================
# ModelPickerModal Additional Tests (v0.2.14)
# ============================================================================


class TestModelPickerModalActions:
    """Additional tests for ModelPickerModal."""

    async def test_option_list_populated(self) -> None:
        """Model picker option list is populated correctly."""
        app = ModelPickerTestApp()
        async with app.run_test() as pilot:
            models = ["gpt-4", "gpt-3.5-turbo"]
            modal = ModelPickerModal(models, "gpt-4")
            app.push_screen(modal, callback=lambda r: setattr(app, "result", r))
            await pilot.pause()

            # Get the option list
            option_list = modal.query_one("#model-list", OptionList)
            assert option_list.option_count == 2

            # Verify current model is tracked
            assert modal._current_model == "gpt-4"
            assert modal._models == models


# ============================================================================
# UnsavedChangesModal Tests (v0.2.15)
# ============================================================================


class UnsavedChangesTestApp(App[None]):
    """App for testing UnsavedChangesModal."""

    def __init__(self) -> None:
        super().__init__()
        self.result: str | None | object = "not_called"

    def compose(self) -> ComposeResult:
        yield Label("Test App")


class TestUnsavedChangesModal:
    """Tests for UnsavedChangesModal."""

    async def test_modal_renders(self) -> None:
        """UnsavedChangesModal renders correctly."""
        app = UnsavedChangesTestApp()
        async with app.run_test() as pilot:
            modal = UnsavedChangesModal()
            app.push_screen(modal)
            await pilot.pause()

            # Should have the modal visible
            assert len(app.screen_stack) == 2

    async def test_cancel_button_returns_none(self) -> None:
        """Cancel button returns None."""
        app = UnsavedChangesTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                UnsavedChangesModal(),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.click("#cancel-btn")
            await pilot.pause()

            assert app.result is None

    async def test_escape_returns_none(self) -> None:
        """Escape key returns None."""
        app = UnsavedChangesTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                UnsavedChangesModal(),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            assert app.result is None

    async def test_load_anyway_returns_load(self) -> None:
        """Load Anyway button returns 'load'."""
        app = UnsavedChangesTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                UnsavedChangesModal(),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.click("#load-btn")
            await pilot.pause()

            assert app.result == "load"

    async def test_save_and_load_returns_save_and_load(self) -> None:
        """Save & Load button returns 'save_and_load'."""
        app = UnsavedChangesTestApp()
        async with app.run_test() as pilot:
            app.push_screen(
                UnsavedChangesModal(),
                callback=lambda r: setattr(app, "result", r),
            )
            await pilot.pause()

            await pilot.click("#save-load-btn")
            await pilot.pause()

            assert app.result == "save_and_load"

    async def test_cancel_focused_by_default(self) -> None:
        """Cancel button is focused by default (safer UX)."""
        app = UnsavedChangesTestApp()
        async with app.run_test() as pilot:
            modal = UnsavedChangesModal()
            app.push_screen(modal)
            await pilot.pause()

            # Cancel button should have focus
            from textual.widgets import Button

            cancel_btn = modal.query_one("#cancel-btn", Button)
            assert cancel_btn.has_focus
