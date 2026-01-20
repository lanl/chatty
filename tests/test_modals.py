"""Tests for chatty modal dialogs.

Tests the modal screens: FileInputModal, SessionBrowserModal,
ModelPickerModal, ModelInputModal.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Input, Label, OptionList

from chatty.ui.modals import (
    FileInputModal,
    ModelInputModal,
    ModelPickerModal,
    SessionBrowserModal,
    SessionDeleteConfirmModal,
    SessionRenameModal,
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
