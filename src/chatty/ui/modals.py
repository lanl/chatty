"""Modal dialog components for chatty UI.

This module contains modal screens for user interaction:
- FileInputModal: File path entry dialog
- SessionBrowserModal: Session list and selection
- ModelPickerModal: Model selection from endpoint
- ModelInputModal: Manual model name entry
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, OptionList
from textual.widgets.option_list import Option

from chatty.core.session import (
    SessionMetadata,
    delete_session,
    get_session_filepath,
    list_sessions,
    rename_session,
)


class FileInputModal(ModalScreen[str | None]):
    """Modal screen for entering a file path.

    Returns the file path string if submitted, None if cancelled.

    Usage:
        app.push_screen(FileInputModal(), callback)
        # callback receives str (file path) or None

    Bindings:
        Enter: Submit the entered path
        Escape: Cancel and dismiss
    """

    CSS = """
    FileInputModal {
        align: center middle;
    }

    #file-dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
    }

    #file-dialog Label {
        margin-bottom: 1;
    }

    #file-dialog Input {
        width: 100%;
        margin-bottom: 1;
    }

    #file-buttons {
        width: 100%;
        height: auto;
        align: right middle;
    }

    #file-buttons Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        """Create the dialog layout."""
        with Vertical(id="file-dialog"):
            yield Label("Enter file path:")
            yield Input(placeholder="/path/to/file.txt", id="file-path")
            with Container(id="file-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Load", variant="primary", id="load-btn")

    def on_mount(self) -> None:
        """Focus the input when modal opens."""
        self.query_one("#file-path", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "load-btn":
            path = self.query_one("#file-path", Input).value.strip()
            self.dismiss(path if path else None)
        else:
            self.dismiss(None)

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        """Handle Enter key in input field."""
        path = self.query_one("#file-path", Input).value.strip()
        self.dismiss(path if path else None)

    def action_cancel(self) -> None:
        """Handle Escape key."""
        self.dismiss(None)


class SessionBrowserModal(ModalScreen[Path | None]):
    """Modal screen for browsing and selecting saved sessions.

    Displays a list of saved sessions with name, date, and message count.
    Returns the selected session file path, or None if cancelled.

    Usage:
        app.push_screen(SessionBrowserModal(session_dir), callback)
        # callback receives Path (session file) or None

    Display Format:
        "Session Name                 2026-01-09  12 msg"

    Bindings:
        Enter: Load selected session
        r: Rename selected session
        Escape: Cancel and dismiss
    """

    CSS = """
    SessionBrowserModal {
        align: center middle;
    }

    #session-dialog {
        width: 70;
        height: 20;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
    }

    #session-dialog Label {
        margin-bottom: 1;
    }

    #session-list {
        height: 1fr;
        margin-bottom: 1;
    }

    #session-buttons {
        width: 100%;
        height: auto;
        align: right middle;
    }

    #session-buttons Button {
        margin-left: 1;
    }

    #empty-message {
        color: $text-muted;
        text-align: center;
        padding: 2;
    }

    #session-hint {
        color: $text-muted;
        text-align: center;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "load", "Load Session"),
        Binding("r", "rename", "Rename"),
        Binding("d", "delete", "Delete"),
    ]

    def __init__(self, session_dir: Path) -> None:
        """Initialize the session browser.

        Args:
            session_dir: Directory containing session files.
        """
        super().__init__()
        self.session_dir = session_dir
        self._sessions: list[SessionMetadata] = []

    def compose(self) -> ComposeResult:
        """Create the dialog layout."""
        with Vertical(id="session-dialog"):
            yield Label("Saved Sessions")
            yield OptionList(id="session-list")
            yield Label(
                "[dim]r[/dim] rename  [dim]d[/dim] delete  [dim]Enter[/dim] load",
                id="session-hint",
            )
            with Container(id="session-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Load", variant="primary", id="load-btn")

    def on_mount(self) -> None:
        """Load sessions when modal opens."""
        self._refresh_session_list()

    def _refresh_session_list(self, select_id: str | None = None) -> None:
        """Refresh the session list from disk.

        Args:
            select_id: Optional session ID to re-select after refresh.
        """
        option_list = self.query_one("#session-list", OptionList)
        option_list.clear_options()
        self._sessions = list_sessions(self.session_dir)

        if not self._sessions:
            # Show empty state message
            option_list.add_option(Option("No saved sessions found", id="empty", disabled=True))
            option_list.add_option(
                Option("Press Ctrl+S to save a session", id="hint", disabled=True)
            )
        else:
            select_index = 0
            for i, session in enumerate(self._sessions):
                # Track index to re-select after refresh
                if select_id and session.id == select_id:
                    select_index = i

                # Format: "Name                    Jan 9   3 msg"
                # Truncate name to fit
                name = session.name[:35]
                if len(session.name) > 35:
                    name = name[:32] + "..."

                # Extract date from updated_at (ISO format)
                date_str = session.updated_at[:10]  # YYYY-MM-DD

                label = f"{name:<38} {date_str}  {session.message_count} msg"
                option_list.add_option(Option(label, id=session.id))

            # Re-select the session if specified
            if self._sessions:
                option_list.highlighted = select_index

        option_list.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "load-btn":
            self._load_selected()
        else:
            self.dismiss(None)

    def on_option_list_option_selected(self, _event: OptionList.OptionSelected) -> None:
        """Handle double-click or Enter on option."""
        self._load_selected()

    def _load_selected(self) -> None:
        """Load the currently selected session."""
        option_list = self.query_one("#session-list", OptionList)
        highlighted = option_list.highlighted

        if highlighted is None or not self._sessions:
            self.dismiss(None)
            return

        # Get session ID from the highlighted option
        if highlighted < len(self._sessions):
            session = self._sessions[highlighted]
            filepath = get_session_filepath(self.session_dir, session.id)
            self.dismiss(filepath)
        else:
            self.dismiss(None)

    def _get_highlighted_session(self) -> tuple[SessionMetadata, Path] | None:
        """Get the currently highlighted session and its filepath.

        Returns:
            Tuple of (SessionMetadata, Path) or None if nothing selected.
        """
        option_list = self.query_one("#session-list", OptionList)
        highlighted = option_list.highlighted

        if highlighted is None or not self._sessions:
            return None

        if highlighted < len(self._sessions):
            session = self._sessions[highlighted]
            filepath = get_session_filepath(self.session_dir, session.id)
            if filepath:
                return (session, filepath)
        return None

    def action_cancel(self) -> None:
        """Handle Escape key."""
        self.dismiss(None)

    def action_load(self) -> None:
        """Handle Enter key."""
        self._load_selected()

    def action_rename(self) -> None:
        """Handle r key - rename selected session."""
        selected = self._get_highlighted_session()
        if not selected:
            return

        session, filepath = selected
        # Push rename modal
        self.app.push_screen(
            SessionRenameModal(session.name),
            lambda new_name: self._handle_rename(filepath, session.id, new_name),
        )

    def _handle_rename(self, filepath: Path, session_id: str, new_name: str | None) -> None:
        """Handle the result of rename modal.

        Args:
            filepath: Path to the session file.
            session_id: ID of session being renamed (for re-selection).
            new_name: New name from modal, or None if cancelled.
        """
        if not new_name:
            return

        try:
            rename_session(filepath, new_name)
            # Refresh list and re-select the renamed session
            self._refresh_session_list(select_id=session_id)
        except Exception as e:
            # Show error - for now just log, could add toast
            self.app.log.error(f"Failed to rename session: {e}")

    def action_delete(self) -> None:
        """Handle d key - delete selected session."""
        selected = self._get_highlighted_session()
        if not selected:
            return

        session, filepath = selected
        # Push confirmation modal
        self.app.push_screen(
            SessionDeleteConfirmModal(session.name),
            lambda confirmed: self._handle_delete(filepath, session.id, confirmed),
        )

    def _handle_delete(self, filepath: Path, session_id: str, confirmed: bool | None) -> None:
        """Handle the result of delete confirmation modal.

        Args:
            filepath: Path to the session file.
            session_id: ID of session being deleted.
            confirmed: True if user confirmed, None if cancelled.
        """
        if not confirmed:
            return

        try:
            # Check if this is the currently loaded session
            if (
                hasattr(self.app, "_current_session")
                and self.app._current_session
                and self.app._current_session.metadata.id == session_id
            ):
                # Clear the app's current session
                self.app._current_session = None

            delete_session(filepath)
            # Refresh list (no re-selection since item is gone)
            self._refresh_session_list()
        except Exception as e:
            self.app.log.error(f"Failed to delete session: {e}")


class SessionRenameModal(ModalScreen[str | None]):
    """Modal screen for renaming a session.

    Displays an input field with the current session name.
    Returns the new name, or None if cancelled.

    Usage:
        app.push_screen(SessionRenameModal(current_name), callback)
        # callback receives str (new name) or None

    Bindings:
        Enter: Apply the new name
        Escape: Cancel and dismiss
    """

    CSS = """
    SessionRenameModal {
        align: center middle;
    }

    #rename-dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
    }

    #rename-dialog Label {
        margin-bottom: 1;
    }

    #rename-dialog Input {
        width: 100%;
        margin-bottom: 1;
    }

    #rename-buttons {
        width: 100%;
        height: auto;
        align: right middle;
    }

    #rename-buttons Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, current_name: str) -> None:
        """Initialize the rename dialog.

        Args:
            current_name: Current session name (shown in input).
        """
        super().__init__()
        self._current_name = current_name

    def compose(self) -> ComposeResult:
        """Create the dialog layout."""
        with Vertical(id="rename-dialog"):
            yield Label("Rename Session")
            yield Input(value=self._current_name, id="session-name", select_on_focus=True)
            with Container(id="rename-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Rename", variant="primary", id="rename-btn")

    def on_mount(self) -> None:
        """Focus the input when modal opens."""
        input_widget = self.query_one("#session-name", Input)
        input_widget.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "rename-btn":
            name = self.query_one("#session-name", Input).value.strip()
            self.dismiss(name if name else None)
        else:
            self.dismiss(None)

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        """Handle Enter key in input field."""
        name = self.query_one("#session-name", Input).value.strip()
        self.dismiss(name if name else None)

    def action_cancel(self) -> None:
        """Handle Escape key."""
        self.dismiss(None)


class SessionDeleteConfirmModal(ModalScreen[bool | None]):
    """Modal screen for confirming session deletion.

    Displays a confirmation prompt before deleting a session.
    Returns True if user confirms deletion, None if cancelled.

    Usage:
        app.push_screen(SessionDeleteConfirmModal(session_name), callback)
        # callback receives True (confirm) or None (cancel)

    Bindings:
        Escape: Cancel and dismiss
    """

    CSS = """
    SessionDeleteConfirmModal {
        align: center middle;
    }

    #delete-dialog {
        width: 50;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $error;
    }

    #delete-dialog Label {
        margin-bottom: 1;
    }

    #delete-title {
        text-style: bold;
    }

    #delete-warning {
        color: $text-muted;
    }

    #delete-buttons {
        width: 100%;
        height: auto;
        align: right middle;
        margin-top: 1;
    }

    #delete-buttons Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, session_name: str) -> None:
        """Initialize the delete confirmation dialog.

        Args:
            session_name: Name of session to delete (shown in prompt).
        """
        super().__init__()
        self._session_name = session_name

    def compose(self) -> ComposeResult:
        """Create the dialog layout."""
        with Vertical(id="delete-dialog"):
            yield Label("Delete Session", id="delete-title")
            yield Label(f"Delete '{self._session_name}'?")
            yield Label("This cannot be undone.", id="delete-warning")
            with Container(id="delete-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Delete", variant="error", id="delete-btn")

    def on_mount(self) -> None:
        """Focus the cancel button by default (safer)."""
        self.query_one("#cancel-btn", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "delete-btn":
            self.dismiss(True)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        """Handle Escape key."""
        self.dismiss(None)


class ModelPickerModal(ModalScreen[str | None]):
    """Modal screen for selecting a model from the endpoint.

    Fetches available models from /models endpoint and displays them.
    Returns selected model name, or None if cancelled.

    Usage:
        models = await client.models()
        app.push_screen(ModelPickerModal(models, current_model), callback)
        # callback receives str (model name) or None

    Display Format:
        "● current-model" (bullet marks current)
        "  other-model"

    Bindings:
        Enter: Select highlighted model
        Escape: Cancel and dismiss
    """

    CSS = """
    ModelPickerModal {
        align: center middle;
    }

    #model-dialog {
        width: 60;
        height: 20;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
    }

    #model-dialog Label {
        margin-bottom: 1;
    }

    #model-list {
        height: 1fr;
        margin-bottom: 1;
    }

    #model-buttons {
        width: 100%;
        height: auto;
        align: right middle;
    }

    #model-buttons Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "select", "Select Model"),
    ]

    def __init__(self, models: list[str], current_model: str) -> None:
        """Initialize the model picker.

        Args:
            models: List of available model names.
            current_model: Currently selected model (for highlighting).
        """
        super().__init__()
        self._models = models
        self._current_model = current_model

    def compose(self) -> ComposeResult:
        """Create the dialog layout."""
        with Vertical(id="model-dialog"):
            yield Label("Select Model")
            yield OptionList(id="model-list")
            with Container(id="model-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Select", variant="primary", id="select-btn")

    def on_mount(self) -> None:
        """Populate model list when modal opens."""
        option_list = self.query_one("#model-list", OptionList)

        for model in self._models:
            # Mark current model with bullet
            prefix = "●" if model == self._current_model else " "
            option_list.add_option(Option(f"{prefix} {model}", id=model))

        option_list.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "select-btn":
            self._select_model()
        else:
            self.dismiss(None)

    def on_option_list_option_selected(self, _event: OptionList.OptionSelected) -> None:
        """Handle double-click or Enter on option."""
        self._select_model()

    def _select_model(self) -> None:
        """Select the highlighted model."""
        option_list = self.query_one("#model-list", OptionList)
        highlighted = option_list.highlighted

        if highlighted is not None and highlighted < len(self._models):
            self.dismiss(self._models[highlighted])
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        """Handle Escape key."""
        self.dismiss(None)

    def action_select(self) -> None:
        """Handle Enter key."""
        self._select_model()


class UnsavedChangesModal(ModalScreen[str | None]):
    """Modal screen warning about unsaved changes.

    Displayed when user tries to load a session while having
    unsaved changes in the current conversation.

    Returns:
        "save_and_load": Save current session, then load new
        "load": Load without saving (discard current)
        None: Cancel operation

    Usage:
        app.push_screen(UnsavedChangesModal(), callback)
        # callback receives "save_and_load", "load", or None

    Bindings:
        Escape: Cancel and dismiss
    """

    CSS = """
    UnsavedChangesModal {
        align: center middle;
    }

    #unsaved-dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $warning;
    }

    #unsaved-title {
        text-style: bold;
    }

    #unsaved-message {
        margin-bottom: 1;
    }

    #unsaved-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
    }

    #unsaved-buttons Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        """Create the dialog layout."""
        with Vertical(id="unsaved-dialog"):
            yield Label("Unsaved Changes", id="unsaved-title")
            yield Label(
                "You have unsaved changes that will be lost " "if you load a different session.",
                id="unsaved-message",
            )
            with Container(id="unsaved-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Load Anyway", variant="warning", id="load-btn")
                yield Button("Save & Load", variant="primary", id="save-load-btn")

    def on_mount(self) -> None:
        """Focus the cancel button by default (safer UX)."""
        self.query_one("#cancel-btn", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "save-load-btn":
            self.dismiss("save_and_load")
        elif event.button.id == "load-btn":
            self.dismiss("load")
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        """Handle Escape key."""
        self.dismiss(None)


class ModelInputModal(ModalScreen[str | None]):
    """Modal screen for manually entering a model name.

    Used when /models endpoint is not available or returns an error.
    Shows the error message and allows manual model name entry.
    Returns entered model name, or None if cancelled.

    Usage:
        app.push_screen(ModelInputModal(current_model, error_msg), callback)
        # callback receives str (model name) or None

    Bindings:
        Enter: Apply the entered model name
        Escape: Cancel and dismiss
    """

    CSS = """
    ModelInputModal {
        align: center middle;
    }

    #model-input-dialog {
        width: 60;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
    }

    #model-input-dialog Label {
        margin-bottom: 1;
    }

    #model-input-dialog Input {
        width: 100%;
        margin-bottom: 1;
    }

    #model-input-buttons {
        width: 100%;
        height: auto;
        align: right middle;
    }

    #model-input-buttons Button {
        margin-left: 1;
    }

    #model-hint {
        color: $text-muted;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, current_model: str, error_message: str = "") -> None:
        """Initialize the model input dialog.

        Args:
            current_model: Current model name (shown as placeholder).
            error_message: Optional error message to display.
        """
        super().__init__()
        self._current_model = current_model
        self._error_message = error_message

    def compose(self) -> ComposeResult:
        """Create the dialog layout."""
        with Vertical(id="model-input-dialog"):
            yield Label("Enter Model Name")
            if self._error_message:
                yield Label(self._error_message, id="model-hint")
            else:
                yield Label(
                    "Endpoint doesn't support /models listing",
                    id="model-hint",
                )
            yield Input(
                placeholder=self._current_model,
                value=self._current_model,
                id="model-name",
            )
            with Container(id="model-input-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Apply", variant="primary", id="apply-btn")

    def on_mount(self) -> None:
        """Focus the input when modal opens."""
        self.query_one("#model-name", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks."""
        if event.button.id == "apply-btn":
            model = self.query_one("#model-name", Input).value.strip()
            self.dismiss(model if model else None)
        else:
            self.dismiss(None)

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        """Handle Enter key in input field."""
        model = self.query_one("#model-name", Input).value.strip()
        self.dismiss(model if model else None)

    def action_cancel(self) -> None:
        """Handle Escape key."""
        self.dismiss(None)
