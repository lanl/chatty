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

from chatty.core.session import SessionMetadata, get_session_filepath, list_sessions


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
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "load", "Load Session"),
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
            with Container(id="session-buttons"):
                yield Button("Cancel", variant="default", id="cancel-btn")
                yield Button("Load", variant="primary", id="load-btn")

    def on_mount(self) -> None:
        """Load sessions when modal opens."""
        option_list = self.query_one("#session-list", OptionList)
        self._sessions = list_sessions(self.session_dir)

        if not self._sessions:
            # Show empty state message
            option_list.add_option(Option("No saved sessions found", id="empty", disabled=True))
            option_list.add_option(
                Option("Press Ctrl+S to save a session", id="hint", disabled=True)
            )
        else:
            for session in self._sessions:
                # Format: "Name                    Jan 9   3 msg"
                # Truncate name to fit
                name = session.name[:35]
                if len(session.name) > 35:
                    name = name[:32] + "..."

                # Extract date from updated_at (ISO format)
                date_str = session.updated_at[:10]  # YYYY-MM-DD

                label = f"{name:<38} {date_str}  {session.message_count} msg"
                option_list.add_option(Option(label, id=session.id))

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

    def action_cancel(self) -> None:
        """Handle Escape key."""
        self.dismiss(None)

    def action_load(self) -> None:
        """Handle Enter key."""
        self._load_selected()


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
