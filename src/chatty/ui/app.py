"""Textual application and widgets for chatty.

This module implements the terminal UI using Textual. The app provides a
full-screen chat interface with streaming responses, keyboard shortcuts,
and status tracking.

Widget Hierarchy
----------------
    ChatApp (main application)
    ├── Header (Textual built-in - shows app title)
    ├── Container
    │   └── ChatLog (scrollable message display with markdown)
    ├── Vertical
    │   └── TextArea (multi-line text entry)
    ├── StatusBar (model, tokens, connection status)
    └── Footer (Textual built-in - shows keybindings)

Data Flow
---------
    1. User types message in TextArea widget
    2. Ctrl+E triggers action_submit()
    3. Message added to Conversation state
    4. RAGProvider.augment() called (NullProvider passthrough in v0.1)
    5. OpenAIClient.chat() called with streaming
    6. Tokens streamed to ChatLog via worker
    7. On completion, Conversation updated with full response
    8. StatusBar updated with token count

Keyboard Shortcuts
------------------
    Ctrl+Q      : Quit (clean shutdown)
    Ctrl+E      : Send message
    Ctrl+O      : Load query from file
    Ctrl+N      : New session (clear history)
    Ctrl+R      : Regenerate last response
    Ctrl+T      : Toggle streaming mode on/off
    Enter       : Insert newline (multi-line input)
    Escape      : Cancel current generation

Integration Points
------------------
    - OpenAIClient: Async chat completions with retry
    - Conversation: Message history and token tracking
    - RAGProvider: Context augmentation (NullProvider for v0.1)
    - Config: Model settings, streaming default, system prompt

Key Design Decisions
--------------------
    - Single-file UI for v0.1 simplicity (will split if >800 lines)
    - CSS inline in class (will extract to .tcss if >100 lines)
    - Async streaming via Textual workers to keep UI responsive
    - Cancellation via asyncio task cancellation
    - Error display inline as styled message cards
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import cast

from rich.markdown import Markdown as RichMarkdown
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option
from textual.worker import Worker

from chatty.client.openai_client import (
    ChattyClientError,
    OpenAIClient,
)
from chatty.config import ConfigWithSources, find_config_path, load_config
from chatty.core.conversation import Conversation
from chatty.core.session import (
    Session,
    SessionMetadata,
    generate_session_name,
    get_session_filepath,
    list_sessions,
    load_session,
    save_session,
)
from chatty.core.transcript import TranscriptLogger
from chatty.rag import RAGMetadata, RAGProvider, get_provider


class FileInputModal(ModalScreen[str | None]):
    """Modal screen for entering a file path.

    Returns the file path string if submitted, None if cancelled.
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


class ModelPickerModal(ModalScreen[str | None]):
    """Modal screen for selecting a model from the endpoint.

    Fetches available models from /models endpoint and displays them.
    Returns selected model name, or None if cancelled.
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

    Used when /models endpoint is not available.
    Returns entered model name, or None if cancelled.
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


class SessionBrowserModal(ModalScreen[Path | None]):
    """Modal screen for browsing and selecting saved sessions.

    Displays a list of saved sessions with name, date, and message count.
    Returns the selected session file path, or None if cancelled.
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


class ChatInput(TextArea):
    """Custom TextArea for chat input with Ctrl+E submit.

    Inherits standard TextArea bindings (cursor movement, copy/paste, etc.)
    for normal text editing. Multi-line input via Enter key.

    Ctrl+E overrides TextArea's "end of line" binding to submit instead.
    """

    BINDINGS = [
        Binding("ctrl+e", "send", "Submit Query"),
    ]

    class Submitted(TextArea.Changed):
        """Event posted when Ctrl+E is pressed to submit."""

        pass

    def action_send(self) -> None:
        """Handle Ctrl+E to submit the message."""
        self.post_message(self.Submitted(self))


class MessageWidget(Static):
    """A single chat message with role prefix and markdown rendering.

    Each message is rendered as a separate widget, allowing individual
    styling and future enhancements (timestamps, copy buttons, etc.).

    For assistant messages, content is rendered as markdown when streaming
    is complete. During streaming, raw text is shown for performance.

    Attributes:
        role: The message role (user, assistant, system, error).
        message_content: The message text content.
        is_streaming: Whether the message is currently being streamed.
    """

    ROLE_PREFIXES = {
        "user": "You",
        "assistant": "Assistant",
        "system": "System",
        "error": "⚠ Error",
    }

    def __init__(self, role: str, content: str = "") -> None:
        """Initialize a message widget.

        Args:
            role: Message role (user, assistant, system, error).
            content: Initial message content.
        """
        super().__init__(classes=f"{role}-message")
        self.role = role
        self.message_content: str = content
        self.is_streaming: bool = False
        self._update_display()

    def _update_display(self) -> None:
        """Update the displayed text.

        Shows raw text during streaming, markdown when complete (for assistant).
        """
        prefix = self.ROLE_PREFIXES.get(self.role, self.role)

        if self.role == "assistant" and not self.is_streaming and self.message_content:
            # Assistant messages get markdown rendering when not streaming
            # Include prefix in the markdown content
            markdown_content = f"**{prefix}:**\n\n{self.message_content}"
            self.update(RichMarkdown(markdown_content))
        else:
            # User, system, error messages and streaming assistant use plain text
            self.update(f"**{prefix}:** {self.message_content}")

    def append_content(self, text: str) -> None:
        """Append text to the message content (for streaming).

        Args:
            text: Text to append.
        """
        self.message_content += text
        self._update_display()

    def set_content(self, content: str) -> None:
        """Replace the message content.

        Args:
            content: New content.
        """
        self.message_content = content
        self._update_display()

    def finish_streaming(self) -> None:
        """Mark streaming as complete and re-render with markdown."""
        self.is_streaming = False
        self._update_display()


class ChatLog(VerticalScroll):
    """Scrollable container for chat messages.

    Each message is a separate MessageWidget, allowing individual styling
    and future enhancements (timestamps, copy buttons, bookmarks).

    Methods:
        add_message: Add a new message widget.
        get_last_message: Get the last message widget (for streaming).
        clear_messages: Remove all messages.
        remove_last_message: Remove the last message (for regeneration).
    """

    def __init__(self, id: str | None = None) -> None:  # noqa: A002
        """Initialize the chat log container.

        Args:
            id: Optional DOM identifier for CSS styling.
        """
        super().__init__(id=id)

    def add_message(self, role: str, content: str = "") -> MessageWidget:
        """Add a new message to the chat log.

        Args:
            role: Message role (user, assistant, system, error).
            content: Initial message content (can be empty for streaming).

        Returns:
            The created MessageWidget (useful for streaming updates).
        """
        widget = MessageWidget(role, content)
        self.mount(widget)
        self.scroll_end(animate=False)
        return widget

    def get_last_message(self) -> MessageWidget | None:
        """Get the last message widget.

        Returns:
            The last MessageWidget, or None if no messages.
        """
        children = list(self.query(MessageWidget))
        return children[-1] if children else None

    def append_to_last(self, content: str) -> None:
        """Append content to the last message (for streaming).

        Args:
            content: Content to append.
        """
        last = self.get_last_message()
        if last:
            last.append_content(content)
            self.scroll_end(animate=False)

    def clear_messages(self) -> None:
        """Remove all messages from the chat log."""
        for widget in list(self.query(MessageWidget)):
            widget.remove()

    def remove_last_message(self) -> None:
        """Remove the last message (for regeneration)."""
        last = self.get_last_message()
        if last:
            last.remove()


class StatusBar(Static):
    """Widget to display status information in a single line.

    Shows: connection status, model name, streaming mode, and token usage.
    Updates dynamically as state changes. Includes animated spinner during
    active operations (Thinking, Streaming) with elapsed time display.

    Display Format:
        "Ready (3.2s) | Model: gpt-4.1 | Stream: on | 12K / 128K tokens"
        "⣾ Thinking... (1.5s) | Model: gpt-4.1 | Stream: on"

    Status States:
        - Ready: Idle, waiting for user input (shows response time after generation)
        - Thinking...: Waiting for LLM response (with spinner + elapsed time)
        - Streaming...: Receiving tokens (with spinner + elapsed time)
        - Error: Last request failed
        - Cancelled: User cancelled generation
    """

    # Braille spinner characters - smooth animation
    SPINNER_FRAMES = "⣾⣽⣻⢿⡿⣟⣯⣷"

    def __init__(
        self,
        id: str | None = None,  # noqa: A002
        *,
        model: str = "gpt-4.1",
        streaming: bool = True,
    ) -> None:
        """Initialize the status bar.

        Args:
            id: Optional DOM identifier for CSS styling and queries.
            model: Initial model name to display.
            streaming: Initial streaming mode.
        """
        super().__init__("", id=id)
        self._status = "Ready"
        self._model = model
        self._streaming = streaming
        self._tokens = ""
        self._spinner_index = 0
        self._spinner_timer: object | None = None
        self._start_time: float | None = None
        self._last_response_time: float | None = None
        self._rebuild_display()

    def _format_elapsed(self, seconds: float) -> str:
        """Format elapsed time as a human-readable string.

        Args:
            seconds: Elapsed time in seconds.

        Returns:
            Formatted string like "1.5s" or "1m 23s".
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        remaining = seconds % 60
        return f"{minutes}m {remaining:.0f}s"

    def _rebuild_display(self) -> None:
        """Rebuild the status bar text from current state."""
        # Add spinner prefix and elapsed time for active states
        if self._status in ("Thinking...", "Streaming..."):
            spinner_char = self.SPINNER_FRAMES[self._spinner_index]
            elapsed = time.monotonic() - self._start_time if self._start_time else 0
            status_display = f"{spinner_char} {self._status} ({self._format_elapsed(elapsed)})"
        elif self._status == "Ready" and self._last_response_time is not None:
            # Show response time after generation completes
            status_display = f"Ready ({self._format_elapsed(self._last_response_time)})"
        else:
            status_display = self._status

        parts = [status_display, f"Model: {self._model}"]
        stream_str = "on" if self._streaming else "off"
        parts.append(f"Stream: {stream_str}")
        if self._tokens:
            parts.append(self._tokens)
        self.update(" | ".join(parts))

    def _advance_spinner(self) -> None:
        """Advance spinner to next frame and update elapsed time."""
        self._spinner_index = (self._spinner_index + 1) % len(self.SPINNER_FRAMES)
        self._rebuild_display()

    def _start_spinner(self) -> None:
        """Start the spinner animation and elapsed timer."""
        if self._spinner_timer is None:
            self._spinner_index = 0
            self._start_time = time.monotonic()
            self._spinner_timer = self.set_interval(0.1, self._advance_spinner)

    def _stop_spinner(self) -> None:
        """Stop the spinner animation and record response time."""
        if self._spinner_timer is not None:
            # Calculate final response time
            if self._start_time is not None:
                self._last_response_time = time.monotonic() - self._start_time
            # Remove the timer by calling its stop method
            timer = self._spinner_timer
            self._spinner_timer = None
            self._start_time = None
            if hasattr(timer, "stop"):
                timer.stop()

    def update_status(
        self,
        *,
        status: str | None = None,
        model: str | None = None,
        streaming: bool | None = None,
        tokens: str | None = None,
    ) -> None:
        """Update the status bar display.

        Only provided values are updated; others retain their current state.
        Automatically starts/stops spinner for active states.

        Args:
            status: Status text ("Ready", "Thinking...", "Error", etc.)
            model: Model name to display
            streaming: Whether streaming is enabled
            tokens: Token usage string (e.g., "12K / 128K tokens")
        """
        if status is not None:
            old_status = self._status
            self._status = status

            # Start/stop spinner based on status
            if status in ("Thinking...", "Streaming..."):
                if old_status not in ("Thinking...", "Streaming..."):
                    # Only start spinner if not already running
                    self._start_spinner()
            elif old_status in ("Thinking...", "Streaming..."):
                self._stop_spinner()

        if model is not None:
            self._model = model
        if streaming is not None:
            self._streaming = streaming
        if tokens is not None:
            self._tokens = tokens
        self._rebuild_display()


class ChatApp(App[None]):
    """Chatty terminal UI application.

    The main Textual application class that coordinates the UI components
    and handles user interactions. Manages the chat workflow:

    1. Accept user input
    2. Augment with RAG context (if enabled)
    3. Send to LLM
    4. Stream response to display
    5. Track conversation state

    Attributes:
        query_file: Optional path to file containing initial query.
        streaming: Whether to stream responses (can be toggled).
        config_with_sources: Full configuration with source tracking.
        config: Application configuration.
        client: OpenAI client instance.
        conversation: Current conversation state.
        current_worker: Currently running async worker (for cancellation).

    Class Attributes:
        TITLE: Application title shown in header.
        CSS: Inline stylesheet for widget layout.
        BINDINGS: Keyboard shortcut definitions.
    """

    TITLE = "chatty"

    # Inline CSS - defines the grid layout and widget styling
    # Layout: Header at top, scrollable chat log, input at bottom, status bar, footer
    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: 1fr auto auto;
    }

    #chat-log {
        height: 100%;
        padding: 1;
    }

    MessageWidget {
        margin-bottom: 1;
    }

    /* User messages: dimmed - your input is secondary focus */
    .user-message {
        color: $text-muted;
    }

    /* Assistant messages: default bright - main content */
    .assistant-message {
        color: $text;
    }

    /* System messages: cyan/blue - informational */
    .system-message {
        color: $accent;
    }

    /* Error messages: red - attention needed */
    .error-message {
        color: $error;
        background: $error 10%;
        padding: 1;
    }

    #input-container {
        height: auto;
        max-height: 10;
        padding: 1;
        border-top: solid $primary;
    }

    #status-bar {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }

    #input {
        width: 100%;
        height: auto;
        min-height: 3;
        max-height: 8;
    }
    """

    # Keyboard bindings - order determines display in footer
    # Note: ChatInput's ctrl+e shows first (focused widget), then quit has priority
    # Hidden bindings (show=False) are functional but not shown in footer
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+y", "copy", "Copy", priority=True),
        Binding("ctrl+s", "save", "Save Session"),
        Binding("ctrl+l", "browse_sessions", "Load Session"),
        Binding("ctrl+n", "new_session", "New Session"),
        Binding("ctrl+o", "load_file", "Load File"),
        Binding("escape", "cancel", "Interrupt"),
        # Power user shortcuts (visible in footer but may be truncated on small terminals)
        Binding("ctrl+r", "regenerate", "Regenerate"),
        Binding("ctrl+g", "pick_model", "Models"),
        Binding("ctrl+t", "toggle_stream", "Toggle Stream", show=False),
    ]

    def __init__(
        self,
        query_file: str | None = None,
        session_file: Path | None = None,
        config_with_sources: ConfigWithSources | None = None,
    ) -> None:
        """Initialize the chat application.

        Args:
            query_file: Optional path to file containing initial query to load.
            session_file: Optional path to saved session file to restore.
            config_with_sources: Pre-loaded configuration (loads default if None).
        """
        super().__init__()
        self.query_file = query_file
        self.session_file = session_file
        self.config_with_sources = config_with_sources or load_config()
        self.config = self.config_with_sources.config
        self.streaming = self.config.stream
        self.client: OpenAIClient | None = None
        self.conversation: Conversation | None = None
        self.current_worker: Worker[None] | None = None
        self.transcript: TranscriptLogger = TranscriptLogger(self.config)
        self.rag_provider: RAGProvider = get_provider(self.config)
        self.last_rag_metadata: RAGMetadata | None = None  # For future citation display
        self._pending_user_text: str | None = None  # User message awaiting LLM response
        self._current_session: Session | None = None  # For save/load functionality
        self._current_model: str = self.config.model  # Runtime model (can be changed)

    def compose(self) -> ComposeResult:
        """Create the UI layout.

        Yields widgets in top-to-bottom order. The layout is:
        - Header: Shows app title
        - ChatLog: Scrollable area for messages (extends VerticalScroll)
        - Container with ChatInput: User text input area (multi-line)
        - StatusBar: Single-line status display
        - Footer: Shows available keyboard shortcuts
        """
        yield Header()
        yield ChatLog(id="chat-log")
        yield Container(
            ChatInput(id="input"),
            id="input-container",
        )
        yield StatusBar(id="status-bar", model=self.config.model, streaming=self.streaming)
        yield Footer()

    def on_mount(self) -> None:
        """Handle app mount event.

        Called when the app is fully loaded and ready. Sets up:
        - Initial focus on input widget
        - Check for missing configuration (show warnings)
        - Create OpenAI client
        - Initialize conversation state
        - Start transcript logging if enabled
        - Add system prompt if configured
        - Load query from file if provided
        """
        self.query_one("#input", ChatInput).focus()
        chat_log = self.query_one("#chat-log", ChatLog)

        # Check for missing configuration and warn user
        self._check_startup_config(chat_log)

        # Initialize client and conversation
        self.client = OpenAIClient(self.config)
        self.conversation = Conversation()

        # Start transcript logging
        transcript_file = self.transcript.start_session()
        if transcript_file:
            chat_log.add_message("system", f"Transcript: {transcript_file}")

        # Add system prompt if configured
        if self.config.system_prompt:
            self.conversation.add_system_message(self.config.system_prompt)
            self.transcript.log_message("system", self.config.system_prompt)

        # Update status bar with actual model
        self.query_one("#status-bar", StatusBar).update_status(model=self.config.model)

        # Load session from file if provided (takes precedence over query file)
        if self.session_file:
            self._load_session_file(chat_log)
        elif self.query_file:
            self._load_and_submit_query_file()

    def _check_startup_config(self, chat_log: ChatLog) -> None:
        """Check configuration at startup and show warnings for issues.

        Args:
            chat_log: The chat log widget to display warnings.
        """
        config_path = find_config_path()
        warnings = []

        # Check if no config file found
        if config_path is None:
            warnings.append(
                "⚠ No config file found.\n\n"
                "Searched:\n"
                "  ./chatty.toml\n"
                "  ~/.config/chatty/config.toml\n\n"
                "To fix:\n"
                "  • Run from repo root: cd ~/Code/chatty\n"
                "  • Or set: export CHATTY_CONFIG='/path/to/chatty.toml'\n"
                "  • Or create: ~/.config/chatty/config.toml\n\n"
                "Run 'chatty doctor' to diagnose."
            )

        # Check if base_url is empty
        if not self.config.base_url:
            warnings.append(
                "⚠ No base_url configured. Chat will fail.\n\n"
                "To fix, add to your config file:\n"
                '  base_url = "http://localhost:1234/v1"'
            )

        # Show warnings
        for warning in warnings:
            chat_log.add_message("error", warning)

    def _load_session_file(self, chat_log: ChatLog) -> None:
        """Load a saved session from file.

        Args:
            chat_log: The chat log widget to display messages.
        """
        if not self.session_file:
            return

        try:
            session = load_session(self.session_file)
            self._current_session = session

            # Restore conversation state
            if self.conversation:
                self.conversation.clear()
                for msg in session.messages:
                    if msg.role == "system":
                        self.conversation.add_system_message(msg.content)
                    elif msg.role == "user":
                        self.conversation.add_user_message(msg.content)
                    elif msg.role == "assistant":
                        self.conversation.add_assistant_message(msg.content)

            # Restore chat log display (skip system messages)
            for msg in session.messages:
                if msg.role != "system":
                    chat_log.add_message(msg.role, msg.content)

            # Show confirmation
            chat_log.add_message(
                "system",
                f"Loaded session: {session.metadata.name}\n"
                f"Messages: {session.metadata.message_count}",
            )

        except Exception as e:
            chat_log.add_message("error", f"Failed to load session: {e}")

    def _load_and_submit_query_file(self) -> None:
        """Load query from file and submit it."""
        if not self.query_file:
            return
        try:
            with open(self.query_file) as f:
                content = f.read().strip()
            if content:
                # Set input text and trigger submit
                input_widget = self.query_one("#input", ChatInput)
                input_widget.text = content
                self.action_submit()
        except Exception as e:
            chat_log = self.query_one("#chat-log", ChatLog)
            chat_log.add_message("error", f"Failed to load query file: {e}")

    def on_chat_input_submitted(self, _event: ChatInput.Submitted) -> None:
        """Handle Enter key from ChatInput widget."""
        self.action_submit()

    def action_submit(self) -> None:
        """Submit the current message (Ctrl+E).

        This is the main chat workflow entry point:
        1. Validate input (non-empty)
        2. Add user message to chat log
        3. Clear input for next message
        4. Store pending user text for RAG provider
        5. Start async worker for LLM call
        6. Worker streams response to chat log
        """
        input_widget = self.query_one("#input", ChatInput)
        user_message = input_widget.text.strip()

        if not user_message:
            return

        chat_log = self.query_one("#chat-log", ChatLog)

        # Display user message
        chat_log.add_message("user", user_message)

        # Clear input for next message
        input_widget.text = ""

        # Store pending user text - will be added to conversation after response
        # This allows RAGProvider.augment() to receive conversation history
        # separately from the new user message for query rewriting (v0.3+)
        self._pending_user_text = user_message
        self.transcript.log_message("user", user_message)

        # Start async worker for LLM call
        # Pass method reference (not called) — Textual invokes it
        self.current_worker = self.run_worker(
            self._send_message,
            exclusive=True,
            name="send_message",
        )

    async def _send_message(self) -> None:  # noqa: C901
        """Worker function for async LLM call.

        Handles the complete workflow:
        1. Update status to "Thinking..."
        2. Call RAGProvider.augment() to get messages (with potential context)
        3. Call client.chat() with streaming or non-streaming
        4. Stream tokens to chat log (if streaming)
        5. Update conversation with user message and assistant response
        6. Update status bar with token count
        7. Handle errors gracefully
        """
        if not self.client or not self.conversation:
            return

        user_text = self._pending_user_text
        if not user_text:
            return

        chat_log = self.query_one("#chat-log", ChatLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        status_bar.update_status(status="Thinking...")

        try:
            # Use RAG provider to augment messages with context
            # NullProvider passes through unchanged; LitkitProvider (v0.3+) adds context
            messages, rag_metadata = await self.rag_provider.augment(self.conversation, user_text)
            self.last_rag_metadata = rag_metadata

            if self.streaming:
                # Streaming mode
                status_bar.update_status(status="Streaming...")

                # Add empty assistant message for streaming (marked as streaming)
                msg_widget = chat_log.add_message("assistant", "")
                msg_widget.is_streaming = True
                response_content = ""

                result = await self.client.chat(messages, stream=True)
                stream = cast(AsyncIterator[str], result)

                async for token in stream:
                    chat_log.append_to_last(token)
                    response_content += token

                # Finish streaming - re-render with markdown
                msg_widget.finish_streaming()

                # Update conversation with user message and assistant response
                # User message is added here (after success) to keep conversation
                # in sync with what was actually sent to the API
                self.conversation.add_user_message(user_text)
                self.conversation.add_assistant_message(response_content)

                # Log to transcript with response time
                response_time = status_bar._last_response_time
                self.transcript.log_message(
                    "assistant",
                    response_content,
                    model=self.config.model,
                    response_time_s=response_time,
                    tokens=self.conversation.server_reported_tokens,
                )

            else:
                # Non-streaming mode - message renders with markdown immediately
                from chatty.client.openai_client import AssistantMessage

                result = await self.client.chat(messages, stream=False)
                response = cast(AssistantMessage, result)

                chat_log.add_message("assistant", response.content)

                # Update conversation with user message and assistant response
                self.conversation.add_user_message(user_text)
                self.conversation.add_assistant_message(response.content, response.usage)

                # Log to transcript
                self.transcript.log_message(
                    "assistant",
                    response.content,
                    model=self.config.model,
                    tokens=response.usage.get("total_tokens") if response.usage else None,
                )

            # Update status with token count
            status_bar.update_status(
                status="Ready",
                tokens=self.conversation.get_token_display(),
            )

        except ChattyClientError as e:
            chat_log.add_message("error", str(e))
            status_bar.update_status(status="Error")
            self.transcript.log_message("error", str(e))

        except Exception as e:
            chat_log.add_message("error", f"Unexpected error: {e}")
            status_bar.update_status(status="Error")
            self.transcript.log_message("error", f"Unexpected error: {e}")

        finally:
            self.current_worker = None

    def action_toggle_stream(self) -> None:
        """Toggle streaming mode on/off.

        When streaming is on, tokens are displayed as they arrive.
        When off, the complete response is displayed at once.
        """
        self.streaming = not self.streaming
        self.query_one("#status-bar", StatusBar).update_status(streaming=self.streaming)

    def action_regenerate(self) -> None:
        """Regenerate the last assistant response.

        Removes the last assistant message from conversation and chat log,
        then re-sends the previous user message to get a new response.
        Useful when the response was unsatisfactory.
        """
        if not self.conversation or len(self.conversation.messages) < 2:
            return

        # Check if the last message is from assistant
        if self.conversation.messages[-1].role != "assistant":
            return

        # Get the user message that preceded the assistant response
        # (it's the second-to-last message)
        if self.conversation.messages[-2].role != "user":
            return

        user_text = self.conversation.messages[-2].content

        # Remove the last assistant message AND the user message from conversation
        # (they will be re-added after successful response)
        self.conversation.messages.pop()  # Remove assistant
        self.conversation.messages.pop()  # Remove user

        # Remove assistant message from chat log display
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.remove_last_message()

        # Set pending user text for _send_message
        self._pending_user_text = user_text

        # Re-send
        self.current_worker = self.run_worker(
            self._send_message,
            exclusive=True,
            name="regenerate",
        )

    def action_browse_sessions(self) -> None:
        """Browse and load saved sessions (Ctrl+L).

        Opens a modal dialog showing all saved sessions.
        User can select a session to load.
        """
        session_dir = self.config.get_session_path()
        self.push_screen(SessionBrowserModal(session_dir), self._handle_session_load)

    def _handle_session_load(self, filepath: Path | None) -> None:
        """Handle the session file selected from browser.

        Args:
            filepath: Path to session file, or None if cancelled.
        """
        if not filepath:
            return

        chat_log = self.query_one("#chat-log", ChatLog)

        try:
            session = load_session(filepath)

            # Clear current state
            chat_log.clear_messages()
            if self.conversation:
                self.conversation.clear()

            self._current_session = session

            # Restore conversation state
            if self.conversation:
                for msg in session.messages:
                    if msg.role == "system":
                        self.conversation.add_system_message(msg.content)
                    elif msg.role == "user":
                        self.conversation.add_user_message(msg.content)
                    elif msg.role == "assistant":
                        self.conversation.add_assistant_message(msg.content)

            # Restore chat log display (skip system messages)
            for msg in session.messages:
                if msg.role != "system":
                    chat_log.add_message(msg.role, msg.content)

            # Show confirmation
            chat_log.add_message(
                "system",
                f"Loaded session: {session.metadata.name}\n"
                f"Messages: {session.metadata.message_count}",
            )

            # Update status bar with token count
            if self.conversation:
                self.query_one("#status-bar", StatusBar).update_status(
                    status="Ready",
                    tokens=self.conversation.get_token_display(),
                )

        except Exception as e:
            chat_log.add_message("error", f"Failed to load session: {e}")

    def action_load_file(self) -> None:
        """Load a query from a file.

        Opens a modal dialog for the user to enter a file path.
        The file content is loaded into the input area (not auto-submitted).
        """
        self.push_screen(FileInputModal(), self._handle_file_path)

    def _handle_file_path(self, path: str | None) -> None:
        """Handle the file path returned from the modal.

        Args:
            path: File path entered by user, or None if cancelled.
        """
        if not path:
            return

        chat_log = self.query_one("#chat-log", ChatLog)

        try:
            with open(path) as f:
                content = f.read().strip()

            if content:
                # Load content into input area (don't auto-submit)
                input_widget = self.query_one("#input", ChatInput)
                input_widget.text = content
                input_widget.focus()
                chat_log.add_message("system", f"Loaded query from: {path}")
            else:
                chat_log.add_message("error", f"File is empty: {path}")

        except FileNotFoundError:
            chat_log.add_message("error", f"File not found: {path}")
        except PermissionError:
            chat_log.add_message("error", f"Permission denied: {path}")
        except Exception as e:
            chat_log.add_message("error", f"Failed to load file: {e}")

    def action_save(self) -> None:
        """Save the current session to a file (Ctrl+S).

        Creates or updates a session file with the current conversation.
        Auto-generates a name from the first user message if needed.
        """
        if not self.conversation or not self.conversation.messages:
            chat_log = self.query_one("#chat-log", ChatLog)
            chat_log.add_message("system", "Nothing to save — conversation is empty.")
            return

        chat_log = self.query_one("#chat-log", ChatLog)

        # Create or update session
        if self._current_session is None:
            # Generate name from first user message
            name = generate_session_name(self.conversation.messages)
            metadata = SessionMetadata.create(
                name=name,
                model=self.config.model,
                message_count=len(self.conversation.messages),
            )
            self._current_session = Session(
                metadata=metadata,
                system_prompt=self.config.system_prompt,
                messages=list(self.conversation.messages),
            )
        else:
            # Update existing session
            self._current_session.metadata.message_count = len(self.conversation.messages)
            self._current_session.messages = list(self.conversation.messages)

        # Save to file
        try:
            session_dir = self.config.get_session_path()
            filepath = save_session(self._current_session, session_dir)
            chat_log.add_message(
                "system",
                f"Session saved: {self._current_session.metadata.name}\n" f"File: {filepath}",
            )
        except Exception as e:
            chat_log.add_message("error", f"Failed to save session: {e}")

    def action_new_session(self) -> None:
        """Start a new session, clearing all history.

        Clears the chat log and resets the conversation state.
        Token count is reset. Does not change configuration.
        """
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.clear_messages()

        if self.conversation:
            self.conversation.clear()
            # Re-add system prompt if configured
            if self.config.system_prompt:
                self.conversation.add_system_message(self.config.system_prompt)

        # Reset session tracking
        self._current_session = None

        self.query_one("#status-bar", StatusBar).update_status(
            status="Ready",
            tokens="",
        )

    def action_cancel(self) -> None:
        """Cancel the current generation.

        Cancels any in-flight LLM request by cancelling the async task.
        Partial responses may be kept or discarded based on config.
        """
        if self.current_worker and self.current_worker.is_running:
            self.current_worker.cancel()
            self.query_one("#status-bar", StatusBar).update_status(status="Cancelled")
            self.current_worker = None

    def action_copy(self) -> None:
        """Copy the last assistant response (Ctrl+C).

        Copies to system clipboard. If clipboard is unavailable (headless HPC),
        falls back to writing to configured copy_fallback_path.
        """
        # Find last assistant message
        if not self.conversation or not self.conversation.messages:
            self.notify("Nothing to copy — no messages yet.", severity="warning")
            return

        # Get last assistant message
        last_assistant = None
        for msg in reversed(self.conversation.messages):
            if msg.role == "assistant":
                last_assistant = msg.content
                break

        if not last_assistant:
            self.notify("No assistant response to copy.", severity="warning")
            return

        # Try clipboard first, fall back to file
        result = self._copy_to_clipboard(last_assistant)
        if "Clipboard unavailable" in result:
            self.notify(result, severity="warning", timeout=5)
        else:
            self.notify(result, severity="information", timeout=2)

    def _copy_to_clipboard(self, text: str) -> str:
        """Copy text to clipboard with fallback to file.

        Args:
            text: Text to copy.

        Returns:
            Status message describing what happened.
        """
        try:
            import pyperclip

            pyperclip.copy(text)
            return "Copied to clipboard"
        except Exception:
            # Clipboard unavailable — fall back to file
            return self._write_copy_file(text)

    def _write_copy_file(self, text: str) -> str:
        """Write text to fallback copy file.

        Args:
            text: Text to write.

        Returns:
            Status message with file path.
        """
        try:
            copy_dir = self.config.get_copy_fallback_path()
            copy_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            filepath = copy_dir / f"copy-{timestamp}.txt"

            filepath.write_text(text)
            return f"Clipboard unavailable. Saved to {filepath}"
        except Exception as e:
            return f"Failed to copy: {e}"

    def action_pick_model(self) -> None:
        """Open the model picker (Ctrl+G).

        Fetches available models from endpoint and shows picker.
        Falls back to manual input if /models not supported.
        """
        # Run model fetch in worker to avoid blocking UI
        self.run_worker(self._fetch_and_show_models, exclusive=False, name="fetch_models")  # type: ignore[arg-type]

    async def _fetch_and_show_models(self) -> None:
        """Fetch models from endpoint and show picker modal."""
        if not self.client:
            return

        chat_log = self.query_one("#chat-log", ChatLog)

        try:
            models = await self.client.models()
            if models:
                # Show picker with available models
                self.push_screen(
                    ModelPickerModal(models, self._current_model),
                    self._handle_model_selection,
                )
            else:
                # Empty list - show manual input
                self.push_screen(
                    ModelInputModal(self._current_model, "No models returned"),
                    self._handle_model_selection,
                )
        except ChattyClientError as e:
            # /models not supported - show manual input
            self.push_screen(
                ModelInputModal(self._current_model, str(e)[:50]),
                self._handle_model_selection,
            )
        except Exception as e:
            chat_log.add_message("error", f"Failed to fetch models: {e}")

    def _handle_model_selection(self, model: str | None) -> None:
        """Handle the model selected from picker or input.

        Args:
            model: Selected model name, or None if cancelled.
        """
        if not model or model == self._current_model:
            return

        # Update runtime model
        old_model = self._current_model
        self._current_model = model

        # Also update config.model so client uses new model
        # This is a runtime-only change; config file is not modified
        self.config.model = model

        # Update status bar
        self.query_one("#status-bar", StatusBar).update_status(model=model)

        # Show confirmation
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.add_message("system", f"Switched model: {old_model} → {model}")

    async def on_unmount(self) -> None:
        """Clean up when app is closing."""
        self.transcript.close()
        if self.client:
            await self.client.close()


def main(
    query_file: str | None = None,
    session_file: Path | None = None,
    config_with_sources: ConfigWithSources | None = None,
) -> None:
    """Run the chat application.

    Entry point for the chatty UI. Creates and runs the Textual app.

    Args:
        query_file: Optional path to file containing initial query.
        session_file: Optional path to saved session file to restore.
        config_with_sources: Pre-loaded configuration (loads default if None).
    """
    app = ChatApp(
        query_file=query_file,
        session_file=session_file,
        config_with_sources=config_with_sources,
    )
    app.run()
