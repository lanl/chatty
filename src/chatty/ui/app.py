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
    │   └── Input (user text entry)
    ├── StatusBar (model, tokens, connection status)
    └── Footer (Textual built-in - shows keybindings)

Data Flow
---------
    1. User types message in Input widget
    2. on_input_submitted() receives the event
    3. Message added to Conversation state
    4. RAGProvider.augment() called (NullProvider passthrough in v0.1)
    5. OpenAIClient.chat() called with streaming
    6. Tokens streamed to ChatLog via worker
    7. On completion, Conversation updated with full response
    8. StatusBar updated with token count

Keyboard Shortcuts
------------------
    Ctrl+C : Quit (clean shutdown)
    Ctrl+T : Toggle streaming mode on/off
    Ctrl+R : Regenerate last response
    Ctrl+O : Load query from file
    Ctrl+N : New session (clear history)
    Ctrl+Y : Copy last message to clipboard
    Escape : Cancel current generation

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

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, Static


class ChatLog(Static):
    """Scrollable widget to display chat messages with markdown rendering.

    This widget maintains a list of messages and renders them with role
    prefixes. Messages are formatted as markdown and updated incrementally
    during streaming.

    Attributes:
        messages: List of formatted message strings for display.

    Example:
        chat_log = ChatLog(id="chat-log")
        chat_log.add_message("user", "Hello!")
        chat_log.add_message("assistant", "Hi there!")
    """

    def __init__(self, id: str | None = None) -> None:  # noqa: A002
        """Initialize the chat log widget.

        Args:
            id: Optional DOM identifier for CSS styling and queries.
        """
        super().__init__("", id=id)
        self.messages: list[str] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat log.

        Formats the message with a role prefix and updates the display.
        Roles are mapped to human-readable names (user → "You", etc.).

        Args:
            role: Message role ("user", "assistant", "system", or custom).
            content: The message content (supports markdown).
        """
        prefix = {"user": "You", "assistant": "Assistant", "system": "System"}.get(role, role)
        self.messages.append(f"**{prefix}:** {content}")
        self.update("\n\n".join(self.messages))

    def append_to_last(self, content: str) -> None:
        """Append content to the last message (for streaming).

        Used during streaming to incrementally build the assistant's response.

        Args:
            content: Content to append to the last message.
        """
        if self.messages:
            self.messages[-1] += content
            self.update("\n\n".join(self.messages))

    def clear_messages(self) -> None:
        """Clear all messages from the chat log."""
        self.messages = []
        self.update("")


class StatusBar(Static):
    """Widget to display status information in a single line.

    Shows: connection status, model name, streaming mode, and token usage.
    Updates dynamically as state changes.

    Display Format:
        "Ready | Model: gpt-4.1 | Stream: on | 12K / 128K tokens"

    Status States:
        - Ready: Idle, waiting for user input
        - Thinking...: Waiting for LLM response
        - Streaming: Receiving tokens
        - Error: Last request failed
    """

    def __init__(self, id: str | None = None) -> None:  # noqa: A002
        """Initialize the status bar.

        Args:
            id: Optional DOM identifier for CSS styling and queries.
        """
        super().__init__("Ready | Model: gpt-4.1 | Stream: on", id=id)

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

        Args:
            status: Status text ("Ready", "Thinking...", "Error", etc.)
            model: Model name to display
            streaming: Whether streaming is enabled
            tokens: Token usage string (e.g., "12K / 128K tokens")
        """
        # TODO: Parse current state, update only changed values, rebuild string
        pass


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
        config: Application configuration (set on mount).
        client: OpenAI client instance (set on mount).
        conversation: Current conversation state (set on mount).

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
        overflow-y: auto;
        padding: 1;
        border: solid green;
    }

    #input-container {
        height: auto;
        padding: 1;
    }

    #status-bar {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }

    Input {
        width: 100%;
    }
    """

    # Keyboard bindings - each tuple is (key, action_method, description)
    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+t", "toggle_stream", "Toggle Stream"),
        ("ctrl+r", "regenerate", "Regenerate"),
        ("ctrl+o", "load_file", "Load File"),
        ("ctrl+n", "new_session", "New Session"),
        ("ctrl+y", "copy_message", "Copy"),
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, query_file: str | None = None) -> None:
        """Initialize the chat application.

        Args:
            query_file: Optional path to file containing initial query to load.
        """
        super().__init__()
        self.query_file = query_file
        self.streaming = True
        # These will be initialized in on_mount() with actual config
        # self.config = None
        # self.client = None
        # self.conversation = None

    def compose(self) -> ComposeResult:
        """Create the UI layout.

        Yields widgets in top-to-bottom order. The layout is:
        - Header: Shows app title
        - Container with ChatLog: Main scrollable area for messages
        - Vertical with Input: User text input area
        - StatusBar: Single-line status display
        - Footer: Shows available keyboard shortcuts
        """
        yield Header()
        yield Container(
            ChatLog(id="chat-log"),
            id="chat-container",
        )
        yield Vertical(
            Input(placeholder="Type your message...", id="input"),
            id="input-container",
        )
        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Handle app mount event.

        Called when the app is fully loaded and ready. Sets up:
        - Initial focus on input widget
        - Load config and create client
        - Load query from file if provided
        """
        self.query_one("#input", Input).focus()

        # TODO: Initialize config, client, and conversation
        # self.config = load_config()
        # self.client = OpenAIClient(self.config)
        # self.conversation = Conversation()

        # TODO: If query_file provided, load and submit it

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle message submission from the input widget.

        This is the main chat workflow entry point:
        1. Validate input (non-empty)
        2. Add user message to chat log
        3. Clear input for next message
        4. Call LLM (streaming or non-streaming)
        5. Display response
        6. Update conversation state

        Args:
            event: Input submission event containing the message text.
        """
        if not event.value.strip():
            return

        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.add_message("user", event.value)

        # Clear input for next message
        event.input.value = ""

        # TODO: Implement actual LLM integration
        # 1. Update status to "Thinking..."
        # 2. Add message to conversation
        # 3. Call RAGProvider.augment()
        # 4. Call client.chat() with streaming
        # 5. Stream tokens to chat_log.append_to_last()
        # 6. Update conversation with complete response
        # 7. Update status bar with token count

        chat_log.add_message("assistant", "Chat functionality not yet implemented.")

    def action_toggle_stream(self) -> None:
        """Toggle streaming mode on/off.

        When streaming is on, tokens are displayed as they arrive.
        When off, the complete response is displayed at once.
        """
        self.streaming = not self.streaming
        status = "on" if self.streaming else "off"
        self.query_one("#status-bar", StatusBar).update(
            f"Ready | Model: gpt-4.1 | Stream: {status}"
        )

    def action_regenerate(self) -> None:
        """Regenerate the last assistant response.

        Removes the last assistant message from conversation and chat log,
        then re-sends the previous user message to get a new response.
        Useful when the response was unsatisfactory.
        """
        # TODO: Implement regeneration
        # 1. Remove last assistant message from conversation
        # 2. Remove last message from chat log
        # 3. Re-send previous user message
        pass

    def action_load_file(self) -> None:
        """Load a query from a file.

        Opens a file dialog (or prompts for path) to load a query file.
        Useful for long committee-written queries that are awkward to paste.
        """
        # TODO: Implement file loading
        # 1. Prompt for file path (or use file dialog)
        # 2. Read file content
        # 3. Submit as user message
        pass

    def action_new_session(self) -> None:
        """Start a new session, clearing all history.

        Clears the chat log and resets the conversation state.
        Token count is reset. Does not change configuration.
        """
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.clear_messages()

        # TODO: Reset conversation state
        # self.conversation.clear()
        # Update status bar to show reset token count

    def action_copy_message(self) -> None:
        """Copy the last message to clipboard.

        Attempts to use pyperclip. If clipboard is unavailable
        (headless HPC node), falls back to writing to a temp file
        and displays the file path to the user.
        """
        # TODO: Implement clipboard copy with fallback
        # 1. Get last message content
        # 2. Try pyperclip.copy()
        # 3. On failure, write to /tmp/chatty-export-{uuid}.txt
        # 4. Notify user of result
        pass

    def action_cancel(self) -> None:
        """Cancel the current generation.

        Cancels any in-flight LLM request by cancelling the async task.
        Partial responses may be kept or discarded based on config.
        """
        # TODO: Implement cancellation
        # 1. Cancel the streaming worker task
        # 2. Update status to "Cancelled"
        # 3. Optionally keep partial response
        pass


def main(query_file: str | None = None) -> None:
    """Run the chat application.

    Entry point for the chatty UI. Creates and runs the Textual app.

    Args:
        query_file: Optional path to file containing initial query.
    """
    app = ChatApp(query_file=query_file)
    app.run()
