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
    Ctrl+E      : Send message (E for execute/enter)
    Ctrl+O      : Load query from file
    Ctrl+N      : New session (clear history)
    Ctrl+R      : Regenerate last response
    Ctrl+T      : Toggle streaming mode on/off
    Ctrl+Y      : Copy last message to clipboard
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

import tempfile
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, cast

from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.events import Key
from textual.widgets import Footer, Header, Static, TextArea
from textual.worker import Worker

from chatty.client.openai_client import (
    ChattyClientError,
    Message,
    OpenAIClient,
)
from chatty.config import ConfigWithSources, load_config
from chatty.core.conversation import Conversation

if TYPE_CHECKING:
    pass


class ChatInput(TextArea):
    """Custom TextArea that submits on Ctrl+E.

    Overrides the default TextArea key handling to intercept Ctrl+E
    for submitting. Enter inserts newlines for multi-line input.

    Note: Ctrl+Enter is often captured by terminal emulators (e.g., iTerm2
    opens "New Tab"). Ctrl+E is more reliable across terminals.

    Default TextArea bindings (Ctrl+E=End, Ctrl+C=Copy) are cleared to
    avoid conflicts with app-level shortcuts.
    """

    # Clear TextArea's default bindings to avoid conflicts
    BINDINGS = []

    class Submitted(TextArea.Changed):
        """Event posted when Enter is pressed to submit input."""

        pass

    async def _on_key(self, event: Key) -> None:
        """Intercept key events for submit behavior.

        Ctrl+Enter is captured by many terminal emulators (e.g., iTerm2).
        Ctrl+E is more reliable and mnemonic (E for execute/enter).

        Args:
            event: The key event to handle.
        """
        # Ctrl+E submits (Ctrl+Enter often captured by terminal)
        if event.key == "ctrl+e":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self))
            return

        # Enter and Shift+Enter: let TextArea insert newline
        await super()._on_key(event)


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
            role: Message role ("user", "assistant", "system", "error", or custom).
            content: The message content (supports markdown).
        """
        prefix_map = {
            "user": "You",
            "assistant": "Assistant",
            "system": "System",
            "error": "⚠ Error",
        }
        prefix = prefix_map.get(role, role)
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

    def remove_last_message(self) -> None:
        """Remove the last message (for regeneration)."""
        if self.messages:
            self.messages.pop()
            self.update("\n\n".join(self.messages))


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
        - Cancelled: User cancelled generation
    """

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
        self._rebuild_display()

    def _rebuild_display(self) -> None:
        """Rebuild the status bar text from current state."""
        parts = [self._status, f"Model: {self._model}"]
        stream_str = "on" if self._streaming else "off"
        parts.append(f"Stream: {stream_str}")
        if self._tokens:
            parts.append(self._tokens)
        self.update(" | ".join(parts))

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
        if status is not None:
            self._status = status
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

    #chat-container {
        height: 100%;
    }

    #chat-log {
        height: auto;
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

    # Keyboard bindings - each tuple is (key, action_method, description)
    # Order determines display in footer (most important first)
    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+e", "submit", "Send"),
        ("ctrl+o", "load_file", "Load"),
        ("ctrl+n", "new_session", "New"),
        ("ctrl+r", "regenerate", "Regen"),
        ("ctrl+t", "toggle_stream", "Stream"),
        ("ctrl+y", "copy_message", "Copy"),
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        query_file: str | None = None,
        config_with_sources: ConfigWithSources | None = None,
    ) -> None:
        """Initialize the chat application.

        Args:
            query_file: Optional path to file containing initial query to load.
            config_with_sources: Pre-loaded configuration (loads default if None).
        """
        super().__init__()
        self.query_file = query_file
        self.config_with_sources = config_with_sources or load_config()
        self.config = self.config_with_sources.config
        self.streaming = self.config.stream
        self.client: OpenAIClient | None = None
        self.conversation: Conversation | None = None
        self.current_worker: Worker[None] | None = None

    def compose(self) -> ComposeResult:
        """Create the UI layout.

        Yields widgets in top-to-bottom order. The layout is:
        - Header: Shows app title
        - VerticalScroll with ChatLog: Main scrollable area for messages
        - Container with ChatInput: User text input area (multi-line)
        - StatusBar: Single-line status display
        - Footer: Shows available keyboard shortcuts
        """
        yield Header()
        yield VerticalScroll(
            ChatLog(id="chat-log"),
            id="chat-container",
        )
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
        - Create OpenAI client
        - Initialize conversation state
        - Add system prompt if configured
        - Load query from file if provided
        """
        self.query_one("#input", ChatInput).focus()

        # Initialize client and conversation
        self.client = OpenAIClient(self.config)
        self.conversation = Conversation()

        # Add system prompt if configured
        if self.config.system_prompt:
            self.conversation.add_system_message(self.config.system_prompt)

        # Update status bar with actual model
        self.query_one("#status-bar", StatusBar).update_status(model=self.config.model)

        # Load query from file if provided
        if self.query_file:
            self._load_and_submit_query_file()

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
        4. Start async worker for LLM call
        5. Worker streams response to chat log
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

        # Add to conversation state
        if self.conversation:
            self.conversation.add_user_message(user_message)

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
        2. Call client.chat() with streaming or non-streaming
        3. Stream tokens to chat log (if streaming)
        4. Update conversation with complete response
        5. Update status bar with token count
        6. Handle errors gracefully
        """
        if not self.client or not self.conversation:
            return

        chat_log = self.query_one("#chat-log", ChatLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        status_bar.update_status(status="Thinking...")

        try:
            # Get messages for API
            messages = [Message(m.role, m.content) for m in self.conversation.messages]

            if self.streaming:
                # Streaming mode
                status_bar.update_status(status="Streaming...")

                # Add empty assistant message for streaming
                chat_log.add_message("assistant", "")
                response_content = ""

                result = await self.client.chat(messages, stream=True)
                stream = cast(AsyncIterator[str], result)

                async for token in stream:
                    chat_log.append_to_last(token)
                    response_content += token
                    # Scroll to bottom
                    container = self.query_one("#chat-container", VerticalScroll)
                    container.scroll_end(animate=False)

                # Update conversation with complete response
                self.conversation.add_assistant_message(response_content)

            else:
                # Non-streaming mode
                from chatty.client.openai_client import AssistantMessage

                result = await self.client.chat(messages, stream=False)
                response = cast(AssistantMessage, result)

                chat_log.add_message("assistant", response.content)
                self.conversation.add_assistant_message(response.content, response.usage)

                # Scroll to bottom
                container = self.query_one("#chat-container", VerticalScroll)
                container.scroll_end(animate=False)

            # Update status with token count
            status_bar.update_status(
                status="Ready",
                tokens=self.conversation.get_token_display(),
            )

        except ChattyClientError as e:
            chat_log.add_message("error", str(e))
            status_bar.update_status(status="Error")

        except Exception as e:
            chat_log.add_message("error", f"Unexpected error: {e}")
            status_bar.update_status(status="Error")

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

        # Remove the last assistant message from conversation
        self.conversation.messages.pop()

        # Remove from chat log display
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.remove_last_message()

        # Re-send (the last user message is still in conversation)
        self.current_worker = self.run_worker(
            self._send_message,
            exclusive=True,
            name="regenerate",
        )

    def action_load_file(self) -> None:
        """Load a query from a file.

        In v0.1, displays a message directing user to use --query-file flag.
        Future versions may implement an interactive file picker.
        """
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.add_message(
            "system",
            "To load a query from file, restart with: `chatty chat --query-file PATH`",
        )

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

        self.query_one("#status-bar", StatusBar).update_status(
            status="Ready",
            tokens="",
        )

    def action_copy_message(self) -> None:
        """Copy the last message to clipboard.

        Attempts to use pyperclip. If clipboard is unavailable
        (headless HPC node), falls back to writing to a temp file
        and displays the file path to the user.
        """
        if not self.conversation or not self.conversation.messages:
            return

        # Get the last message content (raw, not formatted)
        last_message = self.conversation.messages[-1]
        content = last_message.content

        chat_log = self.query_one("#chat-log", ChatLog)

        # Try pyperclip first (optional dependency)
        try:
            import pyperclip

            pyperclip.copy(content)
            chat_log.add_message("system", "Copied to clipboard.")
            return
        except ImportError:
            pass  # pyperclip not installed
        except Exception:
            pass  # Clipboard unavailable (headless)

        # Fallback: write to temp file
        try:
            temp_dir = Path(tempfile.gettempdir())
            filename = f"chatty-export-{uuid.uuid4().hex[:8]}.txt"
            filepath = temp_dir / filename
            filepath.write_text(content)
            chat_log.add_message("system", f"Saved to: {filepath}")
        except Exception as e:
            chat_log.add_message("error", f"Failed to copy: {e}")

    def action_cancel(self) -> None:
        """Cancel the current generation.

        Cancels any in-flight LLM request by cancelling the async task.
        Partial responses may be kept or discarded based on config.
        """
        if self.current_worker and self.current_worker.is_running:
            self.current_worker.cancel()
            self.query_one("#status-bar", StatusBar).update_status(status="Cancelled")
            self.current_worker = None

    async def on_unmount(self) -> None:
        """Clean up when app is closing."""
        if self.client:
            await self.client.close()


def main(
    query_file: str | None = None,
    config_with_sources: ConfigWithSources | None = None,
) -> None:
    """Run the chat application.

    Entry point for the chatty UI. Creates and runs the Textual app.

    Args:
        query_file: Optional path to file containing initial query.
        config_with_sources: Pre-loaded configuration (loads default if None).
    """
    app = ChatApp(query_file=query_file, config_with_sources=config_with_sources)
    app.run()
