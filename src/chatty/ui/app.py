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

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, cast

from rich.markdown import Markdown as RichMarkdown
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
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
        super().__init__()
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

    #chat-log {
        height: 100%;
        padding: 1;
    }

    MessageWidget {
        margin-bottom: 1;
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
        Binding("ctrl+n", "new_session", "New Session"),
        Binding("ctrl+o", "load_file", "Load File"),
        Binding("escape", "cancel", "Interrupt"),
        # Hidden but functional (accessible via ^p palette or Keys panel)
        Binding("ctrl+r", "regenerate", "Regenerate Last Response", show=False),
        Binding("ctrl+t", "toggle_stream", "Toggle Streaming", show=False),
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

                # Update conversation with complete response
                self.conversation.add_assistant_message(response_content)

            else:
                # Non-streaming mode - message renders with markdown immediately
                from chatty.client.openai_client import AssistantMessage

                result = await self.client.chat(messages, stream=False)
                response = cast(AssistantMessage, result)

                chat_log.add_message("assistant", response.content)
                self.conversation.add_assistant_message(response.content, response.usage)

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
