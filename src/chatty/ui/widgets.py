"""Widget components for chatty UI.

This module contains the core UI widgets used by the chat application:
- ChatInput: Multi-line text entry with Ctrl+P submit
- MessageWidget: Individual chat message with markdown rendering
- ChatLog: Scrollable container for messages
- StatusBar: Status line with spinner, model, and token display
"""

from __future__ import annotations

import time

from rich.markdown import Markdown as RichMarkdown
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Static, TextArea


class ChatInput(TextArea):
    """Custom TextArea for chat input with Ctrl+P submit.

    Inherits standard TextArea bindings (cursor movement, copy/paste, etc.)
    for normal text editing. Multi-line input via Enter key.

    Ctrl+P submits the message ("P for Prompt").

    Bindings:
        Ctrl+P: Submit the current message (posts Submitted event)
        Enter: Insert newline (inherited from TextArea)
    """

    BINDINGS = [
        Binding("ctrl+p", "send", "Submit Query", priority=True),
    ]

    class Submitted(TextArea.Changed):
        """Event posted when Ctrl+P is pressed to submit."""

        pass

    def action_send(self) -> None:
        """Handle Ctrl+P to submit the message."""
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
        highlight_query: Current search query for highlighting.
        is_current_match: Whether this widget contains the current search match.

    CSS Classes:
        .user-message: Applied to user messages (dimmed styling)
        .assistant-message: Applied to assistant messages (default bright)
        .system-message: Applied to system messages (accent color)
        .error-message: Applied to error messages (red with background)
        .search-match: Applied when message contains a search match
        .current-match: Applied when message is the current search match
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
        self.highlight_query: str = ""
        self.is_current_match: bool = False
        self._update_display()

    def _update_display(self) -> None:
        """Update the displayed text.

        Rendering strategy:
        - During streaming: Raw text for performance
        - After streaming complete (assistant, no search): Markdown with syntax highlighting
        - During search: Plain text for all messages (enables word-level highlighting)
        - User/system/error: Always plain text with bold prefix

        Search highlighting:
        - All messages during search: Word-level highlighting with Rich markup
        - When search is cleared, assistant messages return to markdown rendering
        """
        prefix = self.ROLE_PREFIXES.get(self.role, self.role)
        content = self.message_content

        # Use plain text during search to enable word highlighting, or for non-assistant roles
        use_plain_text = (
            self.role != "assistant"
            or self.is_streaming
            or self.highlight_query  # Search active → use plain text
        )

        if use_plain_text:
            # Plain text rendering with word-level search highlighting
            display_content = content
            if self.highlight_query and self.contains_query(self.highlight_query):
                display_content = self._apply_highlight(content, self.highlight_query)
            self.update(f"**{prefix}:** {display_content}")
        else:
            # Assistant messages get markdown rendering when not streaming and not searching
            markdown_content = f"**{prefix}:**\n\n{content}"
            self.update(RichMarkdown(markdown_content))

        # Update CSS classes for search highlighting
        if self.highlight_query and self.contains_query(self.highlight_query):
            self.add_class("search-match")
        else:
            self.remove_class("search-match")

        if self.is_current_match:
            self.add_class("current-match")
        else:
            self.remove_class("current-match")

    def _apply_highlight(self, text: str, query: str) -> str:
        """Apply Rich markup highlighting to text containing the query.

        Uses [reverse] markup to highlight matches.
        Only used for plain text messages (user, system, error).

        Args:
            text: The text to search in.
            query: The query to highlight.

        Returns:
            Text with Rich markup for highlighting.
        """
        if not query:
            return text

        # Case-insensitive search
        query_lower = query.lower()
        text_lower = text.lower()

        # Find all match positions
        result = []
        last_end = 0
        start = text_lower.find(query_lower)

        while start != -1:
            # Add text before match
            result.append(text[last_end:start])
            # Add highlighted match (use original case)
            match_text = text[start : start + len(query)]
            result.append(f"[reverse]{match_text}[/reverse]")
            last_end = start + len(query)
            start = text_lower.find(query_lower, last_end)

        # Add remaining text
        result.append(text[last_end:])

        return "".join(result)

    def contains_query(self, query: str) -> bool:
        """Check if message contains the search query.

        Args:
            query: Search query (case-insensitive).

        Returns:
            True if message contains query.
        """
        if not query:
            return False
        return query.lower() in self.message_content.lower()

    def set_highlight(self, query: str, is_current: bool = False) -> None:
        """Set the highlight query for this message.

        Args:
            query: Search query to highlight (empty to clear).
            is_current: Whether this is the current match.
        """
        self.highlight_query = query
        self.is_current_match = is_current
        self._update_display()

    def clear_highlight(self) -> None:
        """Clear search highlighting."""
        self.highlight_query = ""
        self.is_current_match = False
        self._update_display()

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

    The container auto-scrolls to the bottom when new messages are added.

    Methods:
        add_message: Add a new message widget.
        get_last_message: Get the last message widget (for streaming).
        append_to_last: Append content to the last message.
        clear_messages: Remove all messages.
        remove_last_message: Remove the last message (for regeneration).
        search: Search for text in messages.
        clear_search: Clear search highlighting.
        highlight_match: Highlight a specific match.
    """

    def __init__(self, id: str | None = None) -> None:  # noqa: A002
        """Initialize the chat log container.

        Args:
            id: Optional DOM identifier for CSS styling.
        """
        super().__init__(id=id)
        self._search_matches: list[MessageWidget] = []
        self._current_match_index: int = 0

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

    def search(self, query: str) -> list[MessageWidget]:
        """Search for text in all messages.

        Finds all messages containing the query (case-insensitive)
        and applies highlighting.

        Args:
            query: Search query string.

        Returns:
            List of MessageWidgets that contain matches.
        """
        self._search_matches = []

        if not query:
            self.clear_search()
            return []

        # Find all matching messages
        for widget in self.query(MessageWidget):
            if widget.contains_query(query):
                widget.set_highlight(query, is_current=False)
                self._search_matches.append(widget)
            else:
                widget.clear_highlight()

        # Highlight first match as current
        if self._search_matches:
            self._current_match_index = 0
            self._search_matches[0].set_highlight(query, is_current=True)

        return self._search_matches

    def clear_search(self) -> None:
        """Clear all search highlighting."""
        for widget in self.query(MessageWidget):
            widget.clear_highlight()
        self._search_matches = []
        self._current_match_index = 0

    def highlight_match(self, index: int) -> MessageWidget | None:
        """Highlight a specific match and scroll to it.

        Args:
            index: Index of the match to highlight (0-based).

        Returns:
            The highlighted MessageWidget, or None if invalid index.
        """
        if not self._search_matches or index < 0 or index >= len(self._search_matches):
            return None

        # Get query from current matches
        query = self._search_matches[0].highlight_query if self._search_matches else ""

        # Update highlighting
        for i, widget in enumerate(self._search_matches):
            widget.set_highlight(query, is_current=(i == index))

        self._current_match_index = index
        target = self._search_matches[index]

        # Scroll to the matched widget
        self.scroll_to_widget(target, animate=False)

        return target

    def next_match(self) -> MessageWidget | None:
        """Go to next search match.

        Returns:
            The next matched MessageWidget, or None if no matches.
        """
        if not self._search_matches:
            return None
        next_index = (self._current_match_index + 1) % len(self._search_matches)
        return self.highlight_match(next_index)

    def prev_match(self) -> MessageWidget | None:
        """Go to previous search match.

        Returns:
            The previous matched MessageWidget, or None if no matches.
        """
        if not self._search_matches:
            return None
        prev_index = (self._current_match_index - 1) % len(self._search_matches)
        return self.highlight_match(prev_index)

    @property
    def match_count(self) -> int:
        """Get the number of search matches."""
        return len(self._search_matches)

    @property
    def current_match_index(self) -> int:
        """Get the current match index (0-based)."""
        return self._current_match_index


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

    Attributes:
        SPINNER_FRAMES: Braille characters for smooth spinner animation
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

    @property
    def last_response_time(self) -> float | None:
        """Get the last recorded response time.

        Returns:
            Response time in seconds, or None if no response recorded.
        """
        return self._last_response_time
