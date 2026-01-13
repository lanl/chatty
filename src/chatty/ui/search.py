"""Search bar widget for finding text in chat history.

This module provides an inline search bar that appears below the status bar
and allows users to search through the chat log.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, Static


class SearchBar(Horizontal):
    """Inline search bar with input field and match counter.

    The search bar appears below the header when activated with Ctrl+F.
    It provides case-insensitive search through the chat log.

    Bindings:
        Escape: Close the search bar
        Ctrl+F: Next match (handled by app.py)

    Attributes:
        search_query: The current search query (reactive)
        match_index: Current match index (0-based)
        total_matches: Total number of matches found

    Events:
        SearchBar.Closed: Posted when search bar is closed
        SearchBar.QueryChanged: Posted when search query changes
        SearchBar.NextMatch: Posted when user requests next match
    """

    # CSS is in app.tcss - we don't use DEFAULT_CSS to avoid duplication
    # Ctrl+F (in app.py) handles "next match"; Escape closes search

    BINDINGS = [
        Binding("escape", "close", "Close", show=False, priority=True),
    ]

    # Reactive properties
    search_query: reactive[str] = reactive("", init=False)
    match_index: reactive[int] = reactive(0, init=False)
    total_matches: reactive[int] = reactive(0, init=False)  # messages with matches
    total_occurrences: reactive[int] = reactive(0, init=False)  # total occurrences

    class Closed(Message):
        """Posted when the search bar is closed."""

        pass

    class QueryChanged(Message):
        """Posted when the search query changes."""

        def __init__(self, query: str) -> None:
            super().__init__()
            self.query = query

    class NextMatch(Message):
        """Posted when user requests next match."""

        pass

    class PrevMatch(Message):
        """Posted when user requests previous match."""

        pass

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Input(placeholder="Search... (^F=next, Esc=close)", id="search-input")
        yield Static("0/0", id="match-counter")

    def on_mount(self) -> None:
        """Handle mount event."""
        self._update_counter()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        self.search_query = event.value
        self.post_message(self.QueryChanged(event.value))

    def watch_search_query(self, new_query: str) -> None:
        """React to query changes."""
        # Reset match index when query changes
        if new_query != self.search_query:
            self.match_index = 0

    def watch_match_index(self) -> None:
        """React to match index changes."""
        self._update_counter()

    def watch_total_matches(self) -> None:
        """React to total matches changes."""
        self._update_counter()

    def watch_total_occurrences(self) -> None:
        """React to total occurrences changes."""
        self._update_counter()

    def _update_counter(self) -> None:
        """Update the match counter display.

        Format: "N matches (M msgs)" where N is total occurrences,
        M is number of messages containing matches.
        Navigation index shows current message being viewed.
        """
        counter = self.query_one("#match-counter", Static)
        if self.total_matches == 0:
            if self.search_query:
                counter.update("No matches")
                counter.add_class("no-matches")
            else:
                counter.update("")
                counter.remove_class("no-matches")
        else:
            # Show occurrences and message count
            # e.g., "8 matches (3 msgs) [2/3]"
            counter.update(
                f"{self.total_occurrences} matches ({self.total_matches} msgs) "
                f"[{self.match_index + 1}/{self.total_matches}]"
            )
            counter.remove_class("no-matches")

    def action_close(self) -> None:
        """Close the search bar."""
        self.hide()
        self.post_message(self.Closed())

    def action_next_match(self) -> None:
        """Go to next match."""
        if self.total_matches > 0:
            self.match_index = (self.match_index + 1) % self.total_matches
        self.post_message(self.NextMatch())

    def action_prev_match(self) -> None:
        """Go to previous match."""
        if self.total_matches > 0:
            self.match_index = (self.match_index - 1) % self.total_matches
        self.post_message(self.PrevMatch())

    def show(self) -> None:
        """Show the search bar and focus input."""
        self.add_class("visible")
        search_input = self.query_one("#search-input", Input)
        search_input.focus()

    def hide(self) -> None:
        """Hide the search bar and clear state."""
        self.remove_class("visible")
        self.search_query = ""
        self.match_index = 0
        self.total_matches = 0
        self.total_occurrences = 0
        # Clear input
        search_input = self.query_one("#search-input", Input)
        search_input.value = ""

    def set_matches(
        self, total_messages: int, current: int = 0, total_occurrences: int = 0
    ) -> None:
        """Update match count and current index.

        Args:
            total_messages: Number of messages containing matches.
            current: Current message index (0-based).
            total_occurrences: Total number of term occurrences.
        """
        self.total_matches = total_messages
        self.total_occurrences = total_occurrences if total_occurrences else total_messages
        self.match_index = current if total_messages > 0 else 0

    @property
    def is_visible(self) -> bool:
        """Check if search bar is visible."""
        return self.has_class("visible")
