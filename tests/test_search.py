"""Tests for search functionality.

Tests the SearchBar widget and search integration with ChatLog.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from chatty.config import Config, ConfigWithSources
from chatty.ui.app import ChatApp
from chatty.ui.search import SearchBar
from chatty.ui.widgets import ChatLog, MessageWidget


@pytest.fixture
def mock_config_with_sources(tmp_path: Path) -> ConfigWithSources:
    """Create a mock ConfigWithSources for testing."""
    config = Config(
        base_url="http://test.local/v1",
        api_key=SecretStr("test-key"),
        model="test-model",
        session_path=str(tmp_path / "sessions"),
        export_path=str(tmp_path / "exports"),
        copy_fallback_path=str(tmp_path / "copies"),
        transcript_enabled=False,
    )
    return ConfigWithSources(config=config, sources={})


# ============================================================================
# SearchBar Widget Tests
# ============================================================================


class TestSearchBarWidget:
    """Tests for SearchBar widget."""

    async def test_search_bar_hidden_by_default(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """SearchBar is hidden by default."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test():
            search_bar = app.query_one("#search-bar", SearchBar)
            assert not search_bar.is_visible

    async def test_search_bar_shows_on_ctrl_f(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """Ctrl+F shows the search bar."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            search_bar = app.query_one("#search-bar", SearchBar)
            assert not search_bar.is_visible

            app.action_search()
            await pilot.pause()

            assert search_bar.is_visible

    async def test_search_bar_hides_on_escape(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """Escape hides the search bar."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            search_bar = app.query_one("#search-bar", SearchBar)

            # Show it
            app.action_search()
            await pilot.pause()
            assert search_bar.is_visible

            # Escape should close it
            app.action_cancel()
            await pilot.pause()
            assert not search_bar.is_visible

    async def test_ctrl_f_advances_when_open(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """Ctrl+F advances to next match when search is already open."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            search_bar = app.query_one("#search-bar", SearchBar)
            chat_log = app.query_one("#chat-log", ChatLog)

            # Add messages with matches
            chat_log.add_message("user", "Hello 1")
            chat_log.add_message("user", "Hello 2")
            await pilot.pause()

            # Open search
            app.action_search()
            await pilot.pause()
            assert search_bar.is_visible

            # Search for something
            matches = chat_log.search("hello")
            search_bar.set_matches(len(matches), 0)
            await pilot.pause()
            assert chat_log.current_match_index == 0

            # Second Ctrl+F should advance to next match, not close
            app.action_search()
            await pilot.pause()
            assert search_bar.is_visible  # Still visible
            assert chat_log.current_match_index == 1  # Advanced to next

    async def test_match_counter_initial(self, mock_config_with_sources: ConfigWithSources) -> None:
        """Match counter shows 0/0 initially."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            search_bar = app.query_one("#search-bar", SearchBar)

            app.action_search()
            await pilot.pause()

            assert search_bar.total_matches == 0
            assert search_bar.match_index == 0


# ============================================================================
# MessageWidget Highlight Tests
# ============================================================================


class TestMessageWidgetHighlight:
    """Tests for MessageWidget highlight functionality."""

    def test_contains_query_case_insensitive(self) -> None:
        """contains_query is case-insensitive."""
        widget = MessageWidget("user", "Hello World")
        assert widget.contains_query("hello")
        assert widget.contains_query("WORLD")
        assert widget.contains_query("lo wo")

    def test_contains_query_not_found(self) -> None:
        """contains_query returns False for non-matches."""
        widget = MessageWidget("user", "Hello World")
        assert not widget.contains_query("xyz")
        assert not widget.contains_query("goodbye")

    def test_contains_query_empty(self) -> None:
        """contains_query returns False for empty query."""
        widget = MessageWidget("user", "Hello World")
        assert not widget.contains_query("")

    def test_set_highlight_adds_class(self) -> None:
        """set_highlight adds search-match class."""
        widget = MessageWidget("user", "Hello World")
        widget.set_highlight("hello")
        assert widget.has_class("search-match")

    def test_set_highlight_current_adds_class(self) -> None:
        """set_highlight with is_current adds current-match class."""
        widget = MessageWidget("user", "Hello World")
        widget.set_highlight("hello", is_current=True)
        assert widget.has_class("search-match")
        assert widget.has_class("current-match")

    def test_clear_highlight_removes_classes(self) -> None:
        """clear_highlight removes highlight classes."""
        widget = MessageWidget("user", "Hello World")
        widget.set_highlight("hello", is_current=True)
        widget.clear_highlight()
        assert not widget.has_class("search-match")
        assert not widget.has_class("current-match")


# ============================================================================
# ChatLog Search Tests
# ============================================================================


class TestChatLogSearch:
    """Tests for ChatLog search functionality."""

    async def test_search_finds_matches(self, mock_config_with_sources: ConfigWithSources) -> None:
        """search() returns matching messages."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            chat_log = app.query_one("#chat-log", ChatLog)

            chat_log.add_message("user", "Hello world")
            chat_log.add_message("assistant", "Hi there")
            chat_log.add_message("user", "Say hello again")
            await pilot.pause()

            matches = chat_log.search("hello")
            assert len(matches) == 2

    async def test_search_case_insensitive(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """search() is case-insensitive."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            chat_log = app.query_one("#chat-log", ChatLog)

            chat_log.add_message("user", "HELLO")
            chat_log.add_message("user", "hello")
            chat_log.add_message("user", "HeLLo")
            await pilot.pause()

            matches = chat_log.search("hello")
            assert len(matches) == 3

    async def test_search_no_matches(self, mock_config_with_sources: ConfigWithSources) -> None:
        """search() returns empty list for no matches."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            chat_log = app.query_one("#chat-log", ChatLog)

            chat_log.add_message("user", "Hello")
            chat_log.add_message("assistant", "Hi")
            await pilot.pause()

            matches = chat_log.search("goodbye")
            assert len(matches) == 0

    async def test_search_empty_query_clears(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """search() with empty query clears search."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            chat_log = app.query_one("#chat-log", ChatLog)

            chat_log.add_message("user", "Hello")
            await pilot.pause()

            chat_log.search("hello")
            assert chat_log.match_count == 1

            chat_log.search("")
            assert chat_log.match_count == 0

    async def test_clear_search(self, mock_config_with_sources: ConfigWithSources) -> None:
        """clear_search() removes all highlights."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            chat_log = app.query_one("#chat-log", ChatLog)

            chat_log.add_message("user", "Hello")
            chat_log.add_message("user", "Hello again")
            await pilot.pause()

            chat_log.search("hello")
            assert chat_log.match_count == 2

            chat_log.clear_search()
            assert chat_log.match_count == 0

    async def test_next_match_cycles(self, mock_config_with_sources: ConfigWithSources) -> None:
        """next_match() cycles through matches."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            chat_log = app.query_one("#chat-log", ChatLog)

            chat_log.add_message("user", "Hello 1")
            chat_log.add_message("user", "Hello 2")
            chat_log.add_message("user", "Hello 3")
            await pilot.pause()

            chat_log.search("hello")
            assert chat_log.current_match_index == 0

            chat_log.next_match()
            assert chat_log.current_match_index == 1

            chat_log.next_match()
            assert chat_log.current_match_index == 2

            chat_log.next_match()  # Cycles back to 0
            assert chat_log.current_match_index == 0

    async def test_prev_match_cycles(self, mock_config_with_sources: ConfigWithSources) -> None:
        """prev_match() cycles through matches in reverse."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            chat_log = app.query_one("#chat-log", ChatLog)

            chat_log.add_message("user", "Hello 1")
            chat_log.add_message("user", "Hello 2")
            chat_log.add_message("user", "Hello 3")
            await pilot.pause()

            chat_log.search("hello")
            assert chat_log.current_match_index == 0

            chat_log.prev_match()  # Cycles to last
            assert chat_log.current_match_index == 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestSearchIntegration:
    """Integration tests for search functionality."""

    async def test_search_with_messages(self, mock_config_with_sources: ConfigWithSources) -> None:
        """Full search flow with messages."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            chat_log = app.query_one("#chat-log", ChatLog)
            search_bar = app.query_one("#search-bar", SearchBar)

            # Add messages
            chat_log.add_message("user", "What is Python?")
            chat_log.add_message("assistant", "Python is a programming language.")
            chat_log.add_message("user", "Tell me more about Python.")
            await pilot.pause()

            # Open search
            app.action_search()
            await pilot.pause()

            # Search
            matches = chat_log.search("python")
            search_bar.set_matches(len(matches), 0)
            await pilot.pause()

            assert search_bar.total_matches == 3
            assert chat_log.match_count == 3

    async def test_close_search_returns_focus(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """Closing search returns focus to input."""
        from chatty.ui.widgets import ChatInput

        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            input_widget = app.query_one("#input", ChatInput)

            # Open search
            app.action_search()
            await pilot.pause()

            # Close search
            app.action_cancel()
            await pilot.pause()

            # Focus should be back on input
            assert input_widget.has_focus


# ============================================================================
# SearchBar Update Count Tests (v0.2.10)
# ============================================================================


class TestSearchBarUpdateCount:
    """Tests for SearchBar.set_matches method."""

    async def test_set_matches_with_results(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """set_matches updates counter with results."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            search_bar = app.query_one("#search-bar", SearchBar)

            app.action_search()
            await pilot.pause()

            search_bar.set_matches(5, 2)
            await pilot.pause()

            assert search_bar.total_matches == 5
            assert search_bar.match_index == 2

    async def test_set_matches_zero_results(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """set_matches handles zero results."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            search_bar = app.query_one("#search-bar", SearchBar)

            app.action_search()
            await pilot.pause()

            search_bar.set_matches(0, 0)
            await pilot.pause()

            assert search_bar.total_matches == 0

    def test_count_occurrences(self) -> None:
        """count_occurrences returns correct count."""
        widget = MessageWidget("user", "hello hello hello world")
        assert widget.count_occurrences("hello") == 3
        assert widget.count_occurrences("world") == 1
        assert widget.count_occurrences("xyz") == 0

    async def test_total_occurrences_across_messages(
        self, mock_config_with_sources: ConfigWithSources
    ) -> None:
        """ChatLog.total_occurrences counts across all messages."""
        app = ChatApp(config_with_sources=mock_config_with_sources)
        async with app.run_test() as pilot:
            chat_log = app.query_one("#chat-log", ChatLog)

            chat_log.add_message("user", "hello world")
            chat_log.add_message("assistant", "hello hello")
            await pilot.pause()

            total = chat_log.total_occurrences("hello")
            assert total == 3
