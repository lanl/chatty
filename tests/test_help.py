"""Tests for chatty help modal.

Tests the HelpModal screen displaying keyboard shortcuts, configuration,
quick tips, and version information.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Label

from chatty.ui.help import (
    POWER_SHORTCUTS,
    QUICK_TIPS,
    VISIBLE_SHORTCUTS,
    HelpModal,
)


class HelpTestApp(App[None]):
    """App for testing HelpModal."""

    def __init__(self) -> None:
        super().__init__()
        self.modal_dismissed: bool = False

    def compose(self) -> ComposeResult:
        yield Label("Test App")


class TestHelpModal:
    """Tests for HelpModal."""

    async def test_modal_renders(self) -> None:
        """HelpModal renders correctly."""
        app = HelpTestApp()
        async with app.run_test() as pilot:
            modal = HelpModal()
            modal.set_config(
                model="gpt-4",
                streaming=True,
                context_window=128000,
                transcript_enabled=False,
            )
            app.push_screen(modal)
            await pilot.pause()

            # Should have the modal visible
            assert len(app.screen_stack) == 2

            # Should have title label
            title = modal.query_one("#help-title", Label)
            assert title is not None

    async def test_escape_closes_modal(self) -> None:
        """Escape key closes the modal."""
        app = HelpTestApp()
        async with app.run_test() as pilot:
            app.push_screen(HelpModal(), callback=lambda _: setattr(app, "modal_dismissed", True))
            await pilot.pause()

            # Modal should be open
            assert len(app.screen_stack) == 2

            # Press escape
            await pilot.press("escape")
            await pilot.pause()

            # Modal should be closed
            assert app.modal_dismissed is True

    def test_build_help_text_contains_shortcuts(self) -> None:
        """_build_help_text includes visible shortcuts."""
        modal = HelpModal()
        modal.set_config(
            model="gpt-4",
            streaming=True,
            context_window=128000,
            transcript_enabled=False,
        )
        content = modal._build_help_text()

        # Check some visible shortcuts are present
        assert "Submit query" in content
        assert "Load query from file" in content
        assert "Quit chatty" in content

    def test_build_help_text_contains_power_shortcuts(self) -> None:
        """_build_help_text includes power user shortcuts."""
        modal = HelpModal()
        modal.set_config(
            model="gpt-4",
            streaming=True,
            context_window=128000,
            transcript_enabled=False,
        )
        content = modal._build_help_text()

        # Check power shortcuts are mentioned
        assert "hidden from footer" in content
        assert "Export to Markdown" in content

    def test_build_help_text_shows_model(self) -> None:
        """_build_help_text displays the model name."""
        modal = HelpModal()
        modal.set_config(
            model="test-model-123",
            streaming=True,
            context_window=128000,
            transcript_enabled=False,
        )
        content = modal._build_help_text()

        assert "test-model-123" in content

    def test_build_help_text_shows_streaming_on(self) -> None:
        """_build_help_text displays streaming status (On)."""
        modal = HelpModal()
        modal.set_config(
            model="gpt-4",
            streaming=True,
            context_window=128000,
            transcript_enabled=False,
        )
        content = modal._build_help_text()

        assert "Streaming" in content
        assert "On" in content

    def test_build_help_text_shows_streaming_off(self) -> None:
        """_build_help_text displays streaming status (Off)."""
        modal = HelpModal()
        modal.set_config(
            model="gpt-4",
            streaming=False,
            context_window=128000,
            transcript_enabled=False,
        )
        content = modal._build_help_text()

        assert "Off" in content

    def test_build_help_text_shows_context_window(self) -> None:
        """_build_help_text displays context window size."""
        modal = HelpModal()
        modal.set_config(
            model="gpt-4",
            streaming=True,
            context_window=128000,
            transcript_enabled=False,
        )
        content = modal._build_help_text()

        # 128000 tokens should display as "128K tokens"
        assert "128K tokens" in content

    def test_build_help_text_handles_none_context(self) -> None:
        """_build_help_text handles None context window gracefully."""
        modal = HelpModal()
        modal.set_config(
            model="gpt-4",
            streaming=True,
            context_window=None,
            transcript_enabled=False,
        )
        content = modal._build_help_text()

        assert "unknown" in content

    def test_build_help_text_shows_transcript_enabled(self) -> None:
        """_build_help_text displays transcript status (Enabled)."""
        modal = HelpModal()
        modal.set_config(
            model="gpt-4",
            streaming=True,
            context_window=128000,
            transcript_enabled=True,
        )
        content = modal._build_help_text()

        assert "Enabled" in content

    def test_build_help_text_contains_tips(self) -> None:
        """_build_help_text includes quick tips."""
        modal = HelpModal()
        modal.set_config(
            model="gpt-4",
            streaming=True,
            context_window=128000,
            transcript_enabled=False,
        )
        content = modal._build_help_text()

        # Check at least one tip is present
        assert "Press Esc to cancel generation or close dialogs" in content

    def test_build_help_text_contains_version(self) -> None:
        """_build_help_text includes version information."""
        modal = HelpModal()
        modal.set_config(
            model="gpt-4",
            streaming=True,
            context_window=128000,
            transcript_enabled=False,
        )
        content = modal._build_help_text()

        # Should show version and doctor hint
        assert "chatty v" in content
        assert "chatty doctor" in content


class TestFormatContextWindow:
    """Tests for _format_context_window helper."""

    def test_format_none(self) -> None:
        """None context window returns 'unknown'."""
        modal = HelpModal()
        modal._context_window = None
        assert modal._format_context_window() == "unknown"

    def test_format_large_value(self) -> None:
        """Large context window shows K notation."""
        modal = HelpModal()
        modal._context_window = 128000
        assert modal._format_context_window() == "128K tokens"

    def test_format_small_value(self) -> None:
        """Small context window shows exact value."""
        modal = HelpModal()
        modal._context_window = 500
        assert modal._format_context_window() == "500 tokens"

    def test_format_exactly_1k(self) -> None:
        """1000 tokens shows as 1K."""
        modal = HelpModal()
        modal._context_window = 1000
        assert modal._format_context_window() == "1K tokens"


class TestShortcutsConstants:
    """Tests for shortcut constants."""

    def test_visible_shortcuts_not_empty(self) -> None:
        """VISIBLE_SHORTCUTS has entries."""
        assert len(VISIBLE_SHORTCUTS) > 0

    def test_power_shortcuts_not_empty(self) -> None:
        """POWER_SHORTCUTS has entries."""
        assert len(POWER_SHORTCUTS) > 0

    def test_quick_tips_not_empty(self) -> None:
        """QUICK_TIPS has entries."""
        assert len(QUICK_TIPS) > 0

    def test_visible_shortcuts_format(self) -> None:
        """VISIBLE_SHORTCUTS entries are (key, description) tuples."""
        for item in VISIBLE_SHORTCUTS:
            assert len(item) == 2
            key, desc = item
            assert isinstance(key, str)
            assert isinstance(desc, str)
            assert len(key) <= 5  # Keys like "^P", "Esc"
            assert len(desc) > 0

    def test_help_shortcut_in_visible(self) -> None:
        """F1 Help is in VISIBLE_SHORTCUTS."""
        keys = [k for k, _ in VISIBLE_SHORTCUTS]
        assert "F1" in keys
