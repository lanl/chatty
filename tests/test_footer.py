"""Tests for chatty footer widget.

Tests the custom ChattyFooter widget: click detection, width truncation,
binding positions, and generation mode.
"""

from __future__ import annotations

from textual.app import App, ComposeResult

from chatty.ui.footer import ChattyFooter


class FooterTestApp(App[None]):
    """Minimal app for testing ChattyFooter widget."""

    def __init__(self) -> None:
        super().__init__()
        self.action_called: str | None = None

    def compose(self) -> ComposeResult:
        yield ChattyFooter(id="footer")

    def action_submit(self) -> None:
        """Track submit action calls."""
        self.action_called = "submit"

    def action_load_file(self) -> None:
        """Track load_file action calls."""
        self.action_called = "load_file"

    def action_cancel(self) -> None:
        """Track cancel action calls."""
        self.action_called = "cancel"

    def action_copy(self) -> None:
        """Track copy action calls."""
        self.action_called = "copy"

    def action_save(self) -> None:
        """Track save action calls."""
        self.action_called = "save"

    def action_browse_sessions(self) -> None:
        """Track browse_sessions action calls."""
        self.action_called = "browse_sessions"

    def action_new_session(self) -> None:
        """Track new_session action calls."""
        self.action_called = "new_session"

    def action_search(self) -> None:
        """Track search action calls."""
        self.action_called = "search"


class TestChattyFooter:
    """Tests for ChattyFooter widget."""

    async def test_initial_display(self) -> None:
        """Shows all bindings on wide terminal."""
        app = FooterTestApp()
        async with app.run_test() as _:
            footer = app.query_one("#footer", ChattyFooter)
            # Should have binding positions tracked
            assert len(footer._binding_positions) > 0
            # Should have all standard bindings visible
            assert len(footer.VISIBLE_BINDINGS) == 9

    async def test_bindings_display_order(self) -> None:
        """Bindings are displayed in defined order."""
        app = FooterTestApp()
        async with app.run_test() as _:
            footer = app.query_one("#footer", ChattyFooter)

            # Check the order in VISIBLE_BINDINGS
            expected_order = [
                ("^P", "Submit", "submit"),
                ("^O", "File", "load_file"),
                ("Esc", "Stop", "cancel"),
                ("^F", "Find", "search"),
                ("^C", "Copy", "copy"),
                ("^S", "Save", "save"),
                ("^L", "Load", "browse_sessions"),
                ("^N", "New", "new_session"),
                ("^Q", "Quit", "quit"),
            ]
            assert expected_order == footer.VISIBLE_BINDINGS

    async def test_binding_positions_tracked(self) -> None:
        """_binding_positions updated on rebuild."""
        app = FooterTestApp()
        async with app.run_test() as _:
            footer = app.query_one("#footer", ChattyFooter)

            # Binding positions should be tuples of (start, end, action)
            for start, end, action in footer._binding_positions:
                assert isinstance(start, int)
                assert isinstance(end, int)
                assert isinstance(action, str)
                assert end > start

    async def test_priority_bindings_preserved(self) -> None:
        """Submit, Quit, Stop are priority bindings."""
        app = FooterTestApp()
        async with app.run_test() as _:
            footer = app.query_one("#footer", ChattyFooter)

            # Check priority keys are defined
            assert "^P" in footer.PRIORITY_KEYS  # Submit
            assert "^Q" in footer.PRIORITY_KEYS  # Quit
            assert "Esc" in footer.PRIORITY_KEYS  # Stop

    async def test_select_bindings_for_width_all_fit(self) -> None:
        """All bindings shown when width is sufficient."""
        app = FooterTestApp()
        async with app.run_test() as _:
            footer = app.query_one("#footer", ChattyFooter)

            # With plenty of width, all bindings should be shown
            result = footer._select_bindings_for_width(200)
            assert len(result) == len(footer.VISIBLE_BINDINGS)

    async def test_select_bindings_for_width_narrow(self) -> None:
        """Fewer bindings shown when width is limited."""
        app = FooterTestApp()
        async with app.run_test() as _:
            footer = app.query_one("#footer", ChattyFooter)

            # With very narrow width, fewer bindings
            result = footer._select_bindings_for_width(30)
            assert len(result) < len(footer.VISIBLE_BINDINGS)

            # Priority bindings should still be present
            actions = [b[2] for b in result]
            assert "submit" in actions or "quit" in actions

    async def test_generation_mode_reactive(self) -> None:
        """set_generation_mode() updates reactive property."""
        app = FooterTestApp()
        async with app.run_test() as _:
            footer = app.query_one("#footer", ChattyFooter)

            # Initially not generating
            assert footer.is_generating is False

            # Set to generating
            footer.set_generation_mode(True)
            assert footer.is_generating is True

            # Set back
            footer.set_generation_mode(False)
            assert footer.is_generating is False

    async def test_click_triggers_action(self) -> None:
        """Clicking on binding triggers action."""
        app = FooterTestApp()
        async with app.run_test() as pilot:
            footer = app.query_one("#footer", ChattyFooter)

            # Get position of first binding (Submit)
            if footer._binding_positions:
                start, end, action = footer._binding_positions[0]
                # Simulate click in the middle of the binding
                # Add 1 for CSS padding
                click_x = start + 1 + (end - start) // 2
                await pilot.click("#footer", offset=(click_x, 0))

                # Action should have been called
                assert app.action_called == action

    async def test_click_quit_exits(self) -> None:
        """Clicking Quit calls app.exit()."""
        app = FooterTestApp()
        async with app.run_test() as pilot:
            footer = app.query_one("#footer", ChattyFooter)

            # Find quit binding position
            quit_binding = None
            for start, end, action in footer._binding_positions:
                if action == "quit":
                    quit_binding = (start, end, action)
                    break

            if quit_binding:
                start, end, _action = quit_binding
                # Simulate click (add 1 for padding)
                click_x = start + 1 + (end - start) // 2
                await pilot.click("#footer", offset=(click_x, 0))
                # App should have exited (test will complete without errors)
