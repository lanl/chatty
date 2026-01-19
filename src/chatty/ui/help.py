"""Help modal for chatty UI.

This module provides a help modal that displays keyboard shortcuts,
current configuration, quick tips, and version information.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, Static


class HelpModal(ModalScreen[None]):
    """Modal screen displaying help information.

    Shows keyboard shortcuts, current configuration, quick tips,
    and version information. Opened with F1 or Ctrl+H.

    Returns None when dismissed (Escape key).

    Usage:
        modal = HelpModal()
        modal.set_config(model="gpt-4", streaming=True, ...)
        app.push_screen(modal)

    Bindings:
        Escape: Close the help modal
    """

    CSS = """
    HelpModal {
        align: center middle;
    }

    #help-dialog {
        width: 76;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        background: $surface;
        border: thick $primary;
    }

    #help-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
        width: 100%;
    }

    #help-content {
        height: auto;
        max-height: 100%;
    }

    .help-section-header {
        margin-top: 1;
        color: $text-muted;
    }

    .help-section {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "close", "Close"),
    ]

    def __init__(self) -> None:
        """Initialize the help modal."""
        super().__init__()
        self._model: str = ""
        self._streaming: bool = True
        self._context_window: int | None = None
        self._transcript_enabled: bool = False

    def set_config(
        self,
        model: str,
        streaming: bool,
        context_window: int | None,
        transcript_enabled: bool,
    ) -> None:
        """Set configuration values to display.

        Call this before pushing the modal to screen.

        Args:
            model: Current model name.
            streaming: Whether streaming is enabled.
            context_window: Context window size in tokens, or None.
            transcript_enabled: Whether transcript logging is enabled.
        """
        self._model = model
        self._streaming = streaming
        self._context_window = context_window
        self._transcript_enabled = transcript_enabled

    def compose(self) -> ComposeResult:
        """Create the help dialog layout."""
        with VerticalScroll(id="help-dialog"):
            yield Label("Chatty Help", id="help-title")
            yield Static(id="help-content")

    def on_mount(self) -> None:
        """Populate help content when modal opens."""
        content = self.query_one("#help-content", Static)
        content.update(self._build_help_text())

    def _build_help_text(self) -> str:
        """Build the help text content.

        Returns:
            Formatted help text with Rich markup.
        """
        # Placeholder content - will be expanded in subsequent tasks
        lines = [
            "[dim]── Keyboard Shortcuts ─────────────────────────────[/dim]",
            "",
            "(Shortcuts will be added in Task 2)",
            "",
            "[dim]── Current Configuration ─────────────────────────[/dim]",
            "",
            "(Config display will be added in Task 3)",
            "",
            "[dim]── Quick Tips ────────────────────────────────────[/dim]",
            "",
            "(Tips will be added in Task 4)",
            "",
            "[dim]───────────────────────────────────────────────────[/dim]",
            "(Version info will be added in Task 5)",
        ]
        return "\n".join(lines)

    def action_close(self) -> None:
        """Handle Escape key - close the modal."""
        self.dismiss(None)
