"""Custom footer widget for chatty UI.

This module provides a custom footer that gives full control over keybinding
display order and styling, replacing Textual's built-in Footer widget.

The footer displays keyboard shortcuts in a user-defined order, adapts to
terminal width, and supports conditional visibility for context-sensitive
bindings (e.g., showing "Stop" only during generation).
"""

from __future__ import annotations

from textual.reactive import reactive
from textual.widgets import Static


class ChattyFooter(Static):
    """Custom footer with controlled keybinding display.

    Displays keyboard shortcuts in an explicit order, independent of how
    bindings are defined in the application. Adapts to terminal width by
    showing fewer bindings on narrow terminals.

    Attributes:
        VISIBLE_BINDINGS: List of (key, label) tuples in display order.
        is_generating: Whether LLM generation is in progress.

    CSS Classes:
        .footer-key: Styling for the key portion (e.g., "^P")
        .footer-label: Styling for the label portion (e.g., "Submit")
    """

    # Explicit display order - most important bindings first
    # These are shown left-to-right in the footer
    VISIBLE_BINDINGS: list[tuple[str, str]] = [
        ("^P", "Submit"),
        ("^O", "File"),
        ("Esc", "Stop"),
        ("^C", "Copy"),
        ("^S", "Save"),
        ("^L", "Load"),
        ("^N", "New"),
        ("^Q", "Quit"),
    ]

    # Minimum bindings to show on very narrow terminals
    PRIORITY_BINDINGS: list[tuple[str, str]] = [
        ("^P", "Submit"),
        ("^Q", "Quit"),
        ("Esc", "Stop"),
    ]

    # Reactive property to trigger re-render when generation state changes
    is_generating: reactive[bool] = reactive(False)

    def __init__(self, id: str | None = None) -> None:  # noqa: A002
        """Initialize the footer.

        Args:
            id: Optional DOM identifier for CSS styling.
        """
        super().__init__("", id=id)
        self._cached_width: int = 0

    def on_mount(self) -> None:
        """Initialize display on mount."""
        self._rebuild_display()

    def on_resize(self) -> None:
        """Handle terminal resize - rebuild with new width."""
        self._rebuild_display()

    def watch_is_generating(self, _generating: bool) -> None:
        """React to generation state changes.

        Args:
            _generating: Whether generation is currently in progress (unused,
                triggers rebuild via reactive property).
        """
        self._rebuild_display()

    def _rebuild_display(self) -> None:
        """Rebuild the footer text based on current width and state."""
        # Get available width
        try:
            width = self.size.width
        except Exception:
            width = 80  # Fallback

        if width < 1:
            width = 80

        # Calculate how many bindings we can fit
        # Each binding takes ~12 chars: "^P Submit  " (key + space + label + padding)
        bindings_to_show = self._select_bindings_for_width(width)

        # Build the display string
        parts = []
        for key, label in bindings_to_show:
            # Format: [^P] Submit
            parts.append(f"[dim]{key}[/dim] {label}")

        self.update("  ".join(parts))

    def _select_bindings_for_width(self, width: int) -> list[tuple[str, str]]:
        """Select which bindings to show based on terminal width.

        Args:
            width: Available width in characters.

        Returns:
            List of (key, label) tuples to display.
        """
        # Estimate chars per binding: key(3) + space(1) + label(~6) + spacing(2) = ~12
        CHARS_PER_BINDING = 12

        # How many can we fit?
        max_bindings = max(3, width // CHARS_PER_BINDING)

        # Start with all visible bindings
        bindings = list(self.VISIBLE_BINDINGS)

        # If we have room for all, use all
        if len(bindings) <= max_bindings:
            return bindings

        # Otherwise, prioritize: always include Submit, Quit, and Stop
        # Then fill remaining slots from the rest
        priority_keys = {"^P", "^Q", "Esc"}
        priority = [b for b in bindings if b[0] in priority_keys]
        others = [b for b in bindings if b[0] not in priority_keys]

        # Fill remaining slots
        remaining_slots = max_bindings - len(priority)
        result = priority + others[:remaining_slots]

        # Sort back to original order
        original_order = {b[0]: i for i, b in enumerate(self.VISIBLE_BINDINGS)}
        result.sort(key=lambda b: original_order.get(b[0], 999))

        return result

    def set_generation_mode(self, active: bool) -> None:
        """Set whether generation is in progress.

        When active, "Stop" binding may be highlighted differently.

        Args:
            active: True if generating, False otherwise.
        """
        self.is_generating = active
