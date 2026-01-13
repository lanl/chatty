"""Custom footer widget for chatty UI.

This module provides a custom footer that gives full control over keybinding
display order and styling, replacing Textual's built-in Footer widget.

The footer displays keyboard shortcuts in a user-defined order, adapts to
terminal width, and supports conditional visibility for context-sensitive
bindings (e.g., showing "Stop" only during generation).
"""

from __future__ import annotations

from textual.events import Click
from textual.reactive import reactive
from textual.widgets import Static


class ChattyFooter(Static):
    """Custom footer with controlled keybinding display.

    Displays keyboard shortcuts in an explicit order, independent of how
    bindings are defined in the application. Adapts to terminal width by
    showing fewer bindings on narrow terminals. Supports clicking on
    bindings to trigger the associated action.

    Attributes:
        VISIBLE_BINDINGS: List of (key, label, action) tuples in display order.
        is_generating: Whether LLM generation is in progress.

    CSS Classes:
        .footer-key: Styling for the key portion (e.g., "^P")
        .footer-label: Styling for the label portion (e.g., "Submit")
    """

    # Explicit display order - most important bindings first
    # Format: (display_key, label, action_name)
    # These are shown left-to-right in the footer
    VISIBLE_BINDINGS: list[tuple[str, str, str]] = [
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

    # Keys that should always be shown, even on narrow terminals
    PRIORITY_KEYS: set[str] = {"^P", "^Q", "Esc"}

    # Reactive property to trigger re-render when generation state changes
    is_generating: reactive[bool] = reactive(False)

    def __init__(self, id: str | None = None) -> None:  # noqa: A002
        """Initialize the footer.

        Args:
            id: Optional DOM identifier for CSS styling.
        """
        super().__init__("", id=id)
        self._cached_width: int = 0
        # Track positions of bindings for click detection
        # List of (start_pos, end_pos, action_name)
        self._binding_positions: list[tuple[int, int, str]] = []

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

    def on_click(self, event: Click) -> None:
        """Handle click events to trigger bindings.

        Args:
            event: The click event with coordinates.
        """
        # Find which binding was clicked based on x coordinate
        # Account for CSS left padding (padding: 0 1 in app.tcss)
        x = event.x - 1
        if x < 0:
            return

        for start, end, action in self._binding_positions:
            if start <= x < end:
                # Call the action method on the app
                # For quit, use exit() directly; for others, call action_* method
                if action == "quit":
                    self.app.exit()
                else:
                    action_method = getattr(self.app, f"action_{action}", None)
                    if action_method:
                        action_method()
                break

    def _rebuild_display(self) -> None:
        """Rebuild the footer text based on current width and state."""
        # Get available width
        try:
            width = self.size.width
        except Exception:
            width = 80  # Fallback

        if width < 1:
            width = 80

        # Select bindings that fit
        bindings_to_show = self._select_bindings_for_width(width)

        # Build the display string and track positions for click handling
        parts = []
        self._binding_positions = []
        current_pos = 0
        spacing = "  "  # 2 spaces between bindings

        for key, label, action in bindings_to_show:
            # Format: [dim]^P[/dim] Submit
            # The actual rendered length is: key + space + label
            display_text = f"[dim]{key}[/dim] {label}"
            # Calculate rendered width (without markup)
            rendered_width = len(key) + 1 + len(label)

            parts.append(display_text)

            # Track position for click detection
            end_pos = current_pos + rendered_width
            self._binding_positions.append((current_pos, end_pos, action))
            current_pos = end_pos + len(spacing)

        self.update(spacing.join(parts))

    def _select_bindings_for_width(self, width: int) -> list[tuple[str, str, str]]:
        """Select which bindings to show based on terminal width.

        Uses actual content width calculation instead of fixed estimate.

        Args:
            width: Available width in characters.

        Returns:
            List of (key, label, action) tuples to display.
        """
        bindings = list(self.VISIBLE_BINDINGS)
        spacing_width = 2  # "  " between items

        # Calculate total width needed for all bindings
        def calc_total_width(items: list[tuple[str, str, str]]) -> int:
            if not items:
                return 0
            content = sum(len(key) + 1 + len(label) for key, label, _ in items)
            spacing = spacing_width * (len(items) - 1)
            return content + spacing

        # If all fit, show all
        if calc_total_width(bindings) <= width:
            return bindings

        # Otherwise, start with priority bindings and add others until we run out of space
        priority = [b for b in bindings if b[0] in self.PRIORITY_KEYS]
        others = [b for b in bindings if b[0] not in self.PRIORITY_KEYS]

        result = list(priority)
        for binding in others:
            test_result = result + [binding]
            if calc_total_width(test_result) <= width:
                result.append(binding)
            else:
                break  # No more room

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
