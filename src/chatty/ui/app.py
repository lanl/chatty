"""Textual application and widgets for chatty."""

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import Footer, Header, Input, Static


class ChatLog(Static):
    """Widget to display chat messages."""

    def __init__(self, id: str | None = None) -> None:  # noqa: A002
        super().__init__("", id=id)
        self.messages: list[str] = []

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the chat log."""
        prefix = {"user": "You", "assistant": "Assistant", "system": "System"}.get(role, role)
        self.messages.append(f"**{prefix}:** {content}")
        self.update("\n\n".join(self.messages))


class StatusBar(Static):
    """Widget to display status information."""

    def __init__(self, id: str | None = None) -> None:  # noqa: A002
        super().__init__("Ready | Model: gpt-4.1 | Stream: on", id=id)


class ChatApp(App[None]):
    """Chatty terminal UI application."""

    TITLE = "chatty"
    CSS = """
    Screen {
        layout: grid;
        grid-size: 1;
        grid-rows: 1fr auto auto;
    }

    #chat-log {
        height: 100%;
        overflow-y: auto;
        padding: 1;
        border: solid green;
    }

    #input-container {
        height: auto;
        padding: 1;
    }

    #status-bar {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }

    Input {
        width: 100%;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+t", "toggle_stream", "Toggle Stream"),
        ("ctrl+r", "regenerate", "Regenerate"),
        ("ctrl+o", "load_file", "Load File"),
        ("ctrl+n", "new_session", "New Session"),
        ("ctrl+y", "copy_message", "Copy"),
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, query_file: str | None = None) -> None:
        super().__init__()
        self.query_file = query_file
        self.streaming = True

    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header()
        yield Container(
            ChatLog(id="chat-log"),
            id="chat-container",
        )
        yield Vertical(
            Input(placeholder="Type your message...", id="input"),
            id="input-container",
        )
        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Handle app mount."""
        self.query_one("#input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle message submission."""
        if not event.value.strip():
            return

        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.add_message("user", event.value)

        # Clear input
        event.input.value = ""

        # TODO: Send to LLM and display response
        chat_log.add_message("assistant", "Chat functionality not yet implemented.")

    def action_toggle_stream(self) -> None:
        """Toggle streaming mode."""
        self.streaming = not self.streaming
        status = "on" if self.streaming else "off"
        self.query_one("#status-bar", StatusBar).update(
            f"Ready | Model: gpt-4.1 | Stream: {status}"
        )

    def action_regenerate(self) -> None:
        """Regenerate last response."""
        # TODO: Implement regeneration
        pass

    def action_load_file(self) -> None:
        """Load query from file."""
        # TODO: Implement file loading
        pass

    def action_new_session(self) -> None:
        """Start a new session."""
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.messages = []
        chat_log.update("")

    def action_copy_message(self) -> None:
        """Copy last message to clipboard."""
        # TODO: Implement clipboard copy with fallback
        pass

    def action_cancel(self) -> None:
        """Cancel current generation."""
        # TODO: Implement cancellation
        pass


def main(query_file: str | None = None) -> None:
    """Run the chat application."""
    app = ChatApp(query_file=query_file)
    app.run()
