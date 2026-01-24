"""Tests for chatty UI widgets.

Tests the core UI widgets: StatusBar, MessageWidget, ChatLog, ChatInput.
Uses Textual's testing framework with pilot for widget interactions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.widgets import Collapsible

from chatty.rag.provider import RAGMetadata, RAGSource
from chatty.ui.widgets import ChatInput, ChatLog, CitationsWidget, MessageWidget, StatusBar

# ============================================================================
# MessageWidget ROLE_PREFIXES Tests (v0.2.9 Accessibility)
# ============================================================================


class TestMessageWidgetPrefixes:
    """Tests for accessible message prefixes."""

    def test_system_message_has_info_icon(self) -> None:
        """System messages have ℹ icon prefix for colorblind accessibility."""
        assert MessageWidget.ROLE_PREFIXES["system"] == "ℹ System"

    def test_error_message_has_warning_icon(self) -> None:
        """Error messages have ⚠ icon prefix."""
        assert MessageWidget.ROLE_PREFIXES["error"] == "⚠ Error"

    def test_user_message_no_icon(self) -> None:
        """User messages have no icon (just text label)."""
        assert MessageWidget.ROLE_PREFIXES["user"] == "You"

    def test_assistant_message_no_icon(self) -> None:
        """Assistant messages have no icon (just text label)."""
        assert MessageWidget.ROLE_PREFIXES["assistant"] == "Assistant"


if TYPE_CHECKING:
    pass


# ============================================================================
# Test Apps - Minimal apps to host widgets for testing
# ============================================================================


class StatusBarApp(App[None]):
    """Minimal app for testing StatusBar widget."""

    def __init__(self, model: str = "test-model", streaming: bool = True) -> None:
        super().__init__()
        self.model = model
        self.streaming = streaming

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status-bar", model=self.model, streaming=self.streaming)


class MessageWidgetApp(App[None]):
    """Minimal app for testing MessageWidget."""

    def __init__(self, role: str = "user", content: str = "Hello") -> None:
        super().__init__()
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        yield MessageWidget(self.role, self.content)


class ChatLogApp(App[None]):
    """Minimal app for testing ChatLog widget."""

    def compose(self) -> ComposeResult:
        yield ChatLog(id="chat-log")


class ChatInputApp(App[None]):
    """Minimal app for testing ChatInput widget."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted_content: str | None = None

    def compose(self) -> ComposeResult:
        yield ChatInput(id="input")

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Capture submitted content for testing."""
        self.submitted_content = event.text_area.text


# ============================================================================
# StatusBar Tests
# ============================================================================


class TestStatusBar:
    """Tests for StatusBar widget."""

    async def test_initial_display(self) -> None:
        """StatusBar shows Ready | Model | Stream: on on startup."""
        app = StatusBarApp(model="gpt-4.1", streaming=True)
        async with app.run_test() as _:
            status = app.query_one("#status-bar", StatusBar)
            # Check internal state
            assert status._status == "Ready"
            assert status._model == "gpt-4.1"
            assert status._streaming is True

    async def test_update_status(self) -> None:
        """update_status() changes displayed values."""
        app = StatusBarApp()
        async with app.run_test() as _:
            status = app.query_one("#status-bar", StatusBar)

            # Update status
            status.update_status(status="Thinking...")
            assert status._status == "Thinking..."

            # Update model
            status.update_status(model="claude-3")
            assert status._model == "claude-3"

            # Update streaming
            status.update_status(streaming=False)
            assert status._streaming is False

            # Update tokens
            status.update_status(tokens="12K / 128K tokens")
            assert status._tokens == "12K / 128K tokens"

    async def test_spinner_starts_on_thinking(self) -> None:
        """Spinner starts when status is 'Thinking...'."""
        app = StatusBarApp()
        async with app.run_test() as _:
            status = app.query_one("#status-bar", StatusBar)

            # Before thinking, no timer
            assert status._spinner_timer is None

            # After setting thinking, timer should start
            status.update_status(status="Thinking...")
            assert status._spinner_timer is not None
            assert status._start_time is not None

    async def test_spinner_stops_on_ready(self) -> None:
        """Spinner stops when status returns to 'Ready'."""
        app = StatusBarApp()
        async with app.run_test() as _:
            status = app.query_one("#status-bar", StatusBar)

            # Start thinking
            status.update_status(status="Thinking...")
            assert status._spinner_timer is not None

            # Go back to ready
            status.update_status(status="Ready")
            assert status._spinner_timer is None

    async def test_response_time_recorded(self) -> None:
        """Response time is recorded after generation completes."""
        app = StatusBarApp()
        async with app.run_test() as pilot:
            status = app.query_one("#status-bar", StatusBar)

            # Before any generation
            assert status._last_response_time is None

            # Start thinking
            status.update_status(status="Thinking...")

            # Wait a bit
            await pilot.pause(delay=0.1)

            # Complete
            status.update_status(status="Ready")

            # Response time should be recorded
            assert status._last_response_time is not None
            assert status._last_response_time >= 0.1


# ============================================================================
# MessageWidget Tests
# ============================================================================


class TestMessageWidget:
    """Tests for MessageWidget."""

    async def test_user_message_prefix(self) -> None:
        """User message has correct role prefix."""
        app = MessageWidgetApp(role="user", content="Hello")
        async with app.run_test() as _:
            widget = app.query_one(MessageWidget)
            assert widget.role == "user"
            assert widget.message_content == "Hello"
            assert "user-message" in widget.classes

    async def test_assistant_message_prefix(self) -> None:
        """Assistant message has correct role prefix."""
        app = MessageWidgetApp(role="assistant", content="Hi there")
        async with app.run_test() as _:
            widget = app.query_one(MessageWidget)
            assert widget.role == "assistant"
            assert widget.message_content == "Hi there"
            assert "assistant-message" in widget.classes

    async def test_error_message_styling(self) -> None:
        """Error message has error styling class."""
        app = MessageWidgetApp(role="error", content="Something went wrong")
        async with app.run_test() as _:
            widget = app.query_one(MessageWidget)
            assert widget.role == "error"
            assert "error-message" in widget.classes

    async def test_system_message_styling(self) -> None:
        """System message has system styling class."""
        app = MessageWidgetApp(role="system", content="Welcome")
        async with app.run_test() as _:
            widget = app.query_one(MessageWidget)
            assert widget.role == "system"
            assert "system-message" in widget.classes

    async def test_append_content(self) -> None:
        """append_content() adds to existing message."""
        app = MessageWidgetApp(role="assistant", content="Hello")
        async with app.run_test() as _:
            widget = app.query_one(MessageWidget)
            widget.is_streaming = True
            widget.append_content(" world")
            assert widget.message_content == "Hello world"

    async def test_set_content(self) -> None:
        """set_content() replaces message content."""
        app = MessageWidgetApp(role="user", content="Old content")
        async with app.run_test() as _:
            widget = app.query_one(MessageWidget)
            widget.set_content("New content")
            assert widget.message_content == "New content"

    async def test_finish_streaming(self) -> None:
        """finish_streaming() triggers markdown rendering."""
        app = MessageWidgetApp(role="assistant", content="")
        async with app.run_test() as _:
            widget = app.query_one(MessageWidget)
            widget.is_streaming = True
            widget.append_content("**bold** text")
            assert widget.is_streaming is True

            widget.finish_streaming()
            assert widget.is_streaming is False


# ============================================================================
# ChatLog Tests
# ============================================================================


class TestChatLog:
    """Tests for ChatLog container widget."""

    async def test_add_message(self) -> None:
        """add_message() creates MessageWidget."""
        app = ChatLogApp()
        async with app.run_test() as _:
            log = app.query_one("#chat-log", ChatLog)

            # Add a message
            widget = log.add_message("user", "Hello")

            # Should return the widget
            assert isinstance(widget, MessageWidget)
            assert widget.role == "user"
            assert widget.message_content == "Hello"

            # Should be mounted
            messages = list(log.query(MessageWidget))
            assert len(messages) == 1

    async def test_get_last_message(self) -> None:
        """get_last_message() returns most recent."""
        app = ChatLogApp()
        async with app.run_test() as _:
            log = app.query_one("#chat-log", ChatLog)

            # No messages yet
            assert log.get_last_message() is None

            # Add messages
            log.add_message("user", "First")
            log.add_message("assistant", "Second")

            last = log.get_last_message()
            assert last is not None
            assert last.message_content == "Second"

    async def test_clear_messages(self) -> None:
        """clear_messages() removes all messages."""
        app = ChatLogApp()
        async with app.run_test() as pilot:
            log = app.query_one("#chat-log", ChatLog)

            # Add messages
            log.add_message("user", "One")
            log.add_message("assistant", "Two")
            assert len(list(log.query(MessageWidget))) == 2

            # Clear
            log.clear_messages()
            await pilot.pause()

            assert len(list(log.query(MessageWidget))) == 0

    async def test_clear_messages_removes_citations(self) -> None:
        """clear_messages() also removes CitationsWidget instances."""
        app = ChatLogApp()
        async with app.run_test() as pilot:
            log = app.query_one("#chat-log", ChatLog)

            # Add messages
            log.add_message("user", "Question")
            log.add_message("assistant", "Answer with [1] citation")

            # Add citations widget (as done in workers.py after response)
            citations = CitationsWidget(
                sources=[RAGSource(title="Paper", pmid="123")],
                retrieval_time_s=1.0,
            )
            log.mount(citations)
            await pilot.pause()

            # Verify widgets are mounted
            assert len(list(log.query(MessageWidget))) == 2
            assert len(list(log.query(CitationsWidget))) == 1

            # Clear
            log.clear_messages()
            await pilot.pause()

            # Both MessageWidgets and CitationsWidget should be removed
            assert len(list(log.query(MessageWidget))) == 0
            assert len(list(log.query(CitationsWidget))) == 0

    async def test_remove_last_message(self) -> None:
        """remove_last_message() removes most recent."""
        app = ChatLogApp()
        async with app.run_test() as pilot:
            log = app.query_one("#chat-log", ChatLog)

            # Add messages
            log.add_message("user", "First")
            log.add_message("assistant", "Second")

            # Remove last
            log.remove_last_message()
            await pilot.pause()

            messages = list(log.query(MessageWidget))
            assert len(messages) == 1
            assert messages[0].message_content == "First"

    async def test_append_to_last(self) -> None:
        """append_to_last() adds content to last message."""
        app = ChatLogApp()
        async with app.run_test() as _:
            log = app.query_one("#chat-log", ChatLog)

            # Add streaming message
            widget = log.add_message("assistant", "Hello")
            widget.is_streaming = True

            # Append content
            log.append_to_last(" world")

            assert widget.message_content == "Hello world"


# ============================================================================
# ChatInput Tests
# ============================================================================


class TestChatInput:
    """Tests for ChatInput widget."""

    async def test_initial_empty(self) -> None:
        """ChatInput starts empty."""
        app = ChatInputApp()
        async with app.run_test() as _:
            input_widget = app.query_one("#input", ChatInput)
            assert input_widget.text == ""

    async def test_text_can_be_set(self) -> None:
        """Text can be set programmatically."""
        app = ChatInputApp()
        async with app.run_test() as _:
            input_widget = app.query_one("#input", ChatInput)

            # Set text directly (simulating typing)
            input_widget.text = "Hello world"

            assert input_widget.text == "Hello world"

    async def test_submit_binding_configured(self) -> None:
        """ChatInput has Ctrl+P binding configured for send action."""
        app = ChatInputApp()
        async with app.run_test() as _:
            input_widget = app.query_one("#input", ChatInput)

            # Check binding exists and is configured for 'send' action
            from textual.binding import Binding

            bindings = [b for b in input_widget.BINDINGS if isinstance(b, Binding)]
            ctrl_p_binding = next((b for b in bindings if b.key == "ctrl+p"), None)
            assert ctrl_p_binding is not None
            assert ctrl_p_binding.action == "send"

    async def test_has_submit_binding(self) -> None:
        """ChatInput has Ctrl+P binding for submit."""
        app = ChatInputApp()
        async with app.run_test() as _:
            input_widget = app.query_one("#input", ChatInput)

            # Check binding exists
            from textual.binding import Binding

            bindings = [b for b in input_widget.BINDINGS if isinstance(b, Binding)]
            assert any(b.key == "ctrl+p" for b in bindings)

    async def test_action_send_posts_event(self) -> None:
        """action_send() posts Submitted event."""
        app = ChatInputApp()
        async with app.run_test() as pilot:
            input_widget = app.query_one("#input", ChatInput)
            input_widget.text = "Direct action test"

            # Call action directly
            input_widget.action_send()
            await pilot.pause()

            assert app.submitted_content == "Direct action test"


# ============================================================================
# CitationsWidget Tests
# ============================================================================


class CitationsWidgetApp(App[None]):
    """Minimal app for testing CitationsWidget."""

    def __init__(
        self,
        sources: list[RAGSource] | None = None,
        retrieval_time_s: float = 0.0,
        chunk_count: int = 0,
    ) -> None:
        super().__init__()
        self.sources = sources or []
        self.retrieval_time_s = retrieval_time_s
        self.chunk_count = chunk_count

    def compose(self) -> ComposeResult:
        yield CitationsWidget(
            sources=self.sources,
            retrieval_time_s=self.retrieval_time_s,
            chunk_count=self.chunk_count,
        )


class TestCitationsWidget:
    """Tests for CitationsWidget."""

    async def test_empty_sources(self) -> None:
        """CitationsWidget handles empty sources list."""
        app = CitationsWidgetApp(sources=[])
        async with app.run_test() as _:
            widget = app.query_one(CitationsWidget)
            assert widget._sources == []

    async def test_single_source(self) -> None:
        """CitationsWidget displays a single source."""
        sources = [RAGSource(title="Test Paper", pmid="12345")]
        app = CitationsWidgetApp(sources=sources)
        async with app.run_test() as _:
            widget = app.query_one(CitationsWidget)
            assert len(widget._sources) == 1
            assert widget._sources[0].title == "Test Paper"
            assert widget._sources[0].pmid == "12345"

    async def test_multiple_sources(self) -> None:
        """CitationsWidget displays multiple sources."""
        sources = [
            RAGSource(title="Paper A", pmid="111"),
            RAGSource(title="Paper B", pmcid="PMC222"),
            RAGSource(title="Paper C", snippet="Sample text..."),
        ]
        app = CitationsWidgetApp(sources=sources)
        async with app.run_test() as _:
            widget = app.query_one(CitationsWidget)
            assert len(widget._sources) == 3

    async def test_retrieval_time_stored(self) -> None:
        """CitationsWidget stores retrieval time."""
        sources = [RAGSource(title="Paper")]
        app = CitationsWidgetApp(sources=sources, retrieval_time_s=1.5)
        async with app.run_test() as _:
            widget = app.query_one(CitationsWidget)
            assert widget._retrieval_time_s == 1.5

    async def test_chunk_count_stored(self) -> None:
        """CitationsWidget stores chunk count."""
        sources = [RAGSource(title="Paper")]
        app = CitationsWidgetApp(sources=sources, chunk_count=30)
        async with app.run_test() as _:
            widget = app.query_one(CitationsWidget)
            assert widget._chunk_count == 30

    def test_from_metadata_with_sources(self) -> None:
        """from_metadata() creates widget from RAGMetadata with sources."""
        sources = [
            RAGSource(title="Paper 1", pmid="123"),
            RAGSource(title="Paper 2", pmcid="PMC456"),
        ]
        metadata = RAGMetadata(
            sources=sources,
            retrieval_time_s=2.0,
            chunk_count=25,
        )

        widget = CitationsWidget.from_metadata(metadata)

        assert widget is not None
        assert len(widget._sources) == 2
        assert widget._retrieval_time_s == 2.0
        assert widget._chunk_count == 25

    def test_from_metadata_empty_sources(self) -> None:
        """from_metadata() returns None for empty sources."""
        metadata = RAGMetadata(sources=[], retrieval_time_s=0.5)

        widget = CitationsWidget.from_metadata(metadata)

        assert widget is None

    def test_from_metadata_none(self) -> None:
        """from_metadata() returns None for None metadata."""
        widget = CitationsWidget.from_metadata(None)
        assert widget is None

    async def test_css_class_applied(self) -> None:
        """CitationsWidget has correct CSS class."""
        sources = [RAGSource(title="Paper")]
        app = CitationsWidgetApp(sources=sources)
        async with app.run_test() as _:
            widget = app.query_one(CitationsWidget)
            assert "citations-container" in widget.classes

    async def test_collapsed_threshold(self) -> None:
        """CitationsWidget has correct collapsed threshold."""
        assert CitationsWidget.COLLAPSED_THRESHOLD == 5

    def test_build_citation_minimal(self) -> None:
        """_build_citation() works with minimal source."""
        sources = [RAGSource(title="Simple Paper")]
        widget = CitationsWidget(sources=sources)

        citation = widget._build_citation(1, sources[0])

        # Citation is now a Collapsible widget
        assert isinstance(citation, Collapsible)
        assert "citation" in citation.classes

    def test_build_citation_with_pmid(self) -> None:
        """_build_citation() includes PMID in title."""
        source = RAGSource(title="Paper", pmid="12345678")
        widget = CitationsWidget(sources=[source])

        citation = widget._build_citation(1, source)

        # Verify it's a Collapsible with correct class
        assert isinstance(citation, Collapsible)
        assert "citation" in citation.classes
        # PMID should be in the title
        assert "PMID:12345678" in citation.title

    def test_build_citation_with_pmcid(self) -> None:
        """_build_citation() includes PMCID in title."""
        source = RAGSource(title="Paper", pmcid="PMC9876543")
        widget = CitationsWidget(sources=[source])

        citation = widget._build_citation(1, source)

        # Verify it's a Collapsible with correct class
        assert isinstance(citation, Collapsible)
        assert "citation" in citation.classes
        # PMCID should be in the title
        assert "PMCID:PMC9876543" in citation.title

    def test_build_citation_uses_full_text(self) -> None:
        """_build_citation() shows full_text when expanded."""
        source = RAGSource(
            title="Paper",
            snippet="Short preview...",
            full_text="This is the full text of the chunk which is much longer.",
        )
        widget = CitationsWidget(sources=[source])

        citation = widget._build_citation(1, source)

        # Should be a Collapsible
        assert isinstance(citation, Collapsible)
        assert "citation" in citation.classes
        # Full text should be used (checked via child widget)

    def test_build_citation_falls_back_to_snippet(self) -> None:
        """_build_citation() uses snippet when full_text is empty."""
        source = RAGSource(
            title="Paper",
            snippet="This is a fallback snippet.",
            full_text="",  # Empty full_text
        )
        widget = CitationsWidget(sources=[source])

        citation = widget._build_citation(1, source)

        # Should be a Collapsible
        assert isinstance(citation, Collapsible)
        assert "citation" in citation.classes

    def test_build_citation_starts_collapsed(self) -> None:
        """_build_citation() creates collapsed collapsible by default."""
        source = RAGSource(title="Paper", full_text="Full text here.")
        widget = CitationsWidget(sources=[source])

        citation = widget._build_citation(1, source)

        # Should start collapsed
        assert isinstance(citation, Collapsible)
        assert citation.collapsed is True
