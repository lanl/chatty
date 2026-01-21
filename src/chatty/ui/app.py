"""Textual application for chatty.

This module implements the main ChatApp class that coordinates the UI
components and handles user interactions.

Module Structure
----------------
The UI is split across several modules for maintainability:
- app.py (this file): ChatApp class, bindings, and main entry point
- widgets.py: ChatInput, MessageWidget, ChatLog, StatusBar
- modals.py: FileInputModal, SessionBrowserModal, ModelPickerModal, ModelInputModal
- app.tcss: Stylesheet for widget layout and colors

Widget Hierarchy
----------------
    ChatApp (main application)
    ├── Header (Textual built-in - shows app title)
    ├── Container
    │   └── ChatLog (scrollable message display with markdown)
    ├── Vertical
    │   └── ChatInput (multi-line text entry)
    ├── StatusBar (model, tokens, connection status)
    └── ChattyFooter (custom footer with controlled keybinding display)

Data Flow
---------
    1. User types message in ChatInput widget
    2. Ctrl+P triggers action_submit()
    3. Message added to Conversation state
    4. RAGProvider.augment() called (NullProvider passthrough in v0.1-0.2)
    5. OpenAIClient.chat() called with streaming
    6. Tokens streamed to ChatLog via worker
    7. On completion, Conversation updated with full response
    8. StatusBar updated with token count

Keyboard Shortcuts
------------------
    Visible in footer:
    Ctrl+P      : Submit query ("P for Prompt")
    Ctrl+O      : Load query from file
    Escape      : Cancel current generation
    Ctrl+C      : Copy last response
    Ctrl+S      : Save session
    Ctrl+L      : Load/browse sessions
    Ctrl+N      : New session (clear history)
    Ctrl+Q      : Quit (clean shutdown)

    Hidden (power user):
    Ctrl+E      : Export to Markdown
    Ctrl+R      : Regenerate last response
    Ctrl+T      : Toggle streaming mode on/off
    Ctrl+G      : Model picker
    Enter       : Insert newline (multi-line input)

Integration Points
------------------
    - OpenAIClient: Async chat completions with retry
    - Conversation: Message history and token tracking
    - RAGProvider: Context augmentation (NullProvider for v0.1-0.2)
    - Config: Model settings, streaming default, system prompt
    - Session: Save/load conversation state
    - TranscriptLogger: JSONL audit trail

Key Design Decisions
--------------------
    - CSS in separate .tcss file for maintainability
    - Widgets and modals in separate modules (widgets.py, modals.py)
    - Async streaming via Textual workers to keep UI responsive
    - Cancellation via asyncio task cancellation
    - Error display inline as styled message cards
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Header
from textual.worker import Worker

from chatty.client.openai_client import Message, OpenAIClient
from chatty.config import ConfigWithSources, find_config_path, load_config
from chatty.core.conversation import Conversation
from chatty.core.session import (
    Session,
    generate_session_name,
    save_markdown_export,
)
from chatty.core.transcript import TranscriptLogger
from chatty.rag import ConfigurationError, NullProvider, RAGMetadata, RAGProvider, get_provider
from chatty.ui.clipboard import copy_to_clipboard
from chatty.ui.footer import ChattyFooter
from chatty.ui.modals import (
    CompressionPreviewModal,
    FileInputModal,
    SessionBrowserModal,
    SessionRenameModal,
    UnsavedChangesModal,
)
from chatty.ui.search import SearchBar
from chatty.ui.session_handlers import (
    handle_file_path,
    handle_first_save,
    handle_session_load,
    load_and_submit_query_file,
    load_session_file,
    save_current_session,
)
from chatty.ui.widgets import ChatInput, ChatLog, StatusBar
from chatty.ui.workers import (
    fetch_and_show_models,
    fetch_context_window,
    send_message,
)

if TYPE_CHECKING:
    from chatty.rag import RAGMetadata


class ChatApp(App[None]):
    """Chatty terminal UI application.

    The main Textual application class that coordinates the UI components
    and handles user interactions. Manages the chat workflow:

    1. Accept user input
    2. Augment with RAG context (if enabled)
    3. Send to LLM
    4. Stream response to display
    5. Track conversation state

    Attributes:
        query_file: Optional path to file containing initial query.
        session_file: Optional path to session file to restore.
        streaming: Whether to stream responses (can be toggled at runtime).
        config_with_sources: Full configuration with source tracking.
        config: Application configuration.
        client: OpenAI client instance.
        conversation: Current conversation state.
        current_worker: Currently running async worker (for cancellation).
        transcript: JSONL transcript logger.
        rag_provider: RAG provider for context augmentation.

    Class Attributes:
        TITLE: Application title shown in header.
        CSS_PATH: Path to external stylesheet.
        BINDINGS: Keyboard shortcut definitions.
    """

    TITLE = "chatty"
    CSS_PATH = "app.tcss"

    # Disable Textual's command palette (we use Ctrl+P for submit)
    ENABLE_COMMAND_PALETTE = False

    # Keyboard bindings - ChattyFooter controls display order
    # Hidden bindings (show=False) are functional but not shown in footer
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "copy", "Copy", priority=True),
        Binding("ctrl+s", "save", "Save Session"),
        Binding("ctrl+l", "browse_sessions", "Load Session"),
        Binding("ctrl+n", "new_session", "New Session"),
        Binding("ctrl+o", "load_file", "Load File"),
        Binding("escape", "cancel", "Interrupt"),
        Binding("ctrl+f", "search", "Find", show=False, priority=True),
        Binding("ctrl+h", "help", "Help", priority=True),
        Binding("f1", "help", "Help", show=False),
        # Power user shortcuts (hidden from footer)
        Binding("ctrl+r", "regenerate", "Regenerate", show=False),
        Binding("ctrl+g", "pick_model", "Models", show=False),
        Binding("ctrl+e", "export", "Export", show=False),
        Binding("ctrl+t", "toggle_stream", "Toggle Stream", show=False),
        Binding("ctrl+j", "compress", "Compress"),
        Binding("ctrl+y", "undo_compress", "Undo Compress", show=False, priority=True),
    ]

    def __init__(
        self,
        query_file: str | None = None,
        session_file: Path | None = None,
        config_with_sources: ConfigWithSources | None = None,
    ) -> None:
        """Initialize the chat application.

        Args:
            query_file: Optional path to file containing initial query to load.
            session_file: Optional path to saved session file to restore.
            config_with_sources: Pre-loaded configuration (loads default if None).
        """
        super().__init__()
        self.query_file = query_file
        self.session_file = session_file
        self.config_with_sources = config_with_sources or load_config()
        self.config = self.config_with_sources.config
        self.streaming = self.config.stream
        self.client: OpenAIClient | None = None
        self.conversation: Conversation | None = None
        self.current_worker: Worker[None] | None = None
        self.transcript: TranscriptLogger = TranscriptLogger(self.config)
        self.rag_provider: RAGProvider
        self._rag_init_error: str | None = None
        try:
            self.rag_provider = get_provider(self.config)
        except ConfigurationError as e:
            # Fall back to NullProvider and store error for display
            self.rag_provider = NullProvider()
            self._rag_init_error = str(e)
        self.last_rag_metadata: RAGMetadata | None = None  # For future citation display
        self._pending_user_text: str | None = None  # User message awaiting LLM response
        self._current_session: Session | None = None  # For save/load functionality
        self._current_model: str = self.config.model  # Runtime model (can be changed)
        self._session_dirty: bool = False  # Track unsaved changes
        # Compression state
        self._pre_compression_state: list[Message] | None = None
        self._compression_available: bool = False
        self._pending_summary: str | None = None
        # Pending load path for unsaved changes flow
        self._pending_load_path: Path | None = None

    def compose(self) -> ComposeResult:
        """Create the UI layout.

        Yields widgets in top-to-bottom order. The layout is:
        - Header: Shows app title
        - SearchBar: Inline search bar (hidden by default)
        - ChatLog: Scrollable area for messages (extends VerticalScroll)
        - Container with ChatInput: User text input area (multi-line)
        - StatusBar: Single-line status display
        - ChattyFooter: Custom footer with controlled keybinding display
        """
        yield Header()
        yield SearchBar(id="search-bar")
        yield ChatLog(id="chat-log")
        yield Container(
            ChatInput(id="input"),
            id="input-container",
        )
        yield StatusBar(
            id="status-bar",
            model=self.config.model,
            streaming=self.streaming,
            rag_provider=self.config.rag_provider,
        )
        yield ChattyFooter(id="footer")

    def on_mount(self) -> None:
        """Handle app mount event.

        Called when the app is fully loaded and ready. Sets up:
        - Initial focus on input widget
        - Check for missing configuration (show warnings)
        - Create OpenAI client
        - Initialize conversation state
        - Resolve context window (from config or endpoint)
        - Start transcript logging if enabled
        - Add system prompt if configured
        - Load session or query file if provided
        """
        self.query_one("#input", ChatInput).focus()
        chat_log = self.query_one("#chat-log", ChatLog)

        # Check for missing configuration and warn user
        self._check_startup_config(chat_log)

        # Show RAG initialization error if any
        if self._rag_init_error:
            chat_log.add_message(
                "error",
                f"⚠ RAG provider failed to initialize:\n\n{self._rag_init_error}\n\n"
                "Chat will work without RAG. Fix the issue or set `rag_provider = 'none'`.",
            )

        # Initialize client and conversation
        self.client = OpenAIClient(self.config)
        self.conversation = Conversation(model=self.config.model)

        # Resolve context window (may require async call to endpoint)
        self._resolve_context_window()

        # Start transcript logging
        transcript_file = self.transcript.start_session()
        if transcript_file:
            chat_log.add_message("system", f"Transcript: {transcript_file}")

        # Add system prompt if configured
        if self.config.system_prompt:
            self.conversation.add_system_message(self.config.system_prompt)
            self.transcript.log_message("system", self.config.system_prompt)

        # Update status bar with actual model
        self.query_one("#status-bar", StatusBar).update_status(model=self.config.model)

        # Load session from file if provided (takes precedence over query file)
        if self.session_file:
            self._load_session_file()
        elif self.query_file:
            self._load_and_submit_query_file()

    def _resolve_context_window(self) -> None:
        """Resolve context window from config or endpoint.

        Priority:
        1. If config.context_window is an integer, use it directly
        2. If config.context_window is "auto", fetch from /models endpoint
        3. If "auto" and endpoint doesn't provide it, show error

        This method starts an async worker to handle endpoint calls.
        """
        if isinstance(self.config.context_window, int):
            # Explicit value configured - use it
            if self.conversation:
                self.conversation.set_context_window(self.config.context_window)
        else:
            # "auto" - need to fetch from endpoint
            # Textual's run_worker type hints are overly restrictive
            self.run_worker(
                self._fetch_context_window,  # type: ignore[arg-type]
                exclusive=False,
                name="fetch_context_window",
            )

    async def _fetch_context_window(self) -> None:
        """Fetch context window from /models endpoint."""
        await fetch_context_window(self)

    def _check_startup_config(self, chat_log: ChatLog) -> None:
        """Check configuration at startup and show warnings for issues.

        Args:
            chat_log: The chat log widget to display warnings.
        """
        config_path = find_config_path()
        warnings = []

        # Check if no config file found
        if config_path is None:
            warnings.append(
                "⚠ No config file found.\n\n"
                "Searched:\n"
                "  ./chatty.toml\n"
                "  ~/.config/chatty/config.toml\n\n"
                "To fix:\n"
                "  • Run from repo root: cd ~/Code/chatty\n"
                "  • Or set: export CHATTY_CONFIG='/path/to/chatty.toml'\n"
                "  • Or create: ~/.config/chatty/config.toml\n\n"
                "Run 'chatty doctor' to diagnose."
            )

        # Check if base_url is empty
        if not self.config.base_url:
            warnings.append(
                "⚠ No base_url configured. Chat will fail.\n\n"
                "To fix, add to your config file:\n"
                '  base_url = "http://localhost:1234/v1"'
            )

        # Show warnings
        for warning in warnings:
            chat_log.add_message("error", warning)

    def _load_session_file(self) -> None:
        """Load a saved session from file."""
        load_session_file(self)

    def _load_and_submit_query_file(self) -> None:
        """Load query from file and submit it."""
        load_and_submit_query_file(self)

    def on_chat_input_submitted(self, _event: ChatInput.Submitted) -> None:
        """Handle Ctrl+P from ChatInput widget."""
        self.action_submit()

    def action_submit(self) -> None:
        """Submit the current message (Ctrl+P).

        This is the main chat workflow entry point:
        1. Validate input (non-empty)
        2. Add user message to chat log
        3. Clear input for next message
        4. Store pending user text for RAG provider
        5. Start async worker for LLM call
        6. Worker streams response to chat log
        """
        input_widget = self.query_one("#input", ChatInput)
        user_message = input_widget.text.strip()

        if not user_message:
            return

        chat_log = self.query_one("#chat-log", ChatLog)

        # Display user message
        chat_log.add_message("user", user_message)

        # Clear input for next message
        input_widget.text = ""

        # Store pending user text - will be added to conversation after response
        # This allows RAGProvider.augment() to receive conversation history
        # separately from the new user message for query rewriting
        self._pending_user_text = user_message
        self.transcript.log_message("user", user_message)

        # Start async worker for LLM call
        # Pass method reference (not called) — Textual invokes it
        self.current_worker = self.run_worker(
            self._send_message,
            exclusive=True,
            name="send_message",
        )

    async def _send_message(self) -> None:
        """Worker function for async LLM call."""
        await send_message(self)

    def action_toggle_stream(self) -> None:
        """Toggle streaming mode on/off (Ctrl+T).

        When streaming is on, tokens are displayed as they arrive.
        When off, the complete response is displayed at once.
        """
        self.streaming = not self.streaming
        self.query_one("#status-bar", StatusBar).update_status(streaming=self.streaming)

    def action_regenerate(self) -> None:
        """Regenerate the last assistant response (Ctrl+R).

        Removes the last assistant message from conversation and chat log,
        then re-sends the previous user message to get a new response.
        Useful when the response was unsatisfactory.
        """
        if not self.conversation or len(self.conversation.messages) < 2:
            return

        # Check if the last message is from assistant
        if self.conversation.messages[-1].role != "assistant":
            return

        # Get the user message that preceded the assistant response
        # (it's the second-to-last message)
        if self.conversation.messages[-2].role != "user":
            return

        user_text = self.conversation.messages[-2].content

        # Remove the last assistant message AND the user message from conversation
        # (they will be re-added after successful response)
        self.conversation.messages.pop()  # Remove assistant
        self.conversation.messages.pop()  # Remove user

        # Remove assistant message from chat log display
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.remove_last_message()

        # Set pending user text for _send_message
        self._pending_user_text = user_text

        # Re-send
        self.current_worker = self.run_worker(
            self._send_message,
            exclusive=True,
            name="regenerate",
        )

    def action_browse_sessions(self) -> None:
        """Browse and load saved sessions (Ctrl+L).

        Opens a modal dialog showing all saved sessions.
        User can select a session to load.
        """
        session_dir = self.config.get_session_path()
        self.push_screen(
            SessionBrowserModal(session_dir),
            self._handle_session_load,
        )

    def _handle_session_load(self, filepath: Path | None) -> None:
        """Handle the session file selected from browser.

        If there are unsaved changes, shows a warning dialog before loading.

        Args:
            filepath: Path to session file, or None if cancelled.
        """
        if filepath is None:
            return  # User cancelled

        if self._session_dirty:
            # Store the path and show warning
            self._pending_load_path = filepath
            self.push_screen(
                UnsavedChangesModal(),
                self._handle_unsaved_warning,
            )
        else:
            # No unsaved changes, load directly
            handle_session_load(self, filepath)

    def _handle_unsaved_warning(self, choice: str | None) -> None:
        """Handle user choice from unsaved changes modal.

        Args:
            choice: "save_and_load", "load", or None (cancelled).
        """
        pending_path = getattr(self, "_pending_load_path", None)

        if choice == "save_and_load":
            # If no current session, prompt for name first
            if not self._current_session:
                # Don't clear pending_path here - _save_then_load will use it
                self.push_screen(
                    SessionRenameModal(
                        generate_session_name(self.conversation.messages)
                        if self.conversation
                        else "Untitled",
                        button_label="Save",
                    ),
                    self._save_then_load,
                )
                return  # Don't clear pending_path yet
            else:
                save_current_session(self)
                if pending_path:
                    handle_session_load(self, pending_path)
        elif choice == "load" and pending_path:
            # Load without saving
            handle_session_load(self, pending_path)
        # else: None = cancelled, do nothing

        # Clear pending path (only for immediate actions, not when another modal is pushed)
        self._pending_load_path = None

    def _save_then_load(self, name: str | None) -> None:
        """Save with given name, then load the pending session.

        Args:
            name: Session name from rename modal, or None if cancelled.
        """
        pending_path = getattr(self, "_pending_load_path", None)
        if name:
            handle_first_save(self, name)
        if pending_path:
            handle_session_load(self, pending_path)
        self._pending_load_path = None

    def action_load_file(self) -> None:
        """Load a query from a file (Ctrl+O).

        Opens a modal dialog for the user to enter a file path.
        The file content is loaded into the input area (not auto-submitted).
        """
        self.push_screen(FileInputModal(), self._handle_file_path)

    def _handle_file_path(self, path: str | None) -> None:
        """Handle the file path returned from the modal.

        Args:
            path: File path entered by user, or None if cancelled.
        """
        handle_file_path(self, path)

    def action_save(self) -> None:
        """Save the current session to a file (Ctrl+S).

        On first save, prompts for a session name with an auto-generated default.
        On subsequent saves, updates the existing session file.
        """
        if not self.conversation or not self.conversation.messages:
            chat_log = self.query_one("#chat-log", ChatLog)
            chat_log.add_message("system", "Nothing to save — conversation is empty.")
            return

        if self._current_session is None:
            # First save - prompt for name
            default_name = generate_session_name(self.conversation.messages)
            self.push_screen(
                SessionRenameModal(default_name, button_label="Save"),
                self._handle_first_save,
            )
        else:
            # Update existing session and save
            self._save_current_session()

    def _handle_first_save(self, name: str | None) -> None:
        """Handle the name returned from first-save modal.

        Args:
            name: Session name entered by user, or None if cancelled.
        """
        handle_first_save(self, name)

    def _save_current_session(self) -> None:
        """Save the current session to disk."""
        save_current_session(self)

    def action_new_session(self) -> None:
        """Start a new session, clearing all history (Ctrl+N).

        Clears the chat log and resets the conversation state.
        Token count is reset. Does not change configuration.
        """
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.clear_messages()

        if self.conversation:
            self.conversation.clear()
            # Re-add system prompt if configured
            if self.config.system_prompt:
                self.conversation.add_system_message(self.config.system_prompt)

        # Reset session tracking
        self._current_session = None
        self._session_dirty = False

        self.query_one("#status-bar", StatusBar).update_status(
            status="Ready",
            tokens="",
        )

    def action_cancel(self) -> None:
        """Cancel the current generation (Escape).

        Cancels any in-flight LLM request by cancelling the async task.
        Partial responses may remain visible in the chat log.
        Also closes search bar if open.
        """
        # Close search bar if visible
        search_bar = self.query_one("#search-bar", SearchBar)
        if search_bar.is_visible:
            search_bar.hide()
            self._close_search()
            return

        # Otherwise cancel generation
        if self.current_worker and self.current_worker.is_running:
            self.current_worker.cancel()
            self.query_one("#status-bar", StatusBar).update_status(status="Cancelled")
            self.current_worker = None

    def action_search(self) -> None:
        """Search: open search bar or advance to next match (Ctrl+F).

        - First Ctrl+F: Opens the search bar
        - Subsequent Ctrl+F: Advances to the next match
        - Esc: Closes the search bar
        """
        search_bar = self.query_one("#search-bar", SearchBar)
        if search_bar.is_visible:
            # Search already open → advance to next match
            chat_log = self.query_one("#chat-log", ChatLog)
            chat_log.next_match()
            occurrences = chat_log.total_occurrences(search_bar.search_query)
            search_bar.set_matches(chat_log.match_count, chat_log.current_match_index, occurrences)
        else:
            # Search not open → open it
            search_bar.show()

    def on_search_bar_query_changed(self, event: SearchBar.QueryChanged) -> None:
        """Handle search query changes."""
        chat_log = self.query_one("#chat-log", ChatLog)
        search_bar = self.query_one("#search-bar", SearchBar)

        matches = chat_log.search(event.query)
        occurrences = chat_log.total_occurrences(event.query)
        search_bar.set_matches(len(matches), 0, occurrences)

    def on_search_bar_next_match(self, _event: SearchBar.NextMatch) -> None:
        """Handle next match request."""
        chat_log = self.query_one("#chat-log", ChatLog)
        search_bar = self.query_one("#search-bar", SearchBar)

        chat_log.next_match()
        occurrences = chat_log.total_occurrences(search_bar.search_query)
        search_bar.set_matches(chat_log.match_count, chat_log.current_match_index, occurrences)

    def on_search_bar_closed(self, _event: SearchBar.Closed) -> None:
        """Handle search bar close."""
        self._close_search()

    def _close_search(self) -> None:
        """Clear search state and return focus to input."""
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.clear_search()
        self.query_one("#input", ChatInput).focus()

    def action_copy(self) -> None:
        """Copy the last assistant response (Ctrl+C).

        Copies to system clipboard. If clipboard is unavailable (headless HPC),
        falls back to writing to configured copy_fallback_path.
        """
        # Find last assistant message
        if not self.conversation or not self.conversation.messages:
            self.notify("Nothing to copy — no messages yet.", severity="warning")
            return

        # Get last assistant message
        last_assistant = None
        for msg in reversed(self.conversation.messages):
            if msg.role == "assistant":
                last_assistant = msg.content
                break

        if not last_assistant:
            self.notify("No assistant response to copy.", severity="warning")
            return

        # Try clipboard first, fall back to file
        result = self._copy_to_clipboard(last_assistant)
        if "Clipboard unavailable" in result:
            self.notify(result, severity="warning", timeout=5)
        else:
            self.notify(result, severity="information", timeout=2)

    def _copy_to_clipboard(self, text: str) -> str:
        """Copy text to clipboard with fallback to file.

        Delegates to clipboard module for actual implementation.

        Args:
            text: Text to copy.

        Returns:
            Status message describing what happened.
        """
        return copy_to_clipboard(text, self.config.get_copy_fallback_path())

    def action_pick_model(self) -> None:
        """Open the model picker (Ctrl+G).

        Fetches available models from endpoint and shows picker.
        Falls back to manual input if /models not supported.
        """
        # Run model fetch in background worker to avoid blocking UI
        self._start_model_fetch_worker()

    def _start_model_fetch_worker(self) -> None:
        """Start worker to fetch models (separate method for type checking)."""
        # Textual's run_worker type hints are overly restrictive for async methods
        # returning None (valid usage per Textual docs), so we isolate the call
        self.run_worker(
            self._fetch_and_show_models,  # type: ignore[arg-type]
            exclusive=False,
            name="fetch_models",
        )

    async def _fetch_and_show_models(self) -> None:
        """Fetch models from endpoint and show picker modal."""
        await fetch_and_show_models(self)

    def _handle_model_selection(self, model: str | None) -> None:
        """Handle the model selected from picker or input.

        Args:
            model: Selected model name, or None if cancelled.
        """
        if not model or model == self._current_model:
            return

        # Update runtime model
        old_model = self._current_model
        self._current_model = model

        # Also update config.model so client uses new model
        # This is a runtime-only change; config file is not modified
        self.config.model = model

        # Update status bar
        self.query_one("#status-bar", StatusBar).update_status(model=model)

        # Show confirmation
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.add_message("system", f"Switched model: {old_model} → {model}")

    def action_export(self) -> None:
        """Export the current conversation to Markdown (Ctrl+E).

        Saves the conversation as a readable Markdown file in the
        configured export_path directory.
        """
        if not self.conversation or not self.conversation.messages:
            self.notify("Nothing to export — conversation is empty.", severity="warning")
            return

        chat_log = self.query_one("#chat-log", ChatLog)

        # Get session name if available
        session_name = None
        if self._current_session:
            session_name = self._current_session.metadata.name
        else:
            session_name = generate_session_name(self.conversation.messages)

        try:
            export_dir = self.config.get_export_path()
            filepath = save_markdown_export(
                self.conversation.messages,
                export_dir,
                session_name=session_name,
                model=self._current_model,
            )
            chat_log.add_message("system", f"Exported to: {filepath}")
        except Exception as e:
            chat_log.add_message("error", f"Failed to export: {e}")

    def action_help(self) -> None:
        """Show help modal (Ctrl+H or F1).

        Opens a modal dialog displaying keyboard shortcuts, current
        configuration, quick tips, and version information.
        """
        from chatty.ui.help import HelpModal

        modal = HelpModal()
        context_window = None
        if self.conversation:
            context_window = self.conversation.context_window
        modal.set_config(
            model=self._current_model,
            streaming=self.streaming,
            context_window=context_window,
            transcript_enabled=self.config.transcript_enabled,
        )
        self.push_screen(modal)

    def action_compress(self) -> None:
        """Compress conversation context (Ctrl+K).

        Opens a preview modal showing the proposed summary.
        User can apply or cancel.
        """
        # Check if streaming is in progress
        if self.current_worker and self.current_worker.is_running:
            self.notify(
                "Wait for response to complete before compressing.",
                severity="warning",
            )
            return

        # Check conversation has enough content
        if not self.conversation or len(self.conversation.messages) < 3:
            self.notify(
                "Nothing to compress — conversation too short.",
                severity="warning",
            )
            return

        # Start worker to generate summary
        self.run_worker(
            self._generate_compression_preview,  # type: ignore[arg-type]
            exclusive=True,
            name="compress",
        )

    async def _generate_compression_preview(self) -> None:
        """Generate compression preview and show modal."""
        from chatty.core.compression import (
            build_compression_prompt,
            detect_code_blocks,
            estimate_messages_tokens,
            estimate_tokens,
        )

        if not self.conversation or not self.client:
            return

        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.update_status(status="Compressing...")

        try:
            # Build compression prompt
            prompt = build_compression_prompt(self.conversation.messages)

            # Call LLM for summary (non-streaming)
            response = await self.client.chat(
                [Message(role="user", content=prompt)],
                stream=False,
            )

            # Response is AssistantMessage when stream=False
            if hasattr(response, "content"):
                summary = response.content
            else:
                raise RuntimeError("Unexpected streaming response")

            # Calculate token stats
            has_code = detect_code_blocks(self.conversation.messages)
            original_tokens = estimate_messages_tokens(
                self.conversation.messages, self._current_model
            )
            compressed_tokens = estimate_tokens(summary, self._current_model)

            # Store summary for later use
            self._pending_summary = summary

            # Show preview modal (async workers run on main thread, so call directly)
            self.push_screen(
                CompressionPreviewModal(
                    summary=summary,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    has_code_blocks=has_code,
                ),
                self._handle_compression_choice,
            )

        except Exception as e:
            self.notify(f"Compression failed: {e}", severity="error")
        finally:
            status_bar.update_status(status="Ready")

    def _handle_compression_choice(self, apply: bool | None) -> None:
        """Handle compression preview modal result.

        Args:
            apply: True to apply compression, None if cancelled.
        """
        if not apply:
            self._pending_summary = None
            return

        summary = self._pending_summary
        self._pending_summary = None

        if not summary or not self.conversation:
            return

        # Store pre-compression state for undo
        self._pre_compression_state = list(self.conversation.messages)
        self._compression_available = True

        # Replace conversation with summary
        self.conversation.clear()
        if self.config.system_prompt:
            self.conversation.add_system_message(self.config.system_prompt)
        self.conversation.add_system_message(f"[Conversation summary]\n{summary}")

        # Update chat log display
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.clear_messages()
        chat_log.add_message(
            "system",
            f"Context compressed. Press Ctrl+Y to undo.\n\n{summary}",
        )

        # Update status bar
        self.query_one("#status-bar", StatusBar).update_status(
            tokens=self.conversation.get_token_display(),
        )

        # Mark session as dirty
        self._session_dirty = True

    def action_undo_compress(self) -> None:
        """Undo last compression (Ctrl+U).

        Restores the conversation to its pre-compression state.
        Only works if compression was applied in this session.
        """
        if not self._compression_available or not self._pre_compression_state:
            self.notify(
                "Nothing to undo — no compression applied.",
                severity="warning",
            )
            return

        if not self.conversation:
            return

        # Restore pre-compression state
        self.conversation.messages = list(self._pre_compression_state)

        # Clear undo state (single level only)
        self._pre_compression_state = None
        self._compression_available = False

        # Rebuild chat log display
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.clear_messages()
        for msg in self.conversation.messages:
            if msg.role != "system":
                chat_log.add_message(msg.role, msg.content)

        # Update status bar
        self.query_one("#status-bar", StatusBar).update_status(
            status="Compression undone",
            tokens=self.conversation.get_token_display(),
        )

        chat_log.add_message("system", "Compression undone — original history restored.")

    async def on_unmount(self) -> None:
        """Clean up when app is closing."""
        self.transcript.close()
        if self.client:
            await self.client.close()


def main(
    query_file: str | None = None,
    session_file: Path | None = None,
    config_with_sources: ConfigWithSources | None = None,
) -> None:
    """Run the chat application.

    Entry point for the chatty UI. Creates and runs the Textual app.

    Args:
        query_file: Optional path to file containing initial query.
        session_file: Optional path to saved session file to restore.
        config_with_sources: Pre-loaded configuration (loads default if None).
    """
    app = ChatApp(
        query_file=query_file,
        session_file=session_file,
        config_with_sources=config_with_sources,
    )
    app.run()
