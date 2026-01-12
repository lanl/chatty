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
    └── Footer (Textual built-in - shows keybindings)

Data Flow
---------
    1. User types message in ChatInput widget
    2. Ctrl+E triggers action_submit()
    3. Message added to Conversation state
    4. RAGProvider.augment() called (NullProvider passthrough in v0.1-0.2)
    5. OpenAIClient.chat() called with streaming
    6. Tokens streamed to ChatLog via worker
    7. On completion, Conversation updated with full response
    8. StatusBar updated with token count

Keyboard Shortcuts
------------------
    Ctrl+Q      : Quit (clean shutdown)
    Ctrl+E      : Send message
    Ctrl+O      : Load query from file
    Ctrl+N      : New session (clear history)
    Ctrl+S      : Save session
    Ctrl+L      : Load/browse sessions
    Ctrl+Y      : Copy last response
    Ctrl+G      : Model picker
    Ctrl+B      : Export to Markdown
    Ctrl+R      : Regenerate last response
    Ctrl+T      : Toggle streaming mode on/off
    Enter       : Insert newline (multi-line input)
    Escape      : Cancel current generation

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

from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import Footer, Header
from textual.worker import Worker

from chatty.client.openai_client import ChattyClientError, OpenAIClient
from chatty.config import ConfigWithSources, find_config_path, load_config
from chatty.core.conversation import Conversation
from chatty.core.session import (
    Session,
    SessionMetadata,
    generate_session_name,
    load_session,
    save_markdown_export,
    save_session,
)
from chatty.core.transcript import TranscriptLogger
from chatty.rag import RAGMetadata, RAGProvider, get_provider
from chatty.ui.modals import (
    FileInputModal,
    ModelInputModal,
    ModelPickerModal,
    SessionBrowserModal,
)
from chatty.ui.widgets import ChatInput, ChatLog, StatusBar

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

    # Keyboard bindings - order determines display in footer
    # Note: ChatInput's ctrl+e shows first (focused widget), then quit has priority
    # Hidden bindings (show=False) are functional but not shown in footer
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+y", "copy", "Copy", priority=True),
        Binding("ctrl+s", "save", "Save Session"),
        Binding("ctrl+l", "browse_sessions", "Load Session"),
        Binding("ctrl+n", "new_session", "New Session"),
        Binding("ctrl+o", "load_file", "Load File"),
        Binding("escape", "cancel", "Interrupt"),
        # Power user shortcuts (visible in footer but may be truncated on small terminals)
        Binding("ctrl+r", "regenerate", "Regenerate"),
        Binding("ctrl+g", "pick_model", "Models"),
        Binding("ctrl+b", "export", "Export"),
        Binding("ctrl+t", "toggle_stream", "Toggle Stream", show=False),
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
        self.rag_provider: RAGProvider = get_provider(self.config)
        self.last_rag_metadata: RAGMetadata | None = None  # For future citation display
        self._pending_user_text: str | None = None  # User message awaiting LLM response
        self._current_session: Session | None = None  # For save/load functionality
        self._current_model: str = self.config.model  # Runtime model (can be changed)

    def compose(self) -> ComposeResult:
        """Create the UI layout.

        Yields widgets in top-to-bottom order. The layout is:
        - Header: Shows app title
        - ChatLog: Scrollable area for messages (extends VerticalScroll)
        - Container with ChatInput: User text input area (multi-line)
        - StatusBar: Single-line status display
        - Footer: Shows available keyboard shortcuts
        """
        yield Header()
        yield ChatLog(id="chat-log")
        yield Container(
            ChatInput(id="input"),
            id="input-container",
        )
        yield StatusBar(id="status-bar", model=self.config.model, streaming=self.streaming)
        yield Footer()

    def on_mount(self) -> None:
        """Handle app mount event.

        Called when the app is fully loaded and ready. Sets up:
        - Initial focus on input widget
        - Check for missing configuration (show warnings)
        - Create OpenAI client
        - Initialize conversation state
        - Start transcript logging if enabled
        - Add system prompt if configured
        - Load session or query file if provided
        """
        self.query_one("#input", ChatInput).focus()
        chat_log = self.query_one("#chat-log", ChatLog)

        # Check for missing configuration and warn user
        self._check_startup_config(chat_log)

        # Initialize client and conversation
        self.client = OpenAIClient(self.config)
        self.conversation = Conversation()

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
            self._load_session_file(chat_log)
        elif self.query_file:
            self._load_and_submit_query_file()

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

    def _load_session_file(self, chat_log: ChatLog) -> None:
        """Load a saved session from file.

        Args:
            chat_log: The chat log widget to display messages.
        """
        if not self.session_file:
            return

        try:
            session = load_session(self.session_file)
            self._current_session = session

            # Restore conversation state
            if self.conversation:
                self.conversation.clear()
                for msg in session.messages:
                    if msg.role == "system":
                        self.conversation.add_system_message(msg.content)
                    elif msg.role == "user":
                        self.conversation.add_user_message(msg.content)
                    elif msg.role == "assistant":
                        self.conversation.add_assistant_message(msg.content)

            # Restore chat log display (skip system messages)
            for msg in session.messages:
                if msg.role != "system":
                    chat_log.add_message(msg.role, msg.content)

            # Show confirmation
            chat_log.add_message(
                "system",
                f"Loaded session: {session.metadata.name}\n"
                f"Messages: {session.metadata.message_count}",
            )

        except Exception as e:
            chat_log.add_message("error", f"Failed to load session: {e}")

    def _load_and_submit_query_file(self) -> None:
        """Load query from file and submit it."""
        if not self.query_file:
            return
        try:
            with open(self.query_file) as f:
                content = f.read().strip()
            if content:
                # Set input text and trigger submit
                input_widget = self.query_one("#input", ChatInput)
                input_widget.text = content
                self.action_submit()
        except Exception as e:
            chat_log = self.query_one("#chat-log", ChatLog)
            chat_log.add_message("error", f"Failed to load query file: {e}")

    def on_chat_input_submitted(self, _event: ChatInput.Submitted) -> None:
        """Handle Ctrl+E from ChatInput widget."""
        self.action_submit()

    def action_submit(self) -> None:
        """Submit the current message (Ctrl+E).

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

    async def _send_message(self) -> None:  # noqa: C901
        """Worker function for async LLM call.

        Handles the complete workflow:
        1. Update status to "Thinking..."
        2. Call RAGProvider.augment() to get messages (with potential context)
        3. Call client.chat() with streaming or non-streaming
        4. Stream tokens to chat log (if streaming)
        5. Update conversation with user message and assistant response
        6. Update status bar with token count
        7. Handle errors gracefully

        Note: This method is marked noqa: C901 due to inherent complexity
        of handling both streaming and non-streaming modes with error handling.
        """
        if not self.client or not self.conversation:
            return

        user_text = self._pending_user_text
        if not user_text:
            return

        chat_log = self.query_one("#chat-log", ChatLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        status_bar.update_status(status="Thinking...")

        try:
            # Use RAG provider to augment messages with context
            # NullProvider passes through unchanged; LitkitProvider adds context
            messages, rag_metadata = await self.rag_provider.augment(self.conversation, user_text)
            self.last_rag_metadata = rag_metadata

            if self.streaming:
                # Streaming mode
                status_bar.update_status(status="Streaming...")

                # Add empty assistant message for streaming (marked as streaming)
                msg_widget = chat_log.add_message("assistant", "")
                msg_widget.is_streaming = True
                response_content = ""

                result = await self.client.chat(messages, stream=True)
                stream = cast(AsyncIterator[str], result)

                async for token in stream:
                    chat_log.append_to_last(token)
                    response_content += token

                # Finish streaming - re-render with markdown
                msg_widget.finish_streaming()

                # Update conversation with user message and assistant response
                # User message is added here (after success) to keep conversation
                # in sync with what was actually sent to the API
                self.conversation.add_user_message(user_text)
                self.conversation.add_assistant_message(response_content)

                # Log to transcript with response time
                response_time = status_bar.last_response_time
                self.transcript.log_message(
                    "assistant",
                    response_content,
                    model=self.config.model,
                    response_time_s=response_time,
                    tokens=self.conversation.server_reported_tokens,
                )

            else:
                # Non-streaming mode - message renders with markdown immediately
                from chatty.client.openai_client import AssistantMessage

                result = await self.client.chat(messages, stream=False)
                response = cast(AssistantMessage, result)

                chat_log.add_message("assistant", response.content)

                # Update conversation with user message and assistant response
                self.conversation.add_user_message(user_text)
                self.conversation.add_assistant_message(response.content, response.usage)

                # Log to transcript
                self.transcript.log_message(
                    "assistant",
                    response.content,
                    model=self.config.model,
                    tokens=response.usage.get("total_tokens") if response.usage else None,
                )

            # Update status with token count
            status_bar.update_status(
                status="Ready",
                tokens=self.conversation.get_token_display(),
            )

        except ChattyClientError as e:
            chat_log.add_message("error", str(e))
            status_bar.update_status(status="Error")
            self.transcript.log_message("error", str(e))

        except Exception as e:
            chat_log.add_message("error", f"Unexpected error: {e}")
            status_bar.update_status(status="Error")
            self.transcript.log_message("error", f"Unexpected error: {e}")

        finally:
            self.current_worker = None

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
        self.push_screen(SessionBrowserModal(session_dir), self._handle_session_load)

    def _handle_session_load(self, filepath: Path | None) -> None:
        """Handle the session file selected from browser.

        Args:
            filepath: Path to session file, or None if cancelled.
        """
        if not filepath:
            return

        chat_log = self.query_one("#chat-log", ChatLog)

        try:
            session = load_session(filepath)

            # Clear current state
            chat_log.clear_messages()
            if self.conversation:
                self.conversation.clear()

            self._current_session = session

            # Restore conversation state
            if self.conversation:
                for msg in session.messages:
                    if msg.role == "system":
                        self.conversation.add_system_message(msg.content)
                    elif msg.role == "user":
                        self.conversation.add_user_message(msg.content)
                    elif msg.role == "assistant":
                        self.conversation.add_assistant_message(msg.content)

            # Restore chat log display (skip system messages)
            for msg in session.messages:
                if msg.role != "system":
                    chat_log.add_message(msg.role, msg.content)

            # Show confirmation
            chat_log.add_message(
                "system",
                f"Loaded session: {session.metadata.name}\n"
                f"Messages: {session.metadata.message_count}",
            )

            # Update status bar with token count
            if self.conversation:
                self.query_one("#status-bar", StatusBar).update_status(
                    status="Ready",
                    tokens=self.conversation.get_token_display(),
                )

        except Exception as e:
            chat_log.add_message("error", f"Failed to load session: {e}")

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
        if not path:
            return

        chat_log = self.query_one("#chat-log", ChatLog)

        try:
            with open(path) as f:
                content = f.read().strip()

            if content:
                # Load content into input area (don't auto-submit)
                input_widget = self.query_one("#input", ChatInput)
                input_widget.text = content
                input_widget.focus()
                chat_log.add_message("system", f"Loaded query from: {path}")
            else:
                chat_log.add_message("error", f"File is empty: {path}")

        except FileNotFoundError:
            chat_log.add_message("error", f"File not found: {path}")
        except PermissionError:
            chat_log.add_message("error", f"Permission denied: {path}")
        except Exception as e:
            chat_log.add_message("error", f"Failed to load file: {e}")

    def action_save(self) -> None:
        """Save the current session to a file (Ctrl+S).

        Creates or updates a session file with the current conversation.
        Auto-generates a name from the first user message if needed.
        """
        if not self.conversation or not self.conversation.messages:
            chat_log = self.query_one("#chat-log", ChatLog)
            chat_log.add_message("system", "Nothing to save — conversation is empty.")
            return

        chat_log = self.query_one("#chat-log", ChatLog)

        # Create or update session
        if self._current_session is None:
            # Generate name from first user message
            name = generate_session_name(self.conversation.messages)
            metadata = SessionMetadata.create(
                name=name,
                model=self.config.model,
                message_count=len(self.conversation.messages),
            )
            self._current_session = Session(
                metadata=metadata,
                system_prompt=self.config.system_prompt,
                messages=list(self.conversation.messages),
            )
        else:
            # Update existing session
            self._current_session.metadata.message_count = len(self.conversation.messages)
            self._current_session.messages = list(self.conversation.messages)

        # Save to file
        try:
            session_dir = self.config.get_session_path()
            filepath = save_session(self._current_session, session_dir)
            chat_log.add_message(
                "system",
                f"Session saved: {self._current_session.metadata.name}\n" f"File: {filepath}",
            )
        except Exception as e:
            chat_log.add_message("error", f"Failed to save session: {e}")

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

        self.query_one("#status-bar", StatusBar).update_status(
            status="Ready",
            tokens="",
        )

    def action_cancel(self) -> None:
        """Cancel the current generation (Escape).

        Cancels any in-flight LLM request by cancelling the async task.
        Partial responses may remain visible in the chat log.
        """
        if self.current_worker and self.current_worker.is_running:
            self.current_worker.cancel()
            self.query_one("#status-bar", StatusBar).update_status(status="Cancelled")
            self.current_worker = None

    def action_copy(self) -> None:
        """Copy the last assistant response (Ctrl+Y).

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

        Args:
            text: Text to copy.

        Returns:
            Status message describing what happened.
        """
        try:
            import pyperclip

            pyperclip.copy(text)
            return "Copied to clipboard"
        except Exception:
            # Clipboard unavailable — fall back to file
            return self._write_copy_file(text)

    def _write_copy_file(self, text: str) -> str:
        """Write text to fallback copy file.

        Args:
            text: Text to write.

        Returns:
            Status message with file path.
        """
        try:
            copy_dir = self.config.get_copy_fallback_path()
            copy_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            filepath = copy_dir / f"copy-{timestamp}.txt"

            filepath.write_text(text)
            return f"Clipboard unavailable. Saved to {filepath}"
        except Exception as e:
            return f"Failed to copy: {e}"

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
        if not self.client:
            return

        chat_log = self.query_one("#chat-log", ChatLog)

        try:
            models = await self.client.models()
            if models:
                # Show picker with available models
                self.call_from_thread(
                    self.push_screen,
                    ModelPickerModal(models, self._current_model),
                    self._handle_model_selection,
                )
            else:
                # Empty list - show manual input
                self.call_from_thread(
                    self.push_screen,
                    ModelInputModal(self._current_model, "No models returned"),
                    self._handle_model_selection,
                )
        except ChattyClientError as e:
            # /models not supported - show manual input
            self.call_from_thread(
                self.push_screen,
                ModelInputModal(self._current_model, str(e)[:50]),
                self._handle_model_selection,
            )
        except Exception as e:
            chat_log.add_message("error", f"Failed to fetch models: {e}")

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
        """Export the current conversation to Markdown (Ctrl+B).

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
