"""Session management handlers for ChatApp.

This module contains extracted session-related logic from the main
ChatApp class. Functions receive the app instance to access conversation
state, configuration, and UI widgets.

Functions
---------
    load_session_file: Load session from file on startup
    load_and_submit_query_file: Load query file on startup
    handle_session_load: Handle session selected from browser
    handle_first_save: Handle first-save modal result
    save_current_session: Save current session to disk
    handle_file_path: Handle file path modal result
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from chatty.core.session import (
    Session,
    SessionMetadata,
    load_session,
    save_session,
)

if TYPE_CHECKING:
    from chatty.ui.app import ChatApp


def load_session_file(app: ChatApp) -> None:
    """Load a saved session from file.

    Called during app startup when session_file is provided.

    Args:
        app: The ChatApp instance.
    """
    from chatty.ui.widgets import ChatLog, StatusBar

    if not app.session_file:
        return

    chat_log = app.query_one("#chat-log", ChatLog)

    try:
        session = load_session(app.session_file)
        app._current_session = session

        # Restore conversation state
        if app.conversation:
            app.conversation.clear()
            for msg in session.messages:
                if msg.role == "system":
                    app.conversation.add_system_message(msg.content)
                elif msg.role == "user":
                    app.conversation.add_user_message(msg.content)
                elif msg.role == "assistant":
                    app.conversation.add_assistant_message(msg.content)

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
        if app.conversation:
            app.query_one("#status-bar", StatusBar).update_status(
                status="Ready",
                tokens=app.conversation.get_token_display(),
            )

    except Exception as e:
        chat_log.add_message("error", f"Failed to load session: {e}")


def load_and_submit_query_file(app: ChatApp) -> None:
    """Load query from file and submit it.

    Called during app startup when query_file is provided.

    Args:
        app: The ChatApp instance.
    """
    from chatty.ui.widgets import ChatInput, ChatLog

    if not app.query_file:
        return

    try:
        with open(app.query_file) as f:
            content = f.read().strip()
        if content:
            # Set input text and trigger submit
            input_widget = app.query_one("#input", ChatInput)
            input_widget.text = content
            app.action_submit()
    except Exception as e:
        chat_log = app.query_one("#chat-log", ChatLog)
        chat_log.add_message("error", f"Failed to load query file: {e}")


def handle_session_load(app: ChatApp, filepath: Path | None) -> None:
    """Handle the session file selected from browser.

    Args:
        app: The ChatApp instance.
        filepath: Path to session file, or None if cancelled.
    """
    from chatty.ui.widgets import ChatLog, StatusBar

    if not filepath:
        return

    chat_log = app.query_one("#chat-log", ChatLog)

    try:
        session = load_session(filepath)

        # Clear current state
        chat_log.clear_messages()
        if app.conversation:
            app.conversation.clear()

        app._current_session = session

        # Restore conversation state
        if app.conversation:
            for msg in session.messages:
                if msg.role == "system":
                    app.conversation.add_system_message(msg.content)
                elif msg.role == "user":
                    app.conversation.add_user_message(msg.content)
                elif msg.role == "assistant":
                    app.conversation.add_assistant_message(msg.content)

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

        # Clear dirty flag - loaded session is "saved"
        app._session_dirty = False

        # Update status bar with token count
        if app.conversation:
            app.query_one("#status-bar", StatusBar).update_status(
                status="Ready",
                tokens=app.conversation.get_token_display(),
            )

    except Exception as e:
        chat_log.add_message("error", f"Failed to load session: {e}")


def handle_first_save(app: ChatApp, name: str | None) -> None:
    """Handle the name returned from first-save modal.

    Args:
        app: The ChatApp instance.
        name: Session name entered by user, or None if cancelled.
    """
    if not name or not app.conversation:
        return

    # Create new session with the chosen name
    metadata = SessionMetadata.create(
        name=name,
        model=app.config.model,
        message_count=len(app.conversation.messages),
    )
    app._current_session = Session(
        metadata=metadata,
        system_prompt=app.config.get_effective_system_prompt(),
        messages=list(app.conversation.messages),
    )
    # Now save it
    save_current_session(app)


def save_current_session(app: ChatApp) -> None:
    """Save the current session to disk.

    Args:
        app: The ChatApp instance.
    """
    from chatty.ui.widgets import ChatLog

    if not app._current_session or not app.conversation:
        return

    chat_log = app.query_one("#chat-log", ChatLog)

    # Update session with current conversation state
    app._current_session.metadata.message_count = len(app.conversation.messages)
    app._current_session.messages = list(app.conversation.messages)

    # Save to file
    try:
        session_dir = app.config.get_session_path()
        filepath = save_session(app._current_session, session_dir)
        app._session_dirty = False  # Clear dirty flag after successful save
        chat_log.add_message(
            "system",
            f"Session saved: {app._current_session.metadata.name}\n" f"File: {filepath}",
        )
    except Exception as e:
        chat_log.add_message("error", f"Failed to save session: {e}")


def handle_file_path(app: ChatApp, path: str | None) -> None:
    """Handle the file path returned from the modal.

    Args:
        app: The ChatApp instance.
        path: File path entered by user, or None if cancelled.
    """
    from chatty.ui.widgets import ChatInput, ChatLog

    if not path:
        return

    chat_log = app.query_one("#chat-log", ChatLog)

    try:
        with open(path) as f:
            content = f.read().strip()

        if content:
            # Load content into input area (don't auto-submit)
            input_widget = app.query_one("#input", ChatInput)
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
