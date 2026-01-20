"""Session management handlers for chatty.

Provides functions for loading, saving, and managing chat sessions.
These functions are called from ChatApp but isolated here for maintainability.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from chatty.core.session import (
    Session,
    SessionMetadata,
    generate_session_name,
    load_session,
    save_session,
)

if TYPE_CHECKING:
    from chatty.ui.app import ChatApp
    from chatty.ui.widgets import ChatLog


def load_session_file(app: "ChatApp", chat_log: "ChatLog") -> None:
    """Load a saved session from file.

    Called during app startup when session_file is provided.

    Args:
        app: The ChatApp instance.
        chat_log: The ChatLog widget for displaying messages.
    """
    if not app.session_file:
        return

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

    except Exception as e:
        chat_log.add_message("error", f"Failed to load session: {e}")


def load_and_submit_query_file(app: "ChatApp") -> None:
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


def handle_session_load(app: "ChatApp", filepath: Path | None) -> None:
    """Handle the session file selected from browser.

    Called when user selects a session in SessionBrowserModal.

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

        # Update status bar with token count
        if app.conversation:
            app.query_one("#status-bar", StatusBar).update_status(
                status="Ready",
                tokens=app.conversation.get_token_display(),
            )

    except Exception as e:
        chat_log.add_message("error", f"Failed to load session: {e}")


def handle_first_save(app: "ChatApp", name: str | None) -> None:
    """Handle the name returned from first-save modal.

    Called when user enters a name in SessionRenameModal for first save.

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
        system_prompt=app.config.system_prompt,
        messages=list(app.conversation.messages),
    )
    # Now save it
    save_current_session(app)


def save_current_session(app: "ChatApp") -> None:
    """Save the current session to disk.

    Updates the session with current conversation state and saves to file.

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
        chat_log.add_message(
            "system",
            f"Session saved: {app._current_session.metadata.name}\n" f"File: {filepath}",
        )
    except Exception as e:
        chat_log.add_message("error", f"Failed to save session: {e}")


def handle_file_path(app: "ChatApp", path: str | None) -> None:
    """Handle the file path returned from FileInputModal.

    Loads file content into the input area (without auto-submit).

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


def generate_default_session_name(app: "ChatApp") -> str:
    """Generate a default session name from conversation.

    Args:
        app: The ChatApp instance.

    Returns:
        Generated session name.
    """
    if app.conversation and app.conversation.messages:
        return generate_session_name(app.conversation.messages)
    return "Untitled Session"
