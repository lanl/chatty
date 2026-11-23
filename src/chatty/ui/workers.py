"""Async worker functions for ChatApp.

This module contains extracted async worker logic from the main
ChatApp class. Functions receive the app instance to access client,
conversation state, and UI widgets.

Functions
---------
    fetch_context_window: Fetch context window from /models endpoint
    fetch_and_show_models: Fetch models and show picker modal
    send_message: Main LLM call with streaming support
    extract_cited_indices: Parse citation references from LLM response
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from chatty.client.openai_client import ChattyClientError

if TYPE_CHECKING:
    from chatty.ui.app import ChatApp


def extract_cited_indices(text: str) -> set[int]:
    """Extract citation indices from LLM response text.

    Parses bracketed numbers like [1], [2], [1, 3, 7], etc.

    Args:
        text: LLM response text containing citations.

    Returns:
        Set of 1-based citation indices found in text.
    """
    cited: set[int] = set()
    # Match patterns like [1], [2], [1, 3, 7], [1,3,7]
    for match in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        # Parse comma-separated numbers
        for num in match.group(1).split(","):
            num = num.strip()
            if num.isdigit():
                cited.add(int(num))
    return cited


def extract_citations(text: str) -> list[tuple[int | None, int]]:
    """Extract unique citations from LLM response text.

    Parses both simple [N] and turn-prefixed [T.N] citation formats:
    - [1], [2], [3] - simple citations (turn=None)
    - [1.1], [1.2], [2.3] - turn-prefixed citations (turn, index)
    - [1, 2, 3] - comma-separated simple citations

    Note: [T.N] citations are collected first, then [N] citations.
    Output order may not match text appearance order. Duplicates are removed.

    Args:
        text: LLM response text containing citations.

    Returns:
        Deduplicated list of (turn, index) tuples. Turn is None for simple
        [N] format. [T.N] citations appear before [N] citations in output.

    Examples:
        >>> extract_citations("See [1] and [2.3]")
        [(2, 3), (None, 1)]
        >>> extract_citations("Sources [1, 2]")
        [(None, 1), (None, 2)]
    """
    citations: list[tuple[int | None, int]] = []
    seen: set[tuple[int | None, int]] = set()

    # Match [T.N] format (turn-prefixed)
    for match in re.finditer(r"\[(\d+)\.(\d+)\]", text):
        turn = int(match.group(1))
        index = int(match.group(2))
        turn_citation: tuple[int | None, int] = (turn, index)
        if turn_citation not in seen:
            citations.append(turn_citation)
            seen.add(turn_citation)

    # Match [N] format (simple) and [N, N, N] format (comma-separated)
    # Exclude matches that are part of [T.N] format
    for match in re.finditer(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        content = match.group(1)
        # Skip if this looks like T.N format (already handled above)
        if "." in text[match.start() : match.end()]:
            continue
        # Parse comma-separated numbers
        for num_str in content.split(","):
            num_str = num_str.strip()
            if num_str.isdigit():
                simple_citation: tuple[int | None, int] = (None, int(num_str))
                if simple_citation not in seen:
                    citations.append(simple_citation)
                    seen.add(simple_citation)

    return citations


async def fetch_context_window(app: ChatApp) -> None:
    """Fetch context window from /models endpoint.

    Called when context_window = "auto". Fetches model metadata
    and extracts context_length. Shows error if not available.

    Args:
        app: The ChatApp instance.
    """
    from chatty.ui.widgets import ChatLog, StatusBar

    if not app.client or not app.conversation:
        return

    chat_log = app.query_one("#chat-log", ChatLog)

    try:
        context_length = await app.client.get_model_context_length(app.config.model)
        if context_length is not None:
            app.conversation.set_context_window(context_length)
            # Update status bar to reflect actual context window
            app.query_one("#status-bar", StatusBar).update_status(
                tokens=app.conversation.get_token_display()
            )
        else:
            # Endpoint didn't provide context_length
            chat_log.add_message(
                "error",
                f"⚠ context_window = 'auto' but /models endpoint did not "
                f"return context_length for model '{app.config.model}'.\n\n"
                "To fix, set context_window explicitly in chatty.toml:\n"
                "  context_window = 128000  # or your model's limit\n\n"
                "Using fallback value: 128,000 tokens",
            )
    except ChattyClientError as e:
        # Endpoint not accessible
        chat_log.add_message(
            "error",
            f"⚠ context_window = 'auto' but failed to fetch from endpoint:\n"
            f"  {e}\n\n"
            "To fix, set context_window explicitly in chatty.toml:\n"
            "  context_window = 128000  # or your model's limit\n\n"
            "Using fallback value: 128,000 tokens",
        )
    except Exception as e:
        chat_log.add_message(
            "error",
            f"⚠ Failed to determine context window: {e}\n\nUsing fallback value: 128,000 tokens",
        )


async def fetch_and_show_models(app: ChatApp) -> None:
    """Fetch models from endpoint and show picker modal.

    Args:
        app: The ChatApp instance.
    """
    from chatty.ui.modals import ModelInputModal, ModelPickerModal
    from chatty.ui.widgets import ChatLog

    if not app.client:
        return

    chat_log = app.query_one("#chat-log", ChatLog)

    try:
        models = await app.client.models()
        if models:
            # Show picker with available models
            app.push_screen(
                ModelPickerModal(models, app._current_model),
                app._handle_model_selection,
            )
        else:
            # Empty list - show manual input
            app.push_screen(
                ModelInputModal(app._current_model, "No models returned"),
                app._handle_model_selection,
            )
    except ChattyClientError as e:
        # /models not supported - show manual input
        app.push_screen(
            ModelInputModal(app._current_model, str(e)[:50]),
            app._handle_model_selection,
        )
    except Exception as e:
        chat_log.add_message("error", f"Failed to fetch models: {e}")


async def send_message(app: ChatApp, *, regenerate: bool = False) -> None:  # noqa: C901
    """Worker function for async LLM call.

    Handles the complete workflow:
    1. Update status to "Searching corpus..." (if RAG enabled)
    2. Call RAGProvider.augment() to get messages (with potential context)
    3. Update status to "Thinking..."
    4. Call client.chat() with streaming or non-streaming
    5. Stream tokens to chat log (if streaming)
    6. Display citations widget (if RAG returned sources)
    7. Update conversation with user message and assistant response
    8. Update status bar with token count
    9. Handle errors gracefully

    Args:
        app: The ChatApp instance.
        regenerate: Replace the last completed reply using the history before its question.
    """
    from chatty.rag.provider import NullProvider
    from chatty.ui.widgets import ChatLog, CitationsWidget, StatusBar

    if not app.client or not app.conversation:
        return

    user_text = app._pending_user_text
    if not user_text:
        return

    chat_log = app.query_one("#chat-log", ChatLog)
    status_bar = app.query_one("#status-bar", StatusBar)

    worker = app.current_worker
    history = app.conversation
    previous_reply = chat_log.last_reply if regenerate else None
    if regenerate:
        if previous_reply is None:
            return
        # Keep the saved/displayed turn intact until its replacement succeeds.
        history = replace(history, messages=history.messages[:-2], server_reported_tokens=None)
    original_content = previous_reply.message_content if previous_reply is not None else ""

    # Show "Searching corpus..." for RAG providers (not NullProvider)
    is_rag_enabled = not isinstance(app.rag_provider, NullProvider)
    if is_rag_enabled:
        status_bar.update_status(status="Searching corpus...")
    else:
        status_bar.update_status(status="Thinking...")

    try:
        # Use RAG provider to augment messages with context
        # NullProvider passes through unchanged; LitkitProvider adds context
        try:
            messages, rag_metadata = await app.rag_provider.augment(
                history, user_text, client=app.client
            )
        except Exception:
            app.transcript.log_message("user", user_text)
            raise

        # Log user message with RAG metadata (after augment has metadata)
        app.transcript.log_message("user", user_text, rag_metadata=rag_metadata)

        # Update status after RAG retrieval completes
        if is_rag_enabled:
            status_bar.update_status(status="Thinking...")

        usage = None
        streaming = app.streaming
        if streaming:
            status_bar.update_status(status="Streaming...")
            widget = previous_reply or chat_log.add_message("assistant", "")
            widget.is_streaming = True
            widget.set_content("")
            result = await app.client.chat(messages, stream=True)
            async for token in cast(AsyncIterator[str], result):
                widget.append_content(token)
            content = widget.message_content
            widget.finish_streaming()
        else:
            from chatty.client.openai_client import AssistantMessage

            response = cast(AssistantMessage, await app.client.chat(messages, stream=False))
            content, usage = response.content, response.usage
            if previous_reply is None:
                widget = chat_log.add_message("assistant", content)
            else:
                widget = previous_reply
                widget.set_content(content)

        if widget.citations is not None:
            widget.citations.remove()
        widget.citations = CitationsWidget.from_metadata(rag_metadata, original_query=user_text)
        if widget.citations is not None:
            chat_log.mount(widget.citations, after=widget)
            chat_log.scroll_end(animate=False)
        chat_log.last_reply = widget
        app.last_rag_metadata = rag_metadata

        if regenerate:
            app.conversation.messages.pop()
            app.conversation.server_reported_tokens = None
        else:
            app.conversation.add_user_message(user_text)
        app.conversation.add_assistant_message(
            content, usage, citation_turn=rag_metadata.turn_number if rag_metadata.sources else None
        )
        app.transcript.log_message(
            "assistant",
            content,
            model=app.config.model,
            response_time_s=status_bar.last_response_time if streaming else None,
            tokens=app.conversation.server_reported_tokens
            if streaming
            else (usage or {}).get("total_tokens"),
        )

        # Mark session as dirty (has unsaved changes)
        app._session_dirty = True

        # Update status with token count
        status_bar.update_status(
            status="Ready",
            tokens=app.conversation.get_token_display(),
        )

    except ChattyClientError as e:
        chat_log.add_message("error", str(e))
        status_bar.update_status(status="Error")
        app.transcript.log_message("error", str(e))

    except Exception as e:
        chat_log.add_message("error", f"Unexpected error: {e}")
        status_bar.update_status(status="Error")
        app.transcript.log_message("error", f"Unexpected error: {e}")

    finally:
        if previous_reply is not None and previous_reply.is_streaming:
            previous_reply.set_content(original_content)
            previous_reply.finish_streaming()
        if app.current_worker is worker:
            app.current_worker = None
