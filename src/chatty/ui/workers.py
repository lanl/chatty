"""Async worker functions for ChatApp.

This module contains extracted async worker logic from the main
ChatApp class. Functions receive the app instance to access client,
conversation state, and UI widgets.

Functions
---------
    fetch_context_window: Fetch context window from /models endpoint
    fetch_and_show_models: Fetch models and show picker modal
    send_message: Main LLM call with streaming support
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, cast

from chatty.client.openai_client import ChattyClientError

if TYPE_CHECKING:
    from chatty.ui.app import ChatApp


async def fetch_context_window(app: "ChatApp") -> None:
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
        context_length = await app.client.get_model_context_length(
            app.config.model
        )
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
            f"⚠ Failed to determine context window: {e}\n\n"
            "Using fallback value: 128,000 tokens",
        )


async def fetch_and_show_models(app: "ChatApp") -> None:
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


async def send_message(app: "ChatApp") -> None:  # noqa: C901
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

    Args:
        app: The ChatApp instance.
    """
    from chatty.ui.widgets import ChatLog, StatusBar

    if not app.client or not app.conversation:
        return

    user_text = app._pending_user_text
    if not user_text:
        return

    chat_log = app.query_one("#chat-log", ChatLog)
    status_bar = app.query_one("#status-bar", StatusBar)

    status_bar.update_status(status="Thinking...")

    try:
        # Use RAG provider to augment messages with context
        # NullProvider passes through unchanged; LitkitProvider adds context
        messages, rag_metadata = await app.rag_provider.augment(
            app.conversation, user_text
        )
        app.last_rag_metadata = rag_metadata

        if app.streaming:
            # Streaming mode
            status_bar.update_status(status="Streaming...")

            # Add empty assistant message for streaming (marked as streaming)
            msg_widget = chat_log.add_message("assistant", "")
            msg_widget.is_streaming = True
            response_content = ""

            result = await app.client.chat(messages, stream=True)
            stream = cast(AsyncIterator[str], result)

            async for token in stream:
                chat_log.append_to_last(token)
                response_content += token

            # Finish streaming - re-render with markdown
            msg_widget.finish_streaming()

            # Update conversation with user message and assistant response
            # User message is added here (after success) to keep conversation
            # in sync with what was actually sent to the API
            app.conversation.add_user_message(user_text)
            app.conversation.add_assistant_message(response_content)

            # Log to transcript with response time
            response_time = status_bar.last_response_time
            app.transcript.log_message(
                "assistant",
                response_content,
                model=app.config.model,
                response_time_s=response_time,
                tokens=app.conversation.server_reported_tokens,
            )

        else:
            # Non-streaming mode - message renders with markdown immediately
            from chatty.client.openai_client import AssistantMessage

            result = await app.client.chat(messages, stream=False)
            response = cast(AssistantMessage, result)

            chat_log.add_message("assistant", response.content)

            # Update conversation with user message and assistant response
            app.conversation.add_user_message(user_text)
            app.conversation.add_assistant_message(
                response.content, response.usage
            )

            # Log to transcript
            app.transcript.log_message(
                "assistant",
                response.content,
                model=app.config.model,
                tokens=(
                    response.usage.get("total_tokens")
                    if response.usage
                    else None
                ),
            )

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
        app.current_worker = None
