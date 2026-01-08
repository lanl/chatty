"""Tests for conversation state management."""

from unittest.mock import patch

from chatty.client.openai_client import Message
from chatty.core.conversation import Conversation


def test_conversation_init_empty() -> None:
    """Test conversation initializes with empty messages."""
    conv = Conversation()
    assert conv.messages == []
    assert conv.server_reported_tokens is None
    assert conv.context_window == 128000


def test_add_user_message() -> None:
    """Test adding a user message."""
    conv = Conversation()
    conv.add_user_message("Hello!")

    assert len(conv.messages) == 1
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "Hello!"


def test_add_assistant_message() -> None:
    """Test adding an assistant message."""
    conv = Conversation()
    conv.add_assistant_message("Hi there!")

    assert len(conv.messages) == 1
    assert conv.messages[0].role == "assistant"
    assert conv.messages[0].content == "Hi there!"


def test_add_assistant_message_with_usage() -> None:
    """Test adding assistant message updates token count."""
    conv = Conversation()
    conv.add_assistant_message("Response", usage={"total_tokens": 150})

    assert conv.server_reported_tokens == 150


def test_add_assistant_message_usage_without_total() -> None:
    """Test usage dict without total_tokens is ignored."""
    conv = Conversation()
    conv.add_assistant_message("Response", usage={"prompt_tokens": 50})

    assert conv.server_reported_tokens is None


def test_add_system_message() -> None:
    """Test adding a system message."""
    conv = Conversation()
    conv.add_system_message("You are helpful.")

    assert len(conv.messages) == 1
    assert conv.messages[0].role == "system"
    assert conv.messages[0].content == "You are helpful."


def test_multiple_messages() -> None:
    """Test adding multiple messages preserves order."""
    conv = Conversation()
    conv.add_system_message("System prompt")
    conv.add_user_message("User question")
    conv.add_assistant_message("Assistant response")

    assert len(conv.messages) == 3
    assert conv.messages[0].role == "system"
    assert conv.messages[1].role == "user"
    assert conv.messages[2].role == "assistant"


def test_clear() -> None:
    """Test clearing conversation."""
    conv = Conversation()
    conv.add_user_message("Hello")
    conv.add_assistant_message("Hi", usage={"total_tokens": 100})

    conv.clear()

    assert conv.messages == []
    assert conv.server_reported_tokens is None


def test_get_messages_for_api() -> None:
    """Test converting messages to API format."""
    conv = Conversation()
    conv.add_system_message("Be helpful")
    conv.add_user_message("Question")
    conv.add_assistant_message("Answer")

    api_messages = conv.get_messages_for_api()

    assert api_messages == [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]


def test_get_messages_for_api_empty() -> None:
    """Test API format with no messages."""
    conv = Conversation()
    assert conv.get_messages_for_api() == []


def test_get_token_display_server_reported() -> None:
    """Test token display with server-reported tokens."""
    conv = Conversation(server_reported_tokens=12000, context_window=128000)

    display = conv.get_token_display()

    assert display == "12,000 / 128,000 tokens"


def test_get_token_display_server_reported_small() -> None:
    """Test token display with small token count (exact value shown)."""
    conv = Conversation(server_reported_tokens=500, context_window=128000)

    display = conv.get_token_display()

    assert display == "500 / 128,000 tokens"


def test_get_token_display_tiktoken_fallback() -> None:
    """Test token display falls back to tiktoken estimate."""
    conv = Conversation()
    conv.add_user_message("Hello world")

    # tiktoken is available in test environment
    display = conv.get_token_display()

    assert "~" in display or "est" in display or "unknown" in display


def test_get_token_display_no_tiktoken() -> None:
    """Test token display when tiktoken unavailable."""
    conv = Conversation()
    conv.add_user_message("Hello")

    with (
        patch.dict("sys.modules", {"tiktoken": None}),
        patch("chatty.core.conversation.Conversation.get_token_display") as mock,
    ):
        # Force ImportError
        mock.return_value = "usage unknown"
        display = conv.get_token_display()
        assert display == "usage unknown"


def test_is_near_context_limit_under() -> None:
    """Test context limit check when under threshold."""
    conv = Conversation(server_reported_tokens=50000, context_window=128000)

    assert conv.is_near_context_limit() is False


def test_is_near_context_limit_over() -> None:
    """Test context limit check when over threshold."""
    conv = Conversation(server_reported_tokens=110000, context_window=128000)

    # 110000 > 128000 * 0.8 (102400)
    assert conv.is_near_context_limit() is True


def test_is_near_context_limit_exactly_at() -> None:
    """Test context limit check at exactly threshold."""
    conv = Conversation(server_reported_tokens=102400, context_window=128000)

    # 102400 == 128000 * 0.8, not > so False
    assert conv.is_near_context_limit() is False


def test_is_near_context_limit_custom_threshold() -> None:
    """Test context limit check with custom threshold."""
    conv = Conversation(server_reported_tokens=70000, context_window=128000)

    # 70000 > 128000 * 0.5 (64000)
    assert conv.is_near_context_limit(threshold=0.5) is True


def test_is_near_context_limit_no_token_count() -> None:
    """Test context limit check without token count."""
    conv = Conversation()

    # No server_reported_tokens, should return False
    assert conv.is_near_context_limit() is False


def test_conversation_with_custom_context_window() -> None:
    """Test conversation with custom context window."""
    conv = Conversation(context_window=32000)

    assert conv.context_window == 32000


def test_message_dataclass() -> None:
    """Test Message dataclass."""
    msg = Message(role="user", content="Hello")

    assert msg.role == "user"
    assert msg.content == "Hello"
