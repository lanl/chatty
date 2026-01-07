"""Tests for output formatting and display utilities."""

from chatty.core.rendering import format_assistant_message, format_error_message


class TestFormatAssistantMessage:
    """Tests for format_assistant_message."""

    def test_passthrough(self) -> None:
        """Content is returned unchanged (v0.1 behavior)."""
        content = "Hello, world!"
        assert format_assistant_message(content) == content

    def test_multiline(self) -> None:
        """Multi-line content is preserved."""
        content = "Line 1\nLine 2\nLine 3"
        assert format_assistant_message(content) == content

    def test_markdown(self) -> None:
        """Markdown content is preserved."""
        content = "# Header\n\n- item 1\n- item 2\n\n```python\nprint('hi')\n```"
        assert format_assistant_message(content) == content


class TestFormatErrorMessage:
    """Tests for format_error_message."""

    def test_error_only(self) -> None:
        """Error message without hint."""
        result = format_error_message("Connection failed")
        assert result == "Error: Connection failed"

    def test_with_hint(self) -> None:
        """Error message with actionable hint."""
        result = format_error_message(
            "Authentication failed",
            hint="Check your API key with 'chatty doctor'",
        )
        expected = "Error: Authentication failed\n\nHint: Check your API key with 'chatty doctor'"
        assert result == expected

    def test_none_hint(self) -> None:
        """Explicit None hint is same as no hint."""
        result = format_error_message("Some error", hint=None)
        assert result == "Error: Some error"
