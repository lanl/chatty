"""Output formatting and display utilities."""


def format_assistant_message(content: str) -> str:
    """Format assistant message for display.

    Currently a passthrough; will be extended in v0.3 for citations.
    """
    return content


def format_error_message(error: str, hint: str | None = None) -> str:
    """Format an error message with optional actionable hint."""
    msg = f"Error: {error}"
    if hint:
        msg += f"\n\nHint: {hint}"
    return msg
