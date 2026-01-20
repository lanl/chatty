"""Clipboard utilities for chatty.

Provides clipboard copy with fallback to file for headless environments.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def copy_to_clipboard(text: str, fallback_dir: Path) -> str:
    """Copy text to clipboard with fallback to file.

    Attempts to copy text to system clipboard using pyperclip.
    If clipboard is unavailable (headless HPC), falls back to
    writing to a timestamped file in fallback_dir.

    Args:
        text: Text to copy.
        fallback_dir: Directory for fallback file if clipboard unavailable.

    Returns:
        Status message describing what happened.
    """
    try:
        import pyperclip

        pyperclip.copy(text)
        return "Copied to clipboard"
    except Exception:
        # Clipboard unavailable — fall back to file
        return write_copy_file(text, fallback_dir)


def write_copy_file(text: str, copy_dir: Path) -> str:
    """Write text to fallback copy file.

    Creates a timestamped file in copy_dir with the given text.
    Used when system clipboard is unavailable.

    Args:
        text: Text to write.
        copy_dir: Directory to write file in.

    Returns:
        Status message with file path, or error message.
    """
    try:
        copy_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        filepath = copy_dir / f"copy-{timestamp}.txt"

        filepath.write_text(text)
        return f"Clipboard unavailable. Saved to {filepath}"
    except Exception as e:
        return f"Failed to copy: {e}"
