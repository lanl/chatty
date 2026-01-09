"""JSONL transcript logging for conversation history."""

from __future__ import annotations

import contextlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from chatty.config import Config


class TranscriptLogger:
    """Logs conversation messages to a JSONL file.

    Each message is written as a single JSON object per line:
    {"timestamp": "2026-01-09T10:32:15.123", "role": "user", "content": "Hello"}

    File naming: chatty-YYYY-MM-DD-HHMMSS.jsonl

    Usage:
        logger = TranscriptLogger(config)
        logger.start_session()
        logger.log_message("user", "Hello")
        logger.log_message("assistant", "Hi!", response_time_s=2.3)
        logger.close()
    """

    def __init__(self, config: Config) -> None:
        """Initialize the transcript logger.

        Args:
            config: Application configuration with transcript settings.
        """
        self._config = config
        self._file: Path | None = None
        self._handle: TextIO | None = None

    @property
    def enabled(self) -> bool:
        """Check if transcript logging is enabled."""
        return self._config.transcript_enabled

    def start_session(self) -> Path | None:
        """Start a new transcript session.

        Creates the transcript directory if needed and opens a new file.

        Returns:
            Path to the transcript file, or None if logging is disabled.
        """
        if not self.enabled:
            return None

        # Create transcript directory
        transcript_dir = self._config.get_transcript_path()
        transcript_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self._file = transcript_dir / f"chatty-{timestamp}.jsonl"

        # Open file for writing (append mode for safety)
        # Note: We manage the file handle lifecycle manually because we need
        # to write incrementally over the app's lifetime, not in a single block.
        self._handle = open(self._file, "a", encoding="utf-8")  # noqa: SIM115

        return self._file

    def log_message(
        self,
        role: str,
        content: str,
        *,
        model: str | None = None,
        response_time_s: float | None = None,
        tokens: int | None = None,
    ) -> None:
        """Log a message to the transcript.

        Args:
            role: Message role (user, assistant, system, error).
            content: Message content.
            model: Model name (for assistant messages).
            response_time_s: Response time in seconds (for assistant messages).
            tokens: Token count (for assistant messages).
        """
        if not self.enabled or self._handle is None:
            return

        record: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "role": role,
            "content": content,
        }

        # Add optional fields for assistant messages
        if model is not None:
            record["model"] = model
        if response_time_s is not None:
            record["response_time_s"] = round(response_time_s, 2)
        if tokens is not None:
            record["tokens"] = tokens

        try:
            line = json.dumps(record, ensure_ascii=False)
            self._handle.write(line + "\n")
            self._handle.flush()
        except Exception:
            # Don't crash the app if logging fails
            pass

    def close(self) -> None:
        """Close the transcript file."""
        if self._handle is not None:
            with contextlib.suppress(Exception):
                self._handle.close()
            self._handle = None

    def get_file_path(self) -> Path | None:
        """Get the current transcript file path."""
        return self._file
