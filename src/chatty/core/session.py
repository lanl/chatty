"""Session persistence for conversation history."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from chatty.client.openai_client import Message


@dataclass
class SessionMetadata:
    """Metadata about a saved session."""

    id: str
    name: str
    created_at: str  # ISO format
    updated_at: str  # ISO format
    model: str
    message_count: int

    @classmethod
    def create(cls, name: str, model: str, message_count: int) -> SessionMetadata:
        """Create new metadata with generated ID and timestamps."""
        now = datetime.now().isoformat(timespec="seconds")
        return cls(
            id=uuid.uuid4().hex[:12],
            name=name,
            created_at=now,
            updated_at=now,
            model=model,
            message_count=message_count,
        )


@dataclass
class Session:
    """A complete conversation session that can be saved/loaded."""

    metadata: SessionMetadata
    system_prompt: str
    messages: list[Message] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary for JSON serialization."""
        return {
            "version": "1.0",
            "metadata": asdict(self.metadata),
            "system_prompt": self.system_prompt,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Create session from dictionary (loaded from JSON)."""
        # Handle version for future compatibility
        version = data.get("version", "1.0")
        if version != "1.0":
            raise ValueError(f"Unsupported session version: {version}")

        metadata = SessionMetadata(**data["metadata"])
        messages = [Message(role=m["role"], content=m["content"]) for m in data["messages"]]
        return cls(
            metadata=metadata,
            system_prompt=data.get("system_prompt", ""),
            messages=messages,
        )


def generate_session_name(messages: list[Message]) -> str:
    """Generate a session name from the first user message.

    Args:
        messages: List of conversation messages.

    Returns:
        A name derived from the first user message, truncated to 40 chars.
    """
    for msg in messages:
        if msg.role == "user":
            # Take first line, strip whitespace
            first_line = msg.content.split("\n")[0].strip()
            if len(first_line) > 40:
                return first_line[:37] + "..."
            return first_line if first_line else "Untitled session"
    return "Untitled session"


def save_session(session: Session, directory: Path) -> Path:
    """Save a session to a JSON file.

    Args:
        session: The session to save.
        directory: Directory to save the session in.

    Returns:
        Path to the saved session file.
    """
    directory.mkdir(parents=True, exist_ok=True)

    # Update metadata timestamp
    session.metadata.updated_at = datetime.now().isoformat(timespec="seconds")

    # Use session ID as filename
    filename = f"session-{session.metadata.id}.json"
    filepath = directory / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)

    return filepath


def load_session(filepath: Path) -> Session:
    """Load a session from a JSON file.

    Args:
        filepath: Path to the session file.

    Returns:
        The loaded Session object.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is invalid.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Session file not found: {filepath}")

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    return Session.from_dict(data)


def list_sessions(directory: Path) -> list[SessionMetadata]:
    """List all sessions in a directory.

    Args:
        directory: Directory containing session files.

    Returns:
        List of SessionMetadata objects, sorted by updated_at (newest first).
    """
    if not directory.exists():
        return []

    sessions: list[SessionMetadata] = []

    for filepath in directory.glob("session-*.json"):
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            metadata = SessionMetadata(**data["metadata"])
            sessions.append(metadata)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Skip invalid session files
            continue

    # Sort by updated_at, newest first
    sessions.sort(key=lambda s: s.updated_at, reverse=True)
    return sessions


def get_session_filepath(directory: Path, session_id: str) -> Path | None:
    """Get the filepath for a session by ID.

    Args:
        directory: Directory containing session files.
        session_id: The session ID to find.

    Returns:
        Path to the session file, or None if not found.
    """
    filepath = directory / f"session-{session_id}.json"
    if filepath.exists():
        return filepath
    return None


def export_session_markdown(
    messages: list[Message],
    *,
    session_name: str | None = None,
    model: str | None = None,
) -> str:
    """Export conversation messages to Markdown format.

    Args:
        messages: List of messages to export.
        session_name: Optional session name for the header.
        model: Optional model name to include in header.

    Returns:
        Formatted Markdown string.
    """
    lines: list[str] = []

    # Header
    title = session_name or "Chat Session"
    lines.append(f"# {title}")
    lines.append("")

    # Metadata
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines.append(f"Exported: {timestamp}")
    if model:
        lines.append(f"Model: {model}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Messages (skip system messages)
    for msg in messages:
        if msg.role == "system":
            continue

        if msg.role == "user":
            lines.append(f"**You:** {msg.content}")
        elif msg.role == "assistant":
            lines.append(f"**Assistant:** {msg.content}")
        else:
            lines.append(f"**{msg.role.title()}:** {msg.content}")

        lines.append("")
        lines.append("---")
        lines.append("")

    # Footer
    lines.append("*Exported from chatty*")
    lines.append("")

    return "\n".join(lines)


def save_markdown_export(
    messages: list[Message],
    directory: Path,
    *,
    session_name: str | None = None,
    model: str | None = None,
) -> Path:
    """Export conversation to a Markdown file.

    Args:
        messages: List of messages to export.
        directory: Directory to save the export in.
        session_name: Optional session name (used in header and filename).
        model: Optional model name to include in header.

    Returns:
        Path to the saved Markdown file.
    """
    directory.mkdir(parents=True, exist_ok=True)

    # Generate filename (timestamp only - session name goes in header)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    filename = f"chatty-export-{timestamp}.md"
    filepath = directory / filename

    content = export_session_markdown(
        messages,
        session_name=session_name,
        model=model,
    )

    filepath.write_text(content, encoding="utf-8")
    return filepath
