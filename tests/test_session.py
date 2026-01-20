"""Tests for session persistence."""

from pathlib import Path

import pytest

from chatty.client.openai_client import Message
from chatty.config import Config
from chatty.core.session import (
    Session,
    SessionMetadata,
    delete_session,
    export_session_markdown,
    generate_session_name,
    get_session_filepath,
    list_sessions,
    load_session,
    rename_session,
    save_markdown_export,
    save_session,
)


def test_session_metadata_create() -> None:
    """Test SessionMetadata.create() generates proper fields."""
    metadata = SessionMetadata.create(
        name="Test session",
        model="gpt-4.1",
        message_count=5,
    )

    assert metadata.name == "Test session"
    assert metadata.model == "gpt-4.1"
    assert metadata.message_count == 5
    assert len(metadata.id) == 12  # UUID hex[:12]
    assert metadata.created_at == metadata.updated_at


def test_session_to_dict() -> None:
    """Test Session serialization to dict."""
    metadata = SessionMetadata.create("Test", "gpt-4.1", 2)
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]
    session = Session(
        metadata=metadata,
        system_prompt="You are helpful.",
        messages=messages,
    )

    data = session.to_dict()

    assert data["version"] == "1.0"
    assert data["metadata"]["name"] == "Test"
    assert data["system_prompt"] == "You are helpful."
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"


def test_session_from_dict() -> None:
    """Test Session deserialization from dict."""
    data = {
        "version": "1.0",
        "metadata": {
            "id": "abc123",
            "name": "Loaded session",
            "created_at": "2026-01-09T10:00:00",
            "updated_at": "2026-01-09T11:00:00",
            "model": "gpt-4.1",
            "message_count": 2,
        },
        "system_prompt": "Be helpful.",
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
    }

    session = Session.from_dict(data)

    assert session.metadata.id == "abc123"
    assert session.metadata.name == "Loaded session"
    assert session.system_prompt == "Be helpful."
    assert len(session.messages) == 2
    assert session.messages[0].role == "user"


def test_session_from_dict_unsupported_version() -> None:
    """Test Session raises error for unsupported version."""
    data = {"version": "2.0", "metadata": {}, "messages": []}

    with pytest.raises(ValueError, match="Unsupported session version"):
        Session.from_dict(data)


def test_generate_session_name_from_user_message() -> None:
    """Test session name generation from first user message."""
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Tell me about Python"),
        Message(role="assistant", content="Python is..."),
    ]

    name = generate_session_name(messages)

    assert name == "Tell me about Python"


def test_generate_session_name_truncated() -> None:
    """Test session name is truncated for long messages."""
    long_message = "A" * 50  # Longer than 40 chars
    messages = [Message(role="user", content=long_message)]

    name = generate_session_name(messages)

    assert len(name) == 40  # 37 chars + "..."
    assert name.endswith("...")


def test_generate_session_name_first_line_only() -> None:
    """Test session name uses only first line of message."""
    messages = [Message(role="user", content="First line\nSecond line\nThird line")]

    name = generate_session_name(messages)

    assert name == "First line"


def test_generate_session_name_no_user_message() -> None:
    """Test session name defaults when no user message."""
    messages = [Message(role="system", content="You are helpful.")]

    name = generate_session_name(messages)

    assert name == "Untitled session"


def test_generate_session_name_empty_messages() -> None:
    """Test session name defaults for empty messages."""
    name = generate_session_name([])

    assert name == "Untitled session"


def test_save_and_load_session(tmp_path: Path) -> None:
    """Test saving and loading a session."""
    metadata = SessionMetadata.create("Test save/load", "gpt-4.1", 1)
    session = Session(
        metadata=metadata,
        system_prompt="System prompt",
        messages=[Message(role="user", content="Hello")],
    )

    # Save
    filepath = save_session(session, tmp_path)

    assert filepath.exists()
    assert filepath.name == f"session-{metadata.id}.json"

    # Load
    loaded = load_session(filepath)

    assert loaded.metadata.id == session.metadata.id
    assert loaded.metadata.name == "Test save/load"
    assert loaded.system_prompt == "System prompt"
    assert len(loaded.messages) == 1


def test_save_session_creates_directory(tmp_path: Path) -> None:
    """Test save_session creates directory if needed."""
    new_dir = tmp_path / "sessions" / "nested"
    assert not new_dir.exists()

    metadata = SessionMetadata.create("Test", "gpt-4.1", 0)
    session = Session(metadata=metadata, system_prompt="", messages=[])

    save_session(session, new_dir)

    assert new_dir.exists()


def test_save_session_updates_timestamp(tmp_path: Path) -> None:
    """Test save_session updates updated_at timestamp."""
    metadata = SessionMetadata.create("Test", "gpt-4.1", 0)
    # Manually set an old timestamp
    metadata.updated_at = "2020-01-01T00:00:00"
    session = Session(metadata=metadata, system_prompt="", messages=[])

    save_session(session, tmp_path)

    # Metadata should be updated to current time
    assert session.metadata.updated_at != "2020-01-01T00:00:00"
    assert session.metadata.updated_at.startswith("2026-")  # Current year


def test_load_session_not_found(tmp_path: Path) -> None:
    """Test load_session raises error for missing file."""
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path / "nonexistent.json")


def test_list_sessions_empty_directory(tmp_path: Path) -> None:
    """Test list_sessions returns empty for empty directory."""
    sessions = list_sessions(tmp_path)

    assert sessions == []


def test_list_sessions_nonexistent_directory(tmp_path: Path) -> None:
    """Test list_sessions returns empty for nonexistent directory."""
    sessions = list_sessions(tmp_path / "nonexistent")

    assert sessions == []


def test_list_sessions_multiple(tmp_path: Path) -> None:
    """Test list_sessions returns all sessions sorted by date."""
    # Create multiple sessions with explicit timestamps for sorting
    timestamps = [
        ("First", "2026-01-01T10:00:00"),
        ("Second", "2026-01-02T10:00:00"),
        ("Third", "2026-01-03T10:00:00"),
    ]
    for name, timestamp in timestamps:
        metadata = SessionMetadata.create(name, "gpt-4.1", 0)
        metadata.updated_at = timestamp
        session = Session(metadata=metadata, system_prompt="", messages=[])
        # Write directly to avoid save_session updating timestamp
        import json

        filepath = tmp_path / f"session-{metadata.id}.json"
        with open(filepath, "w") as f:
            json.dump(session.to_dict(), f)

    sessions = list_sessions(tmp_path)

    assert len(sessions) == 3
    # Sessions should be sorted by updated_at, newest first
    assert sessions[0].name == "Third"
    assert sessions[1].name == "Second"
    assert sessions[2].name == "First"


def test_list_sessions_skips_invalid_files(tmp_path: Path) -> None:
    """Test list_sessions skips invalid JSON files."""
    # Create valid session
    metadata = SessionMetadata.create("Valid", "gpt-4.1", 0)
    session = Session(metadata=metadata, system_prompt="", messages=[])
    save_session(session, tmp_path)

    # Create invalid files
    (tmp_path / "session-invalid.json").write_text("not json")
    (tmp_path / "session-missing.json").write_text("{}")

    sessions = list_sessions(tmp_path)

    assert len(sessions) == 1
    assert sessions[0].name == "Valid"


def test_get_session_filepath(tmp_path: Path) -> None:
    """Test get_session_filepath finds session by ID."""
    metadata = SessionMetadata.create("Test", "gpt-4.1", 0)
    session = Session(metadata=metadata, system_prompt="", messages=[])
    save_session(session, tmp_path)

    filepath = get_session_filepath(tmp_path, metadata.id)

    assert filepath is not None
    assert filepath.exists()


def test_get_session_filepath_not_found(tmp_path: Path) -> None:
    """Test get_session_filepath returns None for missing session."""
    filepath = get_session_filepath(tmp_path, "nonexistent")

    assert filepath is None


def test_config_session_path_default() -> None:
    """Test config has session_path with default value."""
    config = Config()

    assert config.session_path == "./sessions"


def test_config_get_session_path_expands_tilde() -> None:
    """Test get_session_path expands ~ to home directory."""
    config = Config(session_path="~/sessions")
    path = config.get_session_path()

    assert "~" not in str(path)
    assert str(path).startswith(str(Path.home()))


# ============================================================================
# Session Rename Tests (v0.2.7)
# ============================================================================


def test_rename_session_basic(tmp_path: Path) -> None:
    """Test renaming a session updates metadata.name."""
    metadata = SessionMetadata.create("Original Name", "gpt-4.1", 2)
    session = Session(
        metadata=metadata,
        system_prompt="System",
        messages=[Message(role="user", content="Hello")],
    )
    filepath = save_session(session, tmp_path)

    rename_session(filepath, "New Name")

    loaded = load_session(filepath)
    assert loaded.metadata.name == "New Name"


def test_rename_session_updates_timestamp(tmp_path: Path) -> None:
    """Test renaming updates updated_at timestamp."""
    metadata = SessionMetadata.create("Test", "gpt-4.1", 0)
    metadata.updated_at = "2020-01-01T00:00:00"
    session = Session(metadata=metadata, system_prompt="", messages=[])

    import json

    filepath = tmp_path / f"session-{metadata.id}.json"
    with open(filepath, "w") as f:
        json.dump(session.to_dict(), f)

    rename_session(filepath, "Renamed")

    loaded = load_session(filepath)
    assert loaded.metadata.updated_at != "2020-01-01T00:00:00"


def test_rename_session_preserves_content(tmp_path: Path) -> None:
    """Test renaming preserves messages and system prompt."""
    metadata = SessionMetadata.create("Original", "gpt-4.1", 2)
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]
    session = Session(
        metadata=metadata,
        system_prompt="Be helpful",
        messages=messages,
    )
    filepath = save_session(session, tmp_path)

    rename_session(filepath, "New Name")

    loaded = load_session(filepath)
    assert loaded.system_prompt == "Be helpful"
    assert len(loaded.messages) == 2
    assert loaded.messages[0].content == "Hello"


def test_rename_session_strips_whitespace(tmp_path: Path) -> None:
    """Test rename strips leading/trailing whitespace."""
    metadata = SessionMetadata.create("Test", "gpt-4.1", 0)
    session = Session(metadata=metadata, system_prompt="", messages=[])
    filepath = save_session(session, tmp_path)

    rename_session(filepath, "  Padded Name  ")

    loaded = load_session(filepath)
    assert loaded.metadata.name == "Padded Name"


def test_rename_session_empty_name_raises(tmp_path: Path) -> None:
    """Test rename raises ValueError for empty name."""
    metadata = SessionMetadata.create("Test", "gpt-4.1", 0)
    session = Session(metadata=metadata, system_prompt="", messages=[])
    filepath = save_session(session, tmp_path)

    with pytest.raises(ValueError, match="cannot be empty"):
        rename_session(filepath, "")


def test_rename_session_whitespace_only_raises(tmp_path: Path) -> None:
    """Test rename raises ValueError for whitespace-only name."""
    metadata = SessionMetadata.create("Test", "gpt-4.1", 0)
    session = Session(metadata=metadata, system_prompt="", messages=[])
    filepath = save_session(session, tmp_path)

    with pytest.raises(ValueError, match="cannot be empty"):
        rename_session(filepath, "   ")


def test_rename_session_not_found(tmp_path: Path) -> None:
    """Test rename raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        rename_session(tmp_path / "nonexistent.json", "New Name")


# ============================================================================
# Session Deletion Tests (v0.2.8)
# ============================================================================


def test_delete_session_success(tmp_path: Path) -> None:
    """Test deleting an existing session file."""
    metadata = SessionMetadata.create("To Delete", "gpt-4.1", 2)
    session = Session(
        metadata=metadata,
        system_prompt="System",
        messages=[Message(role="user", content="Hello")],
    )
    filepath = save_session(session, tmp_path)
    assert filepath.exists()

    delete_session(filepath)

    assert not filepath.exists()


def test_delete_session_not_found(tmp_path: Path) -> None:
    """Test delete raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        delete_session(tmp_path / "nonexistent.json")


# ============================================================================
# Markdown Export Tests (v0.2.3f)
# ============================================================================


def test_export_session_markdown_basic() -> None:
    """Test basic Markdown export format."""
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
    ]

    md = export_session_markdown(messages)

    assert "# Chat Session" in md
    assert "**You:** Hello" in md
    assert "**Assistant:** Hi there!" in md
    assert "*Exported from chatty*" in md


def test_export_session_markdown_with_metadata() -> None:
    """Test export includes session name and model in header."""
    messages = [
        Message(role="user", content="Question"),
        Message(role="assistant", content="Answer"),
    ]

    md = export_session_markdown(
        messages,
        session_name="My Chat Session",
        model="gpt-4.1",
    )

    assert "# My Chat Session" in md
    assert "Model: gpt-4.1" in md


def test_export_session_markdown_skips_system() -> None:
    """Test export skips system messages."""
    messages = [
        Message(role="system", content="You are helpful"),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
    ]

    md = export_session_markdown(messages)

    assert "You are helpful" not in md
    assert "**You:** Hello" in md


def test_export_session_markdown_timestamp() -> None:
    """Test export includes timestamp."""
    messages = [Message(role="user", content="Test")]

    md = export_session_markdown(messages)

    assert "Exported:" in md


def test_save_markdown_export_creates_file(tmp_path: Path) -> None:
    """Test save_markdown_export creates file."""
    messages = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi"),
    ]

    filepath = save_markdown_export(messages, tmp_path)

    assert filepath.exists()
    assert filepath.suffix == ".md"
    assert "chatty-export-" in filepath.name


def test_save_markdown_export_content(tmp_path: Path) -> None:
    """Test saved Markdown file has correct content."""
    messages = [
        Message(role="user", content="What is Python?"),
        Message(role="assistant", content="Python is a programming language."),
    ]

    filepath = save_markdown_export(
        messages,
        tmp_path,
        session_name="Python Question",
        model="gpt-4.1",
    )

    content = filepath.read_text()
    assert "# Python Question" in content
    assert "Model: gpt-4.1" in content
    assert "What is Python?" in content


def test_save_markdown_export_creates_directory(tmp_path: Path) -> None:
    """Test save_markdown_export creates directory if needed."""
    new_dir = tmp_path / "exports" / "nested"
    assert not new_dir.exists()

    messages = [Message(role="user", content="Test")]
    save_markdown_export(messages, new_dir)

    assert new_dir.exists()
