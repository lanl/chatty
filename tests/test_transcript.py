"""Tests for JSONL transcript logging."""

import json
from pathlib import Path

import pytest

from chatty.config import Config
from chatty.core.transcript import TranscriptLogger


@pytest.fixture
def transcript_dir(tmp_path: Path) -> Path:
    """Create a temporary transcript directory."""
    transcript_path = tmp_path / "transcripts"
    return transcript_path


@pytest.fixture
def enabled_config(transcript_dir: Path) -> Config:
    """Create a config with transcript logging enabled."""
    return Config(
        transcript_enabled=True,
        transcript_path=str(transcript_dir),
    )


@pytest.fixture
def disabled_config() -> Config:
    """Create a config with transcript logging disabled."""
    return Config(
        transcript_enabled=False,
    )


def test_transcript_disabled_returns_none(disabled_config: Config) -> None:
    """When transcript_enabled=False, start_session returns None."""
    logger = TranscriptLogger(disabled_config)
    result = logger.start_session()
    assert result is None
    assert not logger.enabled


def test_transcript_enabled_property(enabled_config: Config, disabled_config: Config) -> None:
    """Test the enabled property reflects config."""
    enabled_logger = TranscriptLogger(enabled_config)
    disabled_logger = TranscriptLogger(disabled_config)

    assert enabled_logger.enabled is True
    assert disabled_logger.enabled is False


def test_transcript_creates_directory(enabled_config: Config, transcript_dir: Path) -> None:
    """Transcript logger creates directory if missing."""
    assert not transcript_dir.exists()

    logger = TranscriptLogger(enabled_config)
    result = logger.start_session()

    assert transcript_dir.exists()
    assert result is not None
    assert result.parent == transcript_dir
    logger.close()


def test_transcript_file_naming(enabled_config: Config) -> None:
    """Transcript file is named with timestamp pattern."""
    logger = TranscriptLogger(enabled_config)
    result = logger.start_session()

    assert result is not None
    # Should match pattern: chatty-YYYY-MM-DD-HHMMSS.jsonl
    assert result.name.startswith("chatty-")
    assert result.suffix == ".jsonl"
    logger.close()


def test_transcript_writes_jsonl_format(enabled_config: Config) -> None:
    """Each line in transcript is valid JSON."""
    logger = TranscriptLogger(enabled_config)
    file_path = logger.start_session()

    logger.log_message("user", "Hello")
    logger.log_message("assistant", "Hi there!")
    logger.close()

    assert file_path is not None
    content = file_path.read_text()
    lines = content.strip().split("\n")

    assert len(lines) == 2
    for line in lines:
        # Should be valid JSON
        data = json.loads(line)
        assert "timestamp" in data
        assert "role" in data
        assert "content" in data


def test_transcript_logs_user_message(enabled_config: Config) -> None:
    """User messages are logged with correct format."""
    logger = TranscriptLogger(enabled_config)
    file_path = logger.start_session()

    logger.log_message("user", "Test question")
    logger.close()

    assert file_path is not None
    content = file_path.read_text()
    data = json.loads(content.strip())

    assert data["role"] == "user"
    assert data["content"] == "Test question"
    assert "timestamp" in data


def test_transcript_logs_assistant_with_metadata(enabled_config: Config) -> None:
    """Assistant messages include model, response_time, and tokens."""
    logger = TranscriptLogger(enabled_config)
    file_path = logger.start_session()

    logger.log_message(
        "assistant",
        "Test response",
        model="gpt-4.1",
        response_time_s=2.345,
        tokens=150,
    )
    logger.close()

    assert file_path is not None
    content = file_path.read_text()
    data = json.loads(content.strip())

    assert data["role"] == "assistant"
    assert data["content"] == "Test response"
    assert data["model"] == "gpt-4.1"
    assert data["response_time_s"] == 2.35  # Rounded to 2 decimals
    assert data["tokens"] == 150


def test_transcript_optional_metadata_omitted(enabled_config: Config) -> None:
    """Optional metadata fields are omitted when not provided."""
    logger = TranscriptLogger(enabled_config)
    file_path = logger.start_session()

    logger.log_message("assistant", "Response without metadata")
    logger.close()

    assert file_path is not None
    content = file_path.read_text()
    data = json.loads(content.strip())

    assert "model" not in data
    assert "response_time_s" not in data
    assert "tokens" not in data


def test_transcript_no_op_when_disabled(disabled_config: Config, tmp_path: Path) -> None:
    """Logging is a no-op when disabled."""
    logger = TranscriptLogger(disabled_config)
    result = logger.start_session()

    assert result is None
    logger.log_message("user", "This should not be written")
    logger.close()

    # No files should be created
    assert list(tmp_path.iterdir()) == []


def test_transcript_get_file_path(enabled_config: Config) -> None:
    """get_file_path returns the current transcript path."""
    logger = TranscriptLogger(enabled_config)

    assert logger.get_file_path() is None

    file_path = logger.start_session()
    assert logger.get_file_path() == file_path

    logger.close()


def test_transcript_error_messages(enabled_config: Config) -> None:
    """Error messages are logged correctly."""
    logger = TranscriptLogger(enabled_config)
    file_path = logger.start_session()

    logger.log_message("error", "Something went wrong")
    logger.close()

    assert file_path is not None
    content = file_path.read_text()
    data = json.loads(content.strip())

    assert data["role"] == "error"
    assert data["content"] == "Something went wrong"


def test_transcript_logs_rag_metadata(enabled_config: Config) -> None:
    """User messages include RAG metadata when provided."""
    from chatty.rag.provider import RAGMetadata

    logger = TranscriptLogger(enabled_config)
    file_path = logger.start_session()

    rag_metadata = RAGMetadata(
        sources=[],
        retrieval_time_s=1.23,
        chunk_count=5,
        rewritten_query="What are the side effects of HIV treatments?",
        query_mode="FOLLOWUP",
    )

    logger.log_message(
        "user",
        "What about side effects?",
        rag_metadata=rag_metadata,
    )
    logger.close()

    assert file_path is not None
    content = file_path.read_text()
    data = json.loads(content.strip())

    assert data["role"] == "user"
    assert data["content"] == "What about side effects?"
    assert data["query_mode"] == "FOLLOWUP"
    assert data["rewritten_query"] == "What are the side effects of HIV treatments?"
    assert data["retrieval_time_s"] == 1.23
    assert data["chunk_count"] == 5


def test_transcript_rag_metadata_omitted_when_none(enabled_config: Config) -> None:
    """RAG metadata fields omitted when not provided."""
    logger = TranscriptLogger(enabled_config)
    file_path = logger.start_session()

    logger.log_message("user", "Simple question")
    logger.close()

    assert file_path is not None
    content = file_path.read_text()
    data = json.loads(content.strip())

    assert "query_mode" not in data
    assert "rewritten_query" not in data
    assert "retrieval_time_s" not in data
    assert "chunk_count" not in data
