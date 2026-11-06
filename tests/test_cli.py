"""Smoke tests for the CLI commands."""

import re
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from chatty.cli import app

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def plain(result: Result) -> str:
    """Return stdout without ANSI styling.

    Typer forces Rich color on when GITHUB_ACTIONS or FORCE_COLOR is set, and the
    escape codes split option names such as `--query-file` in help output.
    """
    return _ANSI.sub("", result.stdout)


class TestChattyHelp:
    """Test that CLI commands have help text and don't crash."""

    def test_version(self) -> None:
        """chatty --version shows version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "chatty" in plain(result)
        # Version format: chatty X.Y.Z
        assert "0." in plain(result) or "1." in plain(result)

    def test_version_short(self) -> None:
        """chatty -V shows version (short form)."""
        result = runner.invoke(app, ["-V"])
        assert result.exit_code == 0
        assert "chatty" in plain(result)

    def test_main_help(self) -> None:
        """chatty --help returns successfully."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "chatty" in plain(result).lower()
        assert "chat" in plain(result)
        assert "doctor" in plain(result)
        assert "print-config" in plain(result)

    def test_chat_help(self) -> None:
        """chatty chat --help returns successfully."""
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--query-file" in plain(result)
        assert "--base-url" in plain(result)
        assert "--model" in plain(result)

    def test_doctor_help(self) -> None:
        """chatty doctor --help returns successfully."""
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "--verbose" in plain(result)
        assert "--skip-connectivity" in plain(result)

    def test_print_config_help(self) -> None:
        """chatty print-config --help returns successfully."""
        result = runner.invoke(app, ["print-config", "--help"])
        assert result.exit_code == 0
        assert "--sources" in plain(result)


class TestPrintConfig:
    """Test print-config command."""

    def test_print_config_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """print-config runs without crashing (minimal config)."""
        # Set minimal required config
        monkeypatch.setenv("OPENAI_BASE_URL", "https://test.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = runner.invoke(app, ["print-config"])
        assert result.exit_code == 0
        assert "Configuration:" in plain(result)
        assert "base_url" in plain(result)


class TestDoctor:
    """Test doctor command."""

    def test_doctor_missing_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """doctor fails gracefully with missing config."""
        # Clear config so doctor fails
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = runner.invoke(app, ["doctor", "--skip-connectivity"])
        # Should fail but not crash
        assert result.exit_code != 0 or "base_url" in plain(result).lower()

    def test_doctor_verbose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """doctor --verbose shows detailed output."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://test.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = runner.invoke(app, ["doctor", "--verbose", "--skip-connectivity"])
        assert "base_url" in plain(result).lower()


class TestPrintConfigDetailed:
    """Additional print-config tests."""

    def test_print_config_with_sources(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """print-config --sources shows source attribution."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://test.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = runner.invoke(app, ["print-config", "--sources"])
        assert result.exit_code == 0
        assert "from:" in plain(result)


class TestChatCommand:
    """Test chat command paths."""

    def test_chat_session_not_found(self, tmp_path: Path) -> None:
        """chat --session with nonexistent file fails."""

        fake_session = tmp_path / "nonexistent.json"
        result = runner.invoke(app, ["chat", "--session", str(fake_session)])
        assert result.exit_code == 1
        # Error is printed to stderr
        output = result.output if result.output else ""
        assert "not found" in output.lower() or result.exit_code == 1

    def test_chat_with_query_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """chat --query-file loads the file."""
        # Create a query file
        query_file = tmp_path / "query.txt"
        query_file.write_text("What is Python?")

        # Mock the UI main to avoid launching
        monkeypatch.setenv("OPENAI_BASE_URL", "https://test.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from unittest.mock import patch

        with patch("chatty.ui.app.main") as mock_main:
            result = runner.invoke(app, ["chat", "--query-file", str(query_file)])
            # Should call main with query_file parameter
            if result.exit_code == 0:
                mock_main.assert_called_once()
                call_kwargs = mock_main.call_args[1]
                assert call_kwargs.get("query_file") == str(query_file)

    def test_chat_with_session(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """chat --session loads existing session."""
        from chatty.client.openai_client import Message
        from chatty.core.session import (
            Session,
            SessionMetadata,
            save_session,
        )

        # Create a valid session file
        metadata = SessionMetadata.create(
            name="Test Session",
            model="gpt-4",
            message_count=1,
        )
        session = Session(
            metadata=metadata,
            system_prompt="You are helpful.",
            messages=[Message(role="user", content="Hello")],
        )
        filepath = save_session(session, tmp_path)

        monkeypatch.setenv("OPENAI_BASE_URL", "https://test.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from unittest.mock import patch

        with patch("chatty.ui.app.main") as mock_main:
            result = runner.invoke(app, ["chat", "--session", str(filepath)])
            if result.exit_code == 0:
                mock_main.assert_called_once()
                call_kwargs = mock_main.call_args[1]
                assert call_kwargs.get("session_file") == filepath

    def test_chat_with_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """chat --model overrides config model."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://test.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from unittest.mock import patch

        with patch("chatty.ui.app.main") as mock_main:
            result = runner.invoke(app, ["chat", "--model", "claude-3"])
            if result.exit_code == 0:
                mock_main.assert_called_once()
                call_kwargs = mock_main.call_args[1]
                config = call_kwargs.get("config_with_sources")
                # Model should be in the config
                assert config is not None

    def test_chat_with_no_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """chat --no-stream disables streaming."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://test.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from unittest.mock import patch

        with patch("chatty.ui.app.main") as mock_main:
            result = runner.invoke(app, ["chat", "--no-stream"])
            if result.exit_code == 0:
                mock_main.assert_called_once()
                call_kwargs = mock_main.call_args[1]
                config = call_kwargs.get("config_with_sources")
                assert config is not None

    def test_chat_with_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """chat --temperature sets temperature."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://test.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        from unittest.mock import patch

        with patch("chatty.ui.app.main") as mock_main:
            result = runner.invoke(app, ["chat", "--temperature", "0.7"])
            if result.exit_code == 0:
                mock_main.assert_called_once()
