"""Smoke tests for the CLI commands."""

import pytest
from typer.testing import CliRunner

from chatty.cli import app

runner = CliRunner()


class TestChattyHelp:
    """Test that CLI commands have help text and don't crash."""

    def test_main_help(self) -> None:
        """chatty --help returns successfully."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "chatty" in result.stdout.lower()
        assert "chat" in result.stdout
        assert "doctor" in result.stdout
        assert "print-config" in result.stdout

    def test_chat_help(self) -> None:
        """chatty chat --help returns successfully."""
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "--query-file" in result.stdout
        assert "--base-url" in result.stdout
        assert "--model" in result.stdout

    def test_doctor_help(self) -> None:
        """chatty doctor --help returns successfully."""
        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "--verbose" in result.stdout
        assert "--skip-connectivity" in result.stdout

    def test_print_config_help(self) -> None:
        """chatty print-config --help returns successfully."""
        result = runner.invoke(app, ["print-config", "--help"])
        assert result.exit_code == 0
        assert "--sources" in result.stdout


class TestPrintConfig:
    """Test print-config command."""

    def test_print_config_runs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """print-config runs without crashing (minimal config)."""
        # Set minimal required config
        monkeypatch.setenv("OPENAI_BASE_URL", "https://test.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = runner.invoke(app, ["print-config"])
        assert result.exit_code == 0
        assert "Configuration:" in result.stdout
        assert "base_url" in result.stdout


class TestDoctor:
    """Test doctor command."""

    def test_doctor_missing_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """doctor fails gracefully with missing config."""
        # Clear config so doctor fails
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = runner.invoke(app, ["doctor", "--skip-connectivity"])
        # Should fail but not crash
        assert result.exit_code != 0 or "base_url" in result.stdout.lower()
