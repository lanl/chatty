"""Snapshot tests for chatty UI.

Uses pytest-textual-snapshot for visual regression testing of the Textual app.
Snapshots capture the terminal output state for comparison.

Run with --snapshot-update to create/update snapshot baselines.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

    from chatty.config import ConfigWithSources


@pytest.fixture
def mock_config(monkeypatch: MonkeyPatch, tmp_path: Path) -> ConfigWithSources:
    """Provide isolated config for UI tests."""
    import os

    from chatty.config import Config, ConfigWithSources

    # Isolate from user config
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))

    # Clear CHATTY_ and OPENAI_ env vars
    for key in list(os.environ.keys()):
        if key.startswith("CHATTY_") or key.startswith("OPENAI_"):
            monkeypatch.delenv(key, raising=False)

    # Create minimal test config
    config = Config(
        base_url="http://localhost:1234/v1",
        model="test-model",
        stream=True,
    )
    return ConfigWithSources(config=config, sources={})


def test_app_renders(snap_compare: Callable[..., bool], mock_config: ConfigWithSources) -> None:
    """Verify app renders without crash and matches snapshot."""
    from chatty.ui.app import ChatApp

    app = ChatApp(config_with_sources=mock_config)

    # snap_compare will render the app and compare to stored snapshot
    assert snap_compare(app)
