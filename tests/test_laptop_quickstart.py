"""Public commands used before a laptop user configures an LLM."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from chatty.cli import app
from chatty.rag.models import EmbeddingModel

runner = CliRunner()


def test_workspace_selection_and_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "settings.toml"  # An explicit config can have any filename.
    config.write_text('rag_workspace = "from toml"\n')
    monkeypatch.setenv("CHATTY_CONFIG", str(config))
    for var, selection in (
        (None, "from toml"),
        ("LITKIT_WORKSPACE", "native"),
        ("CHATTY_RAG_WORKSPACE", "chosen"),
    ):
        if var:
            monkeypatch.setenv(var, selection)
        result = runner.invoke(app, ["workspace"])
        assert result.exit_code == 0
        assert Path(result.stdout.strip()) == tmp_path / selection
        assert not (tmp_path / selection).exists()

    for _ in range(2):
        assert runner.invoke(app, ["workspace", "--example"]).exit_code == 0
    article = tmp_path / "chosen/example/articles/rain-gauge.xml"
    assert "seventeen millimeters" in article.read_text()
    article.write_text("my edits")
    assert runner.invoke(app, ["workspace", "--example"]).exit_code != 0
    assert article.read_text() == "my edits"


@pytest.mark.parametrize("missing", ["both", "papers.faiss", "chunks.faiss", "database", "neither"])
def test_retrieval_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LITKIT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("LITKIT_DB_FILE", "custom.sqlite3")
    monkeypatch.setattr(EmbeddingModel, "ready", lambda *_: True)
    (tmp_path / "indices").mkdir()
    for name in ("papers.faiss", "chunks.faiss"):
        if missing not in (name, "both"):
            (tmp_path / "indices" / name).touch()
    (tmp_path / "sqlite").mkdir()
    if missing != "database":
        (tmp_path / "sqlite/custom.sqlite3").touch()
    result = runner.invoke(app, ["doctor", "--retrieval"])
    assert (result.exit_code == 0) == (missing == "neither")
    assert "base_url" not in result.stdout and "api_key" not in result.stdout
    assert "custom.sqlite3" in result.stdout
    if missing == "both":
        assert "indices missing" in result.stdout
    elif missing.endswith(".faiss"):
        assert "indices partial" in result.stdout and f"Missing {missing}" in result.stdout
    elif missing == "database":
        assert "SQLite database missing" in result.stdout


def test_llm_diagnostics_show_local_failure_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LITKIT_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("CHATTY_RAG_PROVIDER", "litkit")
    monkeypatch.setattr(EmbeddingModel, "ready", lambda *_: False)
    result = runner.invoke(app, ["doctor", "--skip-connectivity"])
    assert result.exit_code == 1
    assert "missing or incomplete" in result.stdout
    assert "chatty models --download" in result.stdout
    assert "LLM configuration" in result.stdout and "api_key: (not set)" in result.stdout


def test_model_setup_reuses_ready_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    ready = {"allenai/specter2_base"}
    downloads = []

    def download(model: EmbeddingModel, cache: Path) -> None:
        downloads.append((model.repo, cache))
        ready.add(model.repo)

    monkeypatch.setattr(EmbeddingModel, "ready", lambda model, _: model.repo in ready)
    monkeypatch.setattr(EmbeddingModel, "download", download)
    status = runner.invoke(app, ["models"])
    assert status.exit_code == 0 and not downloads
    assert f"cache: {tmp_path / 'hub'}" in status.stdout
    for _ in range(2):
        assert runner.invoke(app, ["models", "--download"]).exit_code == 0
    assert downloads == [("sentence-transformers/all-mpnet-base-v2", tmp_path)]


def test_download_repairs_truncated_weight_in_existing_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import zipfile
    from types import SimpleNamespace

    from chatty.rag.models import MODELS

    model = MODELS[0]
    snapshot = tmp_path / "hub" / "models--allenai--specter2_base" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    for name in model.files:
        (snapshot / name).write_text('{"saved": true}')
    (snapshot / model.weights).write_bytes(b"PK\x03\x04")
    monkeypatch.setitem(
        sys.modules,
        "litkit.embeddings.hf_local",
        SimpleNamespace(local_snapshot_dir=lambda *_args, **_kwargs: snapshot),
    )
    fetched = []

    def download(
        repo: str, name: str, *, revision: str, local_dir: str, force_download: bool
    ) -> str:
        assert force_download
        fetched.append((repo, name, revision))
        path = Path(local_dir) / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("archive/data.pkl", b"model metadata")
        return str(path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            snapshot_download=lambda *_args, **_kwargs: str(snapshot),
            hf_hub_download=download,
        ),
    )
    assert not model.ready(tmp_path)
    model.download(tmp_path)
    assert model.ready(tmp_path)
    assert fetched == [(model.repo, model.weights, snapshot.name)]
