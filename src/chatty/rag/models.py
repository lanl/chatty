"""The two Hugging Face snapshots used by LANL LitKit's offline loaders."""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(frozen=True)
class EmbeddingModel:
    repo: str
    download_mb: int
    weights: str
    extra_files: tuple[str, ...] = ()

    @property
    def files(self) -> tuple[str, ...]:
        return (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            self.weights,
            *self.extra_files,
        )

    def ready(self, cache: Path) -> bool:
        # Use the same ref (or sole snapshot) as the installed LitKit loader.
        from litkit.embeddings.hf_local import local_snapshot_dir

        try:
            snapshot = Path(local_snapshot_dir(self.repo, hf_home=cache))
        except FileNotFoundError:
            return False
        # Both loaders accept local safetensors or the original PyTorch weights.
        # Check the model inputs as well: an interrupted Hub download can leave
        # a valid ref and snapshot directory before the weights arrive.
        required = ("config.json", *self.extra_files)
        # Transformers prefers safetensors when both copies are present.
        weights = snapshot / "model.safetensors"
        if not weights.exists():
            weights = snapshot / "pytorch_model.bin"
        return (
            all(_usable(snapshot / name) for name in required)
            and _usable(snapshot / "tokenizer_config.json")
            and _usable(snapshot / "vocab.txt")
            and all(
                _usable(path)
                for path in (snapshot / "tokenizer.json", snapshot / "special_tokens_map.json")
                if path.exists()
            )
            and _usable(weights)
        )

    def download(self, cache: Path) -> None:
        from huggingface_hub import hf_hub_download, snapshot_download

        # The Hub maintains refs/main and resumes incomplete downloads. Select
        # the runtime files to avoid alternate ONNX/OpenVINO/duplicate weights.
        snapshot = Path(
            snapshot_download(self.repo, cache_dir=cache / "hub", allow_patterns=list(self.files))
        )
        if self.ready(cache):
            return

        # An existing but damaged file is a Hub cache hit. Stage just the bad
        # files on the same volume and replace each only after a verified fetch.
        # This also repairs Windows caches where snapshots contain ordinary files.
        with TemporaryDirectory(dir=cache) as staging:
            for name in self.files:
                target = snapshot / name
                if not _usable(target):
                    fresh = Path(
                        hf_hub_download(
                            self.repo,
                            name,
                            revision=snapshot.name,
                            local_dir=staging,
                            force_download=True,
                        )
                    )
                    if not _usable(fresh):
                        raise OSError(f"Invalid model file from {self.repo}: {name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(fresh, target)
        if not self.ready(cache):
            raise OSError(
                f"Could not load the model files in {snapshot}. Check for an invalid local model.safetensors override."
            )


def _usable(path: Path) -> bool:
    """Detect incomplete configs, weight archives and Git LFS pointer files offline."""
    if not path.is_file() or not path.stat().st_size:
        return False
    if path.suffix == ".json":
        try:
            return bool(json.loads(path.read_bytes()))
        except (ValueError, UnicodeError):
            return False
    if path.suffix == ".safetensors":
        from safetensors import SafetensorError, safe_open

        try:
            # Validates the header and tensor offsets without loading the weights.
            with safe_open(path, framework="np") as tensors:
                return bool(tensors.keys())
        except SafetensorError:
            return False
    if path.suffix == ".bin":
        try:
            with zipfile.ZipFile(path) as weights:
                return any(name.endswith("/data.pkl") for name in weights.namelist())
        except zipfile.BadZipFile:
            return False
    return True


MODELS = (
    EmbeddingModel("allenai/specter2_base", 441, "pytorch_model.bin"),
    EmbeddingModel(
        "sentence-transformers/all-mpnet-base-v2",
        439,
        "model.safetensors",
        (
            "modules.json",
            "1_Pooling/config.json",
            "sentence_bert_config.json",
            "config_sentence_transformers.json",
        ),
    ),
)


def model_cache(workspace: Path) -> Path:
    # LitKit selects HF_HOME first and defaults to <workspace>/hf_cache.
    return Path(os.environ.get("HF_HOME", workspace / "hf_cache")).absolute()
