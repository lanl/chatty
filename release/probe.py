"""Run Chatty's installed entry point and load its LitKit retrieval imports in that Python."""

import json
import platform
import sys
from importlib.metadata import distribution, version
from pathlib import Path

# Invoke the same callable as the generated chatty console script.
(entry,) = distribution("chatty").entry_points.select(group="console_scripts", name="chatty")
entry.load()(args=["--version"], standalone_mode=False)

import litkit.cli  # noqa: E402
from litkit.cli import deps, get_chunks, search_chunks_constrained, shortlist_papers  # noqa: E402
from litkit.db import connect_db  # noqa: E402

import chatty  # noqa: E402

assert all(
    callable(f) for f in (get_chunks, search_chunks_constrained, shortlist_papers, connect_db)
)
assert deps().faiss_available
import faiss  # noqa: E402
import torch  # noqa: E402

assert torch.version.cuda is None
paths = {module.__name__: str(module.__file__) for module in (chatty, litkit, litkit.cli, faiss)}
for path in paths.values():
    assert Path(path).resolve().is_relative_to(Path(sys.prefix).resolve()), path

print(
    json.dumps(
        {
            "executable": sys.executable,
            "base_prefix": sys.base_prefix,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "modules": paths,
            "faiss_module_version": faiss.__version__,
            "faiss_cpu_distribution": version("faiss-cpu"),
            "torch": torch.__version__,
        }
    )
)
