"""Run the laptop guide's code blocks with the installed commands, outside the checkout."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def check(workspace: Path) -> None:
    guide = (Path(__file__).resolve().parents[1] / "docs/laptop-quickstart.md").read_text("utf-8")
    windows = sys.platform == "win32"
    shell = "powershell" if windows else "bash"
    blocks = re.findall(r"^```(\w+)\n(.*?)^```$", guide, re.M | re.S)
    lines = [line for lang, code in blocks if lang in (shell, "sh") for line in code.splitlines()]
    cwd = workspace.parent
    assert not workspace.exists(), f"Expected a new workspace: {workspace}"
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("OPENAI_", "CHATTY_", "LITKIT_", "HF_", "HUGGING_FACE_"))
        and key not in {"API_KEY", "BASE_URL", "RAG_WORKSPACE"}
    }
    # Exercise an explicit config and a non-ASCII name in an ordinary Windows pipe.
    config = cwd / "profile.toml"
    config.write_text(
        f"rag_workspace = {json.dumps(str(workspace), ensure_ascii=False)}\n", encoding="utf-8"
    )
    env["CHATTY_CONFIG"] = str(config)
    if windows:
        env.pop("PYTHONIOENCODING", None)
        env["PYTHONUTF8"] = "0"
        script = cwd / "quickstart.ps1"
        commands = []
        for line in lines:
            commands.append(line)
            if line.startswith(("chatty ", "text-fetch ", "litkit ", "$workspace = chatty")):
                commands.append("if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }")
        script.write_text("\n".join(commands), encoding="utf-8-sig")
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
        ]
    else:
        command = ["/bin/bash", "-ec", "\n".join(lines)]
    result = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE, timeout=900)
    output = result.stdout.decode("utf-8")
    sys.stdout.buffer.write(result.stdout)
    sys.stdout.flush()
    result.check_returncode()
    assert (workspace / "example/tars/sample.tar.gz").is_file(), workspace
    assert (workspace / "indices/papers.faiss").is_file(), workspace
    assert (workspace / "indices/chunks.faiss").is_file(), workspace
    assert "allenai/specter2_base: missing or incomplete" in output
    assert "sentence-transformers/all-mpnet-base-v2: missing or incomplete" in output
    assert "All checks passed." in output
    assert "Rainfall at the cedar grove [chunk" in output
    assert "The copper rain gauge collected seventeen millimeters of rain" in output


if __name__ == "__main__":
    if len(sys.argv) > 1:
        check(Path(sys.argv[1]).resolve())
    else:
        with TemporaryDirectory() as temp:
            check(Path(temp).resolve() / "研究 field notes")
