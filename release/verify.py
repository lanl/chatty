"""Run the README commands in an empty directory and record the verified release install."""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent


def checked_download(url: str, expected: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        data: bytes = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {url}")
    return data


def verify(target: str, evidence: Path) -> None:
    os_name, arch = {"macos-arm64": ("Darwin", "arm64"), "windows-x64": ("Windows", "AMD64")}[
        target
    ]
    assert (platform.system(), platform.machine()) == (os_name, arch)
    readme = (HERE.parent / "README.md").read_text(encoding="utf-8")
    block = readme.split(f"<!-- install-{target} -->\n", 1)[1].split("\n```", 1)[0]
    commands = block.split("\n", 1)[1]
    sources = json.loads((HERE / "sources.json").read_text())
    base = f"https://github.com/lanl/chatty/releases/download/{sources['tag']}"
    manifest_sha = (HERE / "manifest.sha256").read_text().split()[0]
    manifest = json.loads(checked_download(f"{base}/manifest.json", manifest_sha))
    assert (
        platform.python_version()
        == manifest["python"]
        == (HERE / ".python-version").read_text().strip()
    )
    assert manifest["tag"] == sources["tag"]
    for name, source in sources["projects"].items():
        project = manifest["projects"][name]
        assert project["revision"] == source["revision"] and project["version"] == source["version"]
    lock = manifest["platforms"][target]
    requirements = lock["requirements"]
    assert commands.split()[-1] == requirements["url"]
    checked_download(requirements["url"], requirements["sha256"])

    with TemporaryDirectory() as tmp:
        cwd = Path(tmp).resolve()
        assert all(not (p / ".git").exists() for p in (cwd, *cwd.parents))
        windows = os_name == "Windows"
        script = cwd / ("install.ps1" if windows else "install.sh")
        guard = (
            "$ErrorActionPreference='Stop'\n$PSNativeCommandUseErrorActionPreference=$true"
            if windows
            else "set -eu"
        )
        script.write_text(guard + "\n" + commands + "\n", encoding="utf-8")
        shell = ["pwsh", "-NoProfile", "-File"] if windows else ["bash"]
        env = {**os.environ, "PIP_REPORT": str(cwd / "pip-report.json"), "PIP_NO_CACHE_DIR": "1"}
        print(commands, flush=True)
        subprocess.run([*shell, str(script)], cwd=cwd, env=env, check=True)

        report = json.loads((cwd / "pip-report.json").read_text(encoding="utf-8"))
        installed = {}
        for row in report["install"]:
            name = re.sub(r"[-_.]+", "-", row["metadata"]["name"]).lower()
            download = row["download_info"]
            installed[name] = {
                "version": row["metadata"]["version"],
                "url": download["url"],
                "sha256": download["archive_info"]["hashes"]["sha256"],
            }
        assert installed == lock["packages"], "pip installed artifacts outside the manifest"
        scripts = cwd / "chatty-suite" / ("Scripts" if windows else "bin")
        suffix = ".exe" if windows else ""
        python = str(scripts / f"python{suffix}")
        subprocess.run([python, "-m", "pip", "check"], cwd=cwd, check=True)
        versions = {}
        for name, source in sources["projects"].items():
            cmd = str(scripts / (name + suffix))
            result = subprocess.check_output([cmd, "--version"], cwd=cwd, text=True).strip()
            assert result.split()[-1] == source["version"], result
            subprocess.run([cmd, "--help"], cwd=cwd, stdout=subprocess.DEVNULL, check=True)
            versions[name] = result
        probe = subprocess.check_output([python, "-I", str(HERE / "probe.py")], cwd=cwd, text=True)
        runtime = json.loads(probe.splitlines()[-1])
        assert runtime["executable"] == python and runtime["machine"] == arch
        assert runtime["python"] == manifest["python"]
        assert runtime["faiss_cpu_distribution"] == installed["faiss-cpu"]["version"]
        run_id = os.environ.get("GITHUB_RUN_ID")
        result_data = {
            "run": f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{run_id}"
            if run_id
            else None,
            "attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "workflow_revision": os.environ.get("GITHUB_SHA"),
            "runner_image": os.environ.get("ImageVersion", platform.platform()),  # noqa: SIM112
            "platform": target,
            "commands": commands.splitlines(),
            "cwd": str(cwd),
            "manifest_sha256": manifest_sha,
            "requirements_sha256": requirements["sha256"],
            "verified_wheels": len(installed),
            "pip_check": "passed",
            "cli": versions,
            "cli_help": "passed",
            "pip": report["pip_version"],
            "runtime": runtime,
        }
        data = json.dumps(result_data, indent=2)
        print(data, flush=True)
        result_data["packages"] = installed
        evidence.write_text(
            json.dumps(result_data, indent=2) + "\n", encoding="utf-8", newline="\n"
        )


if __name__ == "__main__":
    verify(sys.argv[1], Path(sys.argv[2]).resolve())
