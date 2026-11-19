"""Exercise the published installer and retain its release provenance in a disposable profile."""

import ctypes
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
WINDOWS = sys.platform == "win32"


def checked_download(url: str, expected: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        data: bytes = response.read()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {url}")
    return data


def install_environment(tmp: Path) -> dict[str, str]:
    if sys.platform == "win32":
        # The workflow creates this local standard user specifically for the registry/PATH test.
        assert os.getlogin().lower() == "chattytest" and not ctypes.windll.shell32.IsUserAnAdmin()
        assert Path.home().name.lower() == "chattytest", Path.home()
        system = Path(os.environ["SYSTEMROOT"])
        paths = [system / "System32", system / "System32/WindowsPowerShell/v1.0"]
        env = {**os.environ, "PATH": os.pathsep.join(map(str, paths))}
    else:
        assert os.getuid() != 0
        home, tools = tmp / "home with spaces", tmp / "tools"
        home.mkdir()
        tools.mkdir()
        for name in [
            "uname",
            "sw_vers",
            "cut",
            "mkdir",
            "mktemp",
            "rm",
            "curl",
            "shasum",
            "plutil",
            "readlink",
            "cat",
            "tar",
            "ln",
            "locale",
            "mv",
            "grep",
        ]:
            path = shutil.which(name, path="/usr/bin:/bin:/usr/sbin")
            assert path, name
            (tools / name).symlink_to(path)
        env = {"HOME": str(home), "SHELL": "/bin/zsh", "PATH": str(tools), "TMPDIR": str(tmp)}
    for name in ("git", "cc", "cl", "clang", "gcc", "uv", "python", "python3", "py"):
        assert shutil.which(name, path=env["PATH"]) is None, name
    return env


def fresh_shell(command: str, cwd: Path, env: dict[str, str]) -> str:
    if WINDOWS:
        # A child of the original shell otherwise inherits its pre-install PATH.
        command = (
            "$ErrorActionPreference='Stop'; "
            "$env:Path += ';' + [Environment]::GetEnvironmentVariable('Path','User'); " + command
        )
        args = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
    else:
        args = ["/bin/zsh", "-ic", command]
    return subprocess.check_output(
        args, cwd=cwd, env=env, text=True, stdin=subprocess.DEVNULL
    ).strip()


def command_path(name: str, cwd: Path, env: dict[str, str]) -> Path:
    lookup = (
        f"(Get-Command {name} -CommandType Application).Source" if WINDOWS else f"command -v {name}"
    )
    return Path(fresh_shell(lookup, cwd, env))


def install(command: str, cwd: Path, env: dict[str, str]) -> None:
    shell = (
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
        if WINDOWS
        else ["/bin/sh", "-c"]
    )
    subprocess.run([*shell, command], cwd=cwd, env=env, stdin=subprocess.DEVNULL, check=True)


def exercise_upgrade(
    command: str,
    cwd: Path,
    work: Path,
    env: dict[str, str],
    root: Path,
    user_home: Path,
    manifest_sha: str,
) -> tuple[Path, list[str]]:
    def active_runtime() -> Path:
        runtime = command_path("chatty", work, env).parent.resolve().parent
        assert runtime.parent == root.resolve() / "runtimes", runtime
        return runtime

    original = active_runtime()
    python_path = Path("Scripts/python.exe" if WINDOWS else "bin/python")
    config = user_home / ".config/chatty/config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('session_path = "./sessions"\nrag_workspace = "./workspace"\n')
    corpus = work / "corpus/paper.xml"
    corpus.parent.mkdir()
    corpus.write_text("<article><body><p>Corpus fixture.</p></body></article>")
    index = work / "workspace/indices/chunks.faiss"
    subprocess.run(
        [str(original / python_path), "-I", str(HERE / "probe_data.py"), "create"],
        cwd=work,
        env=env,
        check=True,
    )
    (session_file,) = (work / "sessions").glob("*.json")
    data = {p: p.read_bytes() for p in (corpus, index, config, session_file)}

    install(command, cwd, env)
    assert active_runtime() == original, "Reinstall duplicated the runtime"
    assert all(p.read_bytes() == contents for p, contents in data.items())

    # The first suite has no published predecessor. Give the disposable installation a
    # different release marker to exercise the same replacement path as the next release.
    (original / "manifest.sha256").write_text("0" * 64)
    install(command, cwd, env)
    active = active_runtime()
    assert active != original and (active / "manifest.sha256").read_text().strip() == manifest_sha
    assert all(
        p.read_bytes() == contents for p, contents in data.items()
    ), "Upgrade changed user data"
    python = str(active / python_path)
    subprocess.run(
        [python, "-I", str(HERE / "probe_data.py"), "read"], cwd=work, env=env, check=True
    )
    return active, [p.name for p in data]


def verify(target: str, evidence: Path) -> None:
    os_name, arch = {"macos-arm64": ("Darwin", "arm64"), "windows-x64": ("Windows", "AMD64")}[
        target
    ]
    assert (platform.system(), platform.machine()) == (os_name, arch)
    readme = (HERE.parent / "README.md").read_text(encoding="utf-8")
    block = readme.split(f"<!-- install-{target} -->\n", 1)[1].split("\n```", 1)[0]
    command = block.split("\n", 1)[1]
    installer = "install.ps1" if WINDOWS else "install.sh"
    url = re.search(
        r"https://raw\.githubusercontent\.com/lanl/chatty/[0-9a-f]{40}/release/"
        + re.escape(installer),
        command,
    )
    assert url, "README must pin the published installer revision"
    checked_download(url[0], hashlib.sha256((HERE / installer).read_bytes()).hexdigest())
    sources = json.loads((HERE / "sources.json").read_text())
    base = f"https://github.com/lanl/chatty/releases/download/{sources['tag']}"
    manifest_sha = (HERE / "manifest.sha256").read_text().split()[0]
    manifest = json.loads(checked_download(f"{base}/manifest.json", manifest_sha))
    assert manifest["tag"] == sources["tag"]
    assert manifest["python"] == (HERE / ".python-version").read_text().strip()
    for name, source in sources["projects"].items():
        project = manifest["projects"][name]
        assert project["revision"] == source["revision"] and project["version"] == source["version"]
    lock = manifest["platforms"][target]
    requirements = lock["requirements"]
    checked_download(requirements["url"], requirements["sha256"])

    with TemporaryDirectory() as tmp:
        cwd = Path(tmp).resolve()
        assert all(not (p / ".git").exists() for p in (cwd, *cwd.parents))
        env = install_environment(cwd)
        user_home = Path(env["USERPROFILE"] if WINDOWS else env["HOME"])
        root = (
            Path(env["LOCALAPPDATA"]) if WINDOWS else user_home / "Library/Application Support"
        ) / "ChattySuite"
        assert not root.exists(), f"Expected a clean test profile: {root}"
        print(command, flush=True)
        install(command, cwd, env)
        work = cwd / "research project"
        work.mkdir()

        active, preserved = exercise_upgrade(command, cwd, work, env, root, user_home, manifest_sha)
        python = str(active / ("Scripts/python.exe" if WINDOWS else "bin/python"))
        report = json.loads((active / "pip-report.json").read_text(encoding="utf-8"))
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
        subprocess.run([python, "-I", "-m", "pip", "check"], cwd=work, env=env, check=True)
        versions = {}
        for name, source in sources["projects"].items():
            assert command_path(name, work, env).parent.resolve() == active / "commands"
            result = fresh_shell(f"{name} --version", work, env)
            assert result.split()[-1] == source["version"], result
            assert "--help" in fresh_shell(f"{name} --help", work, env)
            versions[name] = result
        probe = subprocess.check_output(
            [python, "-I", str(HERE / "probe.py")], cwd=work, env=env, text=True
        )
        runtime = json.loads(probe.splitlines()[-1])
        assert Path(runtime["executable"]) == Path(python) and runtime["machine"] == arch
        assert runtime["python"] == manifest["python"]
        assert Path(runtime["base_prefix"]).resolve().is_relative_to(root.resolve() / "python")
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
            "commands": command.splitlines(),
            "cwd": str(work),
            "manifest_sha256": manifest_sha,
            "requirements_sha256": requirements["sha256"],
            "verified_wheels": len(installed),
            "pip_check": "passed",
            "cli": versions,
            "cli_help": "passed",
            "reinstall": "reused runtime",
            "upgrade": "replaced simulated predecessor",
            "preserved": preserved,
            "pip": report["pip_version"],
            "runtime": runtime,
        }
        print(json.dumps(result_data, indent=2), flush=True)
        result_data["packages"] = installed
        evidence.write_text(json.dumps(result_data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    verify(sys.argv[1], Path(sys.argv[2]).resolve())
