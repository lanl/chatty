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
            "grep",
        ]:
            path = shutil.which(name, path="/usr/bin:/bin:/usr/sbin")
            assert path, name
            (tools / name).symlink_to(path)
        env = {"HOME": str(home), "SHELL": "/bin/zsh", "PATH": str(tools), "TMPDIR": str(tmp)}
    env = {
        k: v
        for k, v in env.items()
        if not k.startswith(("OPENAI_", "CHATTY_", "HF_", "LITKIT_"))
        and k not in {"API_KEY", "BASE_URL", "RAG_WORKSPACE"}
    }
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


def check_mac_destination(root: Path, cwd: Path, env: dict[str, str]) -> None:
    # A pre-existing file or directory must survive a failed activation.
    root.mkdir(parents=True)
    dest = root / "bin"
    dest.mkdir()
    for entry in (dest / "next", dest):
        entry.write_text("user file")
        run = subprocess.run(
            ["/bin/sh", str(HERE / "install.sh")], cwd=cwd, env=env, capture_output=True, text=True
        )
        assert run.returncode and "already exists" in run.stderr, run
        assert not entry.is_symlink() and entry.read_text() == "user file"
        assert not (root / "runtimes").exists()
        entry.unlink()
        if dest.is_dir():
            dest.rmdir()


def check_mac_shells(root: Path, work: Path, env: dict[str, str]) -> None:
    for shell in ("/bin/bash", "/bin/zsh"):
        for mode in ("-ic", "-lic"):
            command = 'printf "%s\\n" "$PATH"; command -v chatty text-fetch litkit'
            result = subprocess.check_output([shell, mode, command], cwd=work, env=env, text=True)
            path, *commands = result.strip().splitlines()
            assert path.split(os.pathsep).count(str(root / "bin")) == 1, (shell, mode, path)
            assert commands == [
                str(root / "bin" / name) for name in ("chatty", "text-fetch", "litkit")
            ], result


def installer_command(readme: str, target: str) -> str:
    block = readme.split(f"<!-- install-{target} -->\n", 1)[1].split("\n```", 1)[0]
    return block.split("\n", 1)[1]


def install_baseline(
    baseline: dict[str, str], target: str, cwd: Path, env: dict[str, str]
) -> dict[str, str]:
    """Run the prior release's published instructions and return its application versions."""
    with urlopen(
        f"https://raw.githubusercontent.com/lanl/chatty/{baseline['instructions_revision']}/README.md",
        timeout=60,
    ) as response:
        previous_command = installer_command(response.read().decode("utf-8"), target)
    previous_manifest = json.loads(
        checked_download(
            f"https://github.com/lanl/chatty/releases/download/{baseline['tag']}/manifest.json",
            baseline["manifest_sha256"],
        )
    )
    assert previous_manifest["tag"] == baseline["tag"]
    install(previous_command, cwd, env)
    return {name: project["version"] for name, project in previous_manifest["projects"].items()}


def fixture_probe(runtime: Path, mode: str, work: Path, env: dict[str, str]) -> str:
    python = runtime / ("Scripts/python.exe" if WINDOWS else "bin/python")
    return subprocess.check_output(
        [str(python), "-I", str(HERE / "probe_data.py"), mode], cwd=work, env=env, encoding="utf-8"
    ).strip()


def build_prior_fixture(
    user_home: Path, quickstart: Path, work: Path, original: Path, offline: dict[str, str]
) -> tuple[str, str, dict[Path, str]]:
    config = user_home / ".config/chatty/config.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        'session_path = "./saved sessions"\nrag_workspace = "./workspace"\nrag_provider = "litkit"\n'
    )
    corpus = work / "corpus/rain-gauge.xml"
    corpus.parent.mkdir()
    shutil.copyfile(quickstart / "example/articles/rain-gauge.xml", corpus)
    fresh_shell(
        "text-fetch tarball create --xml-dir corpus --out shards/rain.tar.gz", work, offline
    )
    workspace = work / "workspace"
    fresh_shell(
        "litkit --build-only --faiss-writer --tar-dir shards --papers-index flat --chunks-index flat",
        work,
        {**offline, "LITKIT_WORKSPACE": str(workspace)},
    )

    title, passage = fixture_probe(original, "create", work, offline).splitlines()
    (session_file,) = (work / "saved sessions").glob("*.json")
    paths = (
        corpus,
        work / "shards/rain.tar.gz",
        workspace / "indices/papers.faiss",
        workspace / "indices/chunks.faiss",
        workspace / "sqlite/litkit.sqlite3",
        config,
        session_file,
    )
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    return title, passage, digests


def exercise_upgrade(
    command: str,
    cwd: Path,
    work: Path,
    env: dict[str, str],
    root: Path,
    user_home: Path,
    manifest_sha: str,
    baseline: dict[str, str],
    target: str,
) -> tuple[Path, dict[str, object]]:
    def active_runtime() -> Path:
        runtime = command_path("chatty", work, env).parent.resolve().parent
        assert runtime.parent == root.resolve() / "runtimes", runtime
        return runtime

    python_path = Path("Scripts/python.exe" if WINDOWS else "bin/python")
    path_command = "$env:Path" if WINDOWS else 'print -r -- "$PATH"'
    initial = active_runtime()
    assert (initial / "manifest.sha256").read_text().strip() == manifest_sha
    for name in ("chatty", "text-fetch", "litkit"):
        assert command_path(name, work, env).parent.resolve() == initial / "commands"

    # Run the actual guide in the clean install, including its first model downloads.
    quickstart = cwd / "研究 field notes"
    fresh_env = {**env, "PATH": fresh_shell(path_command, work, env)}
    subprocess.run(
        [
            str(initial / python_path),
            "-I",
            str(HERE.parent / "ci/check_quickstart.py"),
            str(quickstart),
        ],
        cwd=work,
        env=fresh_env,
        check=True,
    )
    # Old Chatty has no model setup command. Reuse only the real models, and
    # build an independent corpus/index with the prior release's installed CLIs.
    previous_versions = install_baseline(baseline, target, cwd, env)
    original = active_runtime()
    assert (
        original != initial
        and (original / "manifest.sha256").read_text().strip() == baseline["manifest_sha256"]
    )
    for name, version in previous_versions.items():
        assert command_path(name, work, env).parent.resolve() == original / "commands"
        assert fresh_shell(f"{name} --version", work, env).split()[-1] == version

    offline = {
        **env,
        "LITKIT_FORCE_DEVICE": "cpu",
        "HF_HOME": str(quickstart / "hf_cache"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    title, passage, digests = build_prior_fixture(user_home, quickstart, work, original, offline)

    def check_preserved() -> None:
        for p, digest in digests.items():
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, f"Installer changed {p}"

    # A bad manifest pin must stop the real installer with the working suite still selected.
    script = "install.ps1" if WINDOWS else "install.sh"
    source = (HERE / script).read_text(encoding="utf-8")
    assert source.count(manifest_sha) == 1
    damaged = cwd / script
    damaged.write_text(source.replace(manifest_sha, "0" * 64), encoding="utf-8")
    shell = (
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
        if WINDOWS
        else ["/bin/sh"]
    )
    rejected = subprocess.run(
        [*shell, str(damaged)], cwd=cwd, env=env, capture_output=True, text=True
    )
    mismatch = "SHA-256 mismatch" if WINDOWS else "manifest.json: FAILED"
    assert rejected.returncode != 0 and mismatch in rejected.stdout + rejected.stderr, rejected
    assert active_runtime() == original

    install(command, cwd, env)
    active = active_runtime()
    assert active != original and (active / "manifest.sha256").read_text().strip() == manifest_sha
    path = fresh_shell(path_command, work, env)
    suite_paths = [
        Path(p).resolve() for p in path.split(os.pathsep) if Path(p).is_relative_to(root)
    ]
    assert suite_paths == [active / "commands"], suite_paths
    check_preserved()
    output = fresh_shell(
        'chatty retrieve "How much rain did the copper gauge collect?"', work, offline
    )
    assert re.search(re.escape(title) + r" \[chunk \d+\]\s+" + re.escape(passage), output), output
    print(output, flush=True)
    fixture_probe(active, "read", work, offline)
    print(fixture_probe(active, "ivfpq", work, offline), flush=True)

    install(command, cwd, env)
    assert active_runtime() == active, "Reinstall duplicated the runtime"
    check_preserved()
    if not WINDOWS:
        check_mac_shells(root, work, env)

    # Damage a native dependency without changing package metadata or the release marker.
    # A reinstall must repair it even though pip check and the CLI versions still pass.
    site = active / ("Lib" if WINDOWS else "lib/python3.12") / "site-packages/faiss"
    extensions = list(site.glob("_swigfaiss*.pyd" if WINDOWS else "_swigfaiss*.so"))
    assert extensions, site
    for extension in extensions:
        extension.unlink()
    install(command, cwd, env)
    repaired = active_runtime()
    assert repaired != active
    check_preserved()
    fixture_probe(repaired, "read", work, offline)
    return repaired, {
        "from": baseline["tag"],
        "from_manifest_sha256": baseline["manifest_sha256"],
        "from_chatty": previous_versions["chatty"],
        "title": title,
        "passage": passage,
        "retrieved_before_and_after": True,
        "ivfpq_retrieval": "passed",
        "preserved_sha256": {p.name: digest for p, digest in digests.items()},
    }


def verify(target: str, evidence: Path) -> None:
    os_name, arch = {"macos-arm64": ("Darwin", "arm64"), "windows-x64": ("Windows", "AMD64")}[
        target
    ]
    assert (platform.system(), platform.machine()) == (os_name, arch)
    readme = (HERE.parent / "README.md").read_text(encoding="utf-8")
    command = installer_command(readme, target)
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
        if not WINDOWS:
            check_mac_destination(root, cwd, env)
            # A Bash login profile commonly sources .bashrc itself.
            (user_home / ".bash_login").write_text('. "$HOME/.bashrc"\n')
            (user_home / ".zshenv").write_text('ZDOTDIR="$HOME/.config/zsh settings"\n')
        print(command, flush=True)
        install(command, cwd, env)
        work = cwd / "research project"
        work.mkdir()

        active, upgrade = exercise_upgrade(
            command, cwd, work, env, root, user_home, manifest_sha, sources["upgrade_from"], target
        )
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
        context = HERE / "runner.json"
        runner = json.loads(context.read_bytes()) if context.exists() else os.environ
        run_id = runner.get("GITHUB_RUN_ID")
        result_data = {
            "run": f"https://github.com/{runner['GITHUB_REPOSITORY']}/actions/runs/{run_id}"
            if run_id
            else None,
            "attempt": runner.get("GITHUB_RUN_ATTEMPT"),
            "workflow_revision": runner.get("GITHUB_SHA"),
            "runner_image": runner.get("ImageVersion", platform.platform()),
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
            "repair": "replaced runtime with missing FAISS extensions",
            "invalid_manifest": "rejected, prior runtime still active",
            "quickstart": "published install, cold model cache, native JATS, CPU retrieval",
            "upgrade": upgrade,
            "pip": report["pip_version"],
            "runtime": runtime,
        }
        print(json.dumps(result_data, indent=2), flush=True)
        result_data["packages"] = installed
        evidence.write_text(json.dumps(result_data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    verify(sys.argv[1], Path(sys.argv[2]).resolve())
