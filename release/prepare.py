"""Build the suite wheels, resolve each native environment, and assemble release assets."""

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
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
SOURCES = json.loads((HERE / "sources.json").read_text())
PYTHON = (HERE / ".python-version").read_text().strip()
BASE = f"https://github.com/lanl/chatty/releases/download/{SOURCES['tag']}"
PLATFORMS = {("Darwin", "arm64"): "macos-arm64", ("Windows", "AMD64"): "windows-x64"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def wheel_path(directory: Path, project: str) -> Path:
    version = SOURCES["projects"][project]["version"]
    return directory / f"{project.replace('-', '_')}-{version}-py3-none-any.whl"


def build(out: Path) -> None:
    for project, source in SOURCES["projects"].items():
        with TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "source"
            built = Path(tmp) / "wheels"
            repo = f"https://github.com/lanl/{project}.git"
            subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            subprocess.run(
                ["git", "fetch", "--depth=1", repo, source["revision"]], cwd=checkout, check=True
            )
            subprocess.run(
                ["git", "-c", "core.autocrlf=false", "checkout", "--detach", "FETCH_HEAD"],
                cwd=checkout,
                check=True,
            )
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
            ).strip()
            if revision != source["revision"]:
                raise ValueError(f"Unexpected {project} revision {revision}")
            epoch = subprocess.check_output(
                ["git", "show", "-s", "--format=%ct"], cwd=checkout, text=True
            ).strip()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-build-isolation",
                    "--no-deps",
                    "--no-index",
                    "-w",
                    str(built),
                    ".",
                ],
                cwd=checkout,
                env={**os.environ, "SOURCE_DATE_EPOCH": epoch, "PYTHONHASHSEED": "0", "TZ": "UTC"},
                check=True,
            )
            wheel = wheel_path(built, project)
            if list(built.glob("*.whl")) != [wheel]:
                raise ValueError(f"{project}@{revision} did not produce {wheel.name}")
            destination = out / wheel.name
            shutil.copyfile(wheel, destination)
            print(project, revision, digest(destination), flush=True)


def reproduce(out: Path) -> None:
    manifest_path = out / "manifest.json"
    if digest(manifest_path) != (HERE / "manifest.sha256").read_text().split()[0]:
        raise ValueError("Downloaded manifest differs from the reviewed digest")
    manifest = json.loads(manifest_path.read_bytes())
    build(out)
    for name, source in SOURCES["projects"].items():
        published = manifest["projects"][name]
        if (
            source["revision"] != published["revision"]
            or digest(wheel_path(out, name)) != published["sha256"]
        ):
            raise ValueError(f"Rebuild differs from published {name} wheel")


def resolve(out: Path) -> None:
    target = PLATFORMS[platform.system(), platform.machine()]
    projects = SOURCES["projects"]
    # Enable Chatty's FAISS extra. The explicit LANL wheel below supplies LitKit.
    roots = [
        f"{p}{'[rag]' if p == 'chatty' else ''} @ {wheel_path(out, p).as_uri()}" for p in projects
    ]
    with TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "--isolated",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--only-binary=:all:",
                "--index-url",
                "https://pypi.org/simple",
                "-c",
                str(HERE / "constraints.txt"),
                "--report",
                str(report),
                *roots,
            ],
            check=True,
        )
        installs = json.loads(report.read_text(encoding="utf-8"))["install"]
    packages = {}
    for item in installs:
        metadata, download = item["metadata"], item["download_info"]
        name = re.sub(r"[-_.]+", "-", metadata["name"]).lower()
        url = download["url"]
        sha = download["archive_info"]["hashes"]["sha256"]
        if name in projects:
            wheel = wheel_path(out, name)
            if (
                url != wheel.as_uri()
                or sha != digest(wheel)
                or metadata["version"] != projects[name]["version"]
            ):
                raise ValueError(f"Unexpected release wheel for {name}")
            url = f"{BASE}/{wheel.name}"
        elif urlsplit(url).hostname != "files.pythonhosted.org" or not unquote(
            urlsplit(url).path
        ).endswith(".whl"):
            raise ValueError(f"Expected a PyPI wheel for {name}, got {url}")
        packages[name] = {"version": metadata["version"], "url": url, "sha256": sha}
    write_json(out / f"{target}.json", dict(sorted(packages.items())))


def assemble(out: Path) -> None:
    manifest = {"tag": SOURCES["tag"], "python": PYTHON, "projects": {}, "platforms": {}}
    for name, source in SOURCES["projects"].items():
        wheel = wheel_path(out, name)
        manifest["projects"][name] = {
            **source,
            "repository": f"https://github.com/lanl/{name}",
            "url": f"{BASE}/{wheel.name}",
            "sha256": digest(wheel),
        }
    for target in PLATFORMS.values():
        packages = json.loads((out / f"{target}.json").read_text())
        for name, source in manifest["projects"].items():
            if any(source[key] != packages[name][key] for key in ("url", "sha256", "version")):
                raise ValueError(f"{target} resolved a different {name} wheel")
        lock = out / f"{target}.txt"
        lock.write_text(
            "".join(
                f"{name}{'[rag]' if name == 'chatty' else ''} @ {row['url']} --hash=sha256:{row['sha256']}\n"
                for name, row in sorted(packages.items())
            ),
            encoding="utf-8",
            newline="\n",
        )
        manifest["platforms"][target] = {
            "requirements": {"url": f"{BASE}/{lock.name}", "sha256": digest(lock)},
            "packages": packages,
        }
    write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    commands = {"build": build, "resolve": resolve, "assemble": assemble, "reproduce": reproduce}
    if len(sys.argv) != 3 or sys.argv[1] not in commands:
        sys.exit("Usage: prepare.py {build|resolve|assemble|reproduce} ASSET_DIRECTORY")
    if platform.python_version() != PYTHON:
        sys.exit(f"Use Python {PYTHON}")
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commands[sys.argv[1]](output)
