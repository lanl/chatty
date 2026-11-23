"""Promote the exact prerelease tested by the native jobs in this workflow run."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent


def promote(evidence_dir: Path) -> None:
    sources = json.loads((HERE / "sources.json").read_bytes())
    repo = os.environ["GITHUB_REPOSITORY"]
    revision = os.environ["GITHUB_SHA"]
    run = f"https://github.com/{repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
    tag = sources["tag"]
    release = ["gh", "release"]
    metadata = json.loads(
        subprocess.check_output(
            [*release, "view", tag, "-R", repo, "--json", "isDraft,isPrerelease,body,assets"]
        )
    )
    if metadata["isDraft"] or not metadata["isPrerelease"]:
        raise ValueError(f"Expected a published prerelease for {tag}")

    # Check the actual public asset identities again immediately before promotion.
    public = {asset["name"]: asset["digest"] for asset in metadata["assets"]}
    base = f"https://github.com/{repo}/releases/download/{tag}"
    with urlopen(f"{base}/manifest.json", timeout=60) as response:
        raw = response.read()
    manifest_sha = (HERE / "manifest.sha256").read_text().split()[0]
    manifest = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() != manifest_sha or manifest["tag"] != tag:
        raise ValueError(f"Manifest mismatch for {tag}")
    locked = {"manifest.json": manifest_sha}
    evidence = []
    for name, expected in sources["projects"].items():
        source = manifest["projects"][name]
        if any(source[key] != value for key, value in expected.items()):
            raise ValueError(f"Unexpected {name} source in {tag}")
        locked[source["url"].rsplit("/", 1)[1]] = source["sha256"]
    for target in ("macos-arm64", "windows-x64"):
        platform = manifest["platforms"][target]
        req = platform["requirements"]
        locked[req["url"].rsplit("/", 1)[1]] = req["sha256"]
        path = evidence_dir / f"{target}.evidence.json"
        record = json.loads(path.read_bytes())
        expected_record = {
            "run": run,
            "workflow_revision": revision,
            "platform": target,
            "manifest_sha256": manifest_sha,
            "packages": platform["packages"],
            "requirements_sha256": req["sha256"],
        }
        for key, value in expected_record.items():
            if record[key] != value:
                raise ValueError(f"{target} evidence has the wrong {key}")
        upgrade = record["upgrade"]
        if (
            upgrade["from_manifest_sha256"] != sources["upgrade_from"]["manifest_sha256"]
            or upgrade["from"] != sources["upgrade_from"]["tag"]
            or upgrade["retrieved_before_and_after"] is not True
            or upgrade["ivfpq_retrieval"] != "passed"
        ):
            raise ValueError(
                f"Missing verified upgrade from {sources['upgrade_from']['tag']} on {target}"
            )
        evidence.append(path)
    for filename, digest in locked.items():
        if public.get(filename) != f"sha256:{digest}":
            raise ValueError(f"Public release asset differs from the tested manifest: {filename}")

    subprocess.run(
        [*release, "upload", tag, *map(str, evidence), "--clobber", "-R", repo], check=True
    )
    assets = json.loads(
        subprocess.check_output([*release, "view", tag, "-R", repo, "--json", "assets"])
    )
    attached = {asset["name"]: asset["digest"] for asset in assets["assets"]}
    for path in evidence:
        locked[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if attached != {name: f"sha256:{sha}" for name, sha in locked.items()}:
        raise ValueError(
            "Release assets do not match the verified wheels, locks, manifest and evidence"
        )
    with TemporaryDirectory() as temp:
        notes = Path(temp) / "notes.md"
        notes.write_text(
            metadata["body"].rstrip()
            + "\n\n"
            + f"[Verified installation and upgrade on macOS arm64 and Windows x64]({run}) · "
            + " · ".join(f"[{p.name.split('.')[0]} evidence]({base}/{p.name})" for p in evidence)
            + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [
                *release,
                "edit",
                tag,
                "-R",
                repo,
                "--prerelease=false",
                "--latest",
                "--notes-file",
                str(notes),
            ],
            check=True,
        )


if __name__ == "__main__":
    promote(Path(sys.argv[1]))
