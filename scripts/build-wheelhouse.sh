#!/bin/bash
# Build a platform-specific bundle of base Chatty and its locked dependencies.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname "$script_dir")"
wheelhouse="${1:-$project_root/wheelhouse}"
mkdir -p "$wheelhouse"
# Publish one pointer after the complete generation is ready. An installer
# already running can finish against the generation it opened.
stage="$(mktemp -d "$wheelhouse/bundle.XXXXXXXX")"
pointer="$(mktemp "$wheelhouse/.current.XXXXXXXX")"
trap 'if [[ -f "$pointer" ]]; then rm -rf "$stage"; rm -f "$pointer"; fi' EXIT

python="$(uv python find "${UV_PYTHON:-3.12}")"
requirements="$stage/requirements.lock"
uv export --project "$project_root" --locked --no-dev --no-emit-project \
    --format requirements-txt --output-file "$requirements" > /dev/null
uv tool run --python "$python" --from pip pip download --only-binary=:all: \
    --require-hashes --no-deps -r "$requirements" -d "$stage"
uv build "$project_root" --wheel --out-dir "$stage"

# Pin the local build too, so rebuilding into an existing wheelhouse selects
# this exact wheel even if it also contains older versions.
"$python" - "$stage" <<'PY'
import hashlib
import sys
from pathlib import Path

stage = Path(sys.argv[1])
wheel, = stage.glob("chatty-*.whl")
version = wheel.name.split("-")[1]
with wheel.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
with (stage / "requirements.lock").open("a") as lock:
    lock.write(f"\nchatty=={version} --hash=sha256:{digest}\n")
(stage / ".python-version").write_text(f"{sys.version_info.major}.{sys.version_info.minor}\n")
PY

basename "$stage" > "$pointer"
mv -f "$pointer" "$wheelhouse/current"
echo "Transfer $wheelhouse/ and scripts/install-offline.sh to the target."
