#!/bin/bash
# Exercise the documented transfer and install outside the source tree.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="$(mktemp -d)"
trap 'rm -rf "$target"' EXIT
mkdir -p "$target/scripts"
cp "$project_root/scripts/install-offline.sh" "$target/scripts/"
cd "$target"
"$project_root/scripts/build-wheelhouse.sh" "$target/wheelhouse"
UV_OFFLINE=1 UV_CACHE_DIR="$target/empty-cache" scripts/install-offline.sh
.venv/bin/chatty --help
uv pip check --offline --python .venv/bin/python

# A checkout can produce a new build with the same version. The installed
# entry point must come from the transferred wheel even on a repeated install.
echo 'exit 1' > .venv/bin/chatty
UV_OFFLINE=1 UV_CACHE_DIR="$target/empty-cache" scripts/install-offline.sh
.venv/bin/chatty --help

# Refuse to reuse a virtual environment for a different Python minor version.
bundle="wheelhouse/$(cat wheelhouse/current)"
echo "3.99" > "$bundle/.python-version"
if scripts/install-offline.sh > mismatch.log 2>&1; then
    echo "Expected an incompatible virtual environment to be rejected" >&2
    exit 1
fi
grep -F "Python 3.99" mismatch.log
