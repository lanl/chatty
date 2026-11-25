#!/bin/bash
# Install base Chatty from a transferred wheelhouse using an installed Python.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname "$script_dir")"
wheelhouse="${1:-$project_root/wheelhouse}"
venv="$project_root/.venv"

if [[ ! -s "$wheelhouse/current" ]]; then
    echo "Incomplete wheelhouse at $wheelhouse. Rebuild with scripts/build-wheelhouse.sh." >&2
    exit 1
fi
bundle="$wheelhouse/$(cat "$wheelhouse/current")"
required_python="$(cat "$bundle/.python-version")"

if [[ -d "$venv" ]]; then
    installed_python="$("$venv/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    if [[ "$installed_python" != "$required_python" ]]; then
        echo "$venv uses Python $installed_python. Move it aside to install the Python $required_python bundle." >&2
        exit 1
    fi
else
    uv venv --offline --python "$required_python" "$venv"
fi
uv pip install --offline --python "$venv/bin/python" --no-index --reinstall-package chatty \
    --find-links "$bundle" --require-hashes -r "$bundle/requirements.lock"

echo "Installation complete. Run $venv/bin/chatty --help"
