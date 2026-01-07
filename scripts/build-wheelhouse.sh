#!/bin/bash
# Build wheelhouse for offline install of chatty
#
# Usage:
#   ./scripts/build-wheelhouse.sh [wheelhouse_dir]
#
# This script downloads all dependencies as wheel files for transfer
# to air-gapped systems. Run on a connected machine.
#
# Example:
#   ./scripts/build-wheelhouse.sh
#   # Creates wheelhouse/ directory
#   # Transfer wheelhouse/ + requirements.lock to air-gapped machine
#
# See: docs/DEVELOPER_GUIDE.md for full offline install instructions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WHEELHOUSE="${1:-${PROJECT_ROOT}/wheelhouse}"
REQUIREMENTS="${PROJECT_ROOT}/requirements.lock"

# Generate requirements.lock if not present
if [[ ! -f "$REQUIREMENTS" ]]; then
    echo "Generating requirements.lock..."
    # Export without editable, filter out local package reference
    uv export --frozen --no-dev --no-editable | grep -v "^\.$" > "$REQUIREMENTS"
fi

echo "Building wheelhouse for offline install..."
echo "  Source: $REQUIREMENTS"
echo "  Target: $WHEELHOUSE/"

# Create wheelhouse directory
mkdir -p "$WHEELHOUSE"

# Ensure pip is available (uv venvs don't include pip by default)
if ! python -m pip --version &>/dev/null; then
    echo "Installing pip in virtual environment..."
    uv pip install pip
fi

# Download all wheels
# Note: Downloads wheels for the current platform
python -m pip download -r "$REQUIREMENTS" -d "$WHEELHOUSE/"

# Report results
WHEEL_COUNT=$(find "$WHEELHOUSE" -name "*.whl" | wc -l | tr -d ' ')
TARBALL_COUNT=$(find "$WHEELHOUSE" -name "*.tar.gz" -o -name "*.zip" | wc -l | tr -d ' ')
SIZE=$(du -sh "$WHEELHOUSE" | cut -f1)

echo ""
echo "Wheelhouse created successfully!"
echo "  Location: $WHEELHOUSE/"
echo "  Size: $SIZE"
echo "  Wheels: $WHEEL_COUNT"
if [[ "$TARBALL_COUNT" -gt 0 ]]; then
    echo "  Source archives: $TARBALL_COUNT (may need build tools on target)"
fi

echo ""
echo "To install on air-gapped machine:"
echo "  1. Transfer wheelhouse/ and requirements.lock to target"
echo "  2. Run: ./scripts/install-offline.sh"