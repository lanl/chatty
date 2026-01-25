#!/bin/bash
# Build wheelhouse for offline install of chatty
#
# Usage:
#   ./scripts/build-wheelhouse.sh [wheelhouse_dir]
#
# This script downloads all dependencies as wheel files for transfer
# to air-gapped systems. Run on a connected machine.
#
# Prerequisites:
#   - uv (for generating requirements.lock)
#   - pip (for downloading wheels — uv does not yet have a download command)
#
# Example:
#   ./scripts/build-wheelhouse.sh
#   # Creates wheelhouse/ directory
#   # Transfer wheelhouse/ to air-gapped machine (requirements.lock is in scripts/)
#
# See: docs/DEVELOPER_GUIDE.md for full offline install instructions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WHEELHOUSE="${1:-${PROJECT_ROOT}/wheelhouse}"
REQUIREMENTS="${SCRIPT_DIR}/requirements.lock"

# --- Prerequisite checks ---

# Check for pip (required for downloading wheels)
if ! command -v pip &>/dev/null; then
    echo "Error: pip is required for building the wheelhouse."
    echo ""
    echo "uv does not yet support 'pip download', so this script uses pip directly."
    echo ""
    echo "Install pip:"
    echo "  - macOS/Linux: https://pip.pypa.io/en/stable/installation/"
    echo "  - Or use system Python: python3 -m ensurepip"
    exit 1
fi

# Check for uv (required for generating requirements.lock)
if ! command -v uv &>/dev/null; then
    echo "Error: uv is required for generating requirements.lock."
    echo ""
    echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# --- Generate requirements.lock if not present ---
if [[ ! -f "$REQUIREMENTS" ]]; then
    echo "Generating requirements.lock..."
    # Export without editable, filter out local package reference
    uv export --frozen --no-dev --no-editable | grep -v "^\.$" > "$REQUIREMENTS"
fi

# --- Build wheelhouse ---
echo "Building wheelhouse for offline install..."
echo "  Source: $REQUIREMENTS"
echo "  Target: $WHEELHOUSE/"

# Create wheelhouse directory
mkdir -p "$WHEELHOUSE"

# Download all wheels using pip
# Note: Downloads wheels for the current platform
pip download -r "$REQUIREMENTS" -d "$WHEELHOUSE/"

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
echo "  1. Transfer wheelhouse/ to target (requirements.lock is in scripts/)"
echo "  2. Run: ./scripts/install-offline.sh"
