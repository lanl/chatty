#!/bin/bash
# Install chatty from local wheelhouse (no network required)
#
# Usage:
#   ./scripts/install-offline.sh [wheelhouse_dir]
#
# This script installs chatty and all dependencies from a pre-built
# wheelhouse directory. Run on air-gapped machines.
#
# Prerequisites:
#   - uv installed on target machine
#   - wheelhouse/ directory transferred from connected machine
#   - scripts/requirements.lock exists (part of repo)
#
# See: docs/DEVELOPER_GUIDE.md for full offline install instructions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WHEELHOUSE="${1:-${PROJECT_ROOT}/wheelhouse}"
REQUIREMENTS="${SCRIPT_DIR}/requirements.lock"

# Check for wheelhouse directory
if [[ ! -d "$WHEELHOUSE" ]]; then
    echo "Error: Wheelhouse directory not found: $WHEELHOUSE"
    echo ""
    echo "Transfer the wheelhouse/ directory from a connected machine."
    echo "On connected machine, run: ./scripts/build-wheelhouse.sh"
    exit 1
fi

# Check for requirements.lock
if [[ ! -f "$REQUIREMENTS" ]]; then
    echo "Error: requirements.lock not found"
    echo ""
    echo "Transfer requirements.lock from the connected machine."
    exit 1
fi

# Count wheels
WHEEL_COUNT=$(find "$WHEELHOUSE" -name "*.whl" | wc -l | tr -d ' ')
if [[ "$WHEEL_COUNT" -eq 0 ]]; then
    echo "Error: No wheel files found in $WHEELHOUSE/"
    exit 1
fi

echo "Installing chatty from local wheelhouse..."
echo "  Wheelhouse: $WHEELHOUSE/ ($WHEEL_COUNT wheels)"
echo "  Requirements: $REQUIREMENTS"

# Create virtual environment if it doesn't exist
if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
    echo ""
    echo "Creating virtual environment..."
    uv venv "${PROJECT_ROOT}/.venv"
fi

# Install from wheelhouse (no network)
echo ""
echo "Installing packages (offline)..."
uv pip install --no-index --find-links="$WHEELHOUSE/" -r "$REQUIREMENTS"

echo ""
echo "Installation complete!"
echo ""
echo "To use chatty:"
echo "  source .venv/bin/activate"
echo "  chatty --help"
echo ""
echo "Or run directly:"
echo "  uv run chatty --help"