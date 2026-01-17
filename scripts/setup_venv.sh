#!/bin/sh
# Setup script for Smartslope on Busbar
# Creates virtual environment and installs package in editable mode

set -e

echo "=== Smartslope Setup ==="

# Target venv location
VENV_DIR="$HOME/venvs/smartslope"

# Check Python version
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found. Please install Python 3.10+."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python: $PYTHON_VERSION"

# Create venv if it doesn't exist
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at: $VENV_DIR"
else
    echo "Creating virtual environment at: $VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
fi

# Activate venv
echo "Activating virtual environment..."
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install package in editable mode
echo "Installing smartslope package (editable mode)..."
python -m pip install -e . --quiet

echo ""
echo "=== Setup Complete ==="
echo "Virtual environment: $VENV_DIR"
echo ""
echo "To activate manually:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  ./scripts/run_pipeline.sh"
echo ""
