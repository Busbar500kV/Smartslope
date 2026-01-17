#!/bin/bash
set -euo pipefail

# Setup virtual environment for Smartslope
# Creates venv at .venv inside repo root

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="$REPO_ROOT/.venv"

echo "=== Smartslope: setup_venv.sh ==="
echo "Repo root: $REPO_ROOT"
echo "Venv location: $VENV_DIR"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
else
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Python: $(python --version)"
echo "Pip: $(pip --version)"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing smartslope package in editable mode..."
pip install -e "$REPO_ROOT"

echo ""
echo "=== Setup complete ==="
echo "To activate the virtual environment manually, run:"
echo "  source $VENV_DIR/bin/activate"
