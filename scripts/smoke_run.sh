#!/bin/bash
set -euo pipefail

# Smoke test for Smartslope
# Creates venv, installs package, runs minimal end-to-end pipeline

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "========================================="
echo "  Smartslope Smoke Run"
echo "========================================="
echo "Repo root: $REPO_ROOT"
echo ""

# Step 1: Setup virtual environment and install package
echo "Step 1: Setting up virtual environment..."
bash "$REPO_ROOT/scripts/setup_venv.sh"

# Step 2: Activate venv for subsequent commands
echo ""
echo "Step 2: Activating virtual environment..."
VENV_DIR="$REPO_ROOT/.venv"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Step 3: Run the pipeline
echo ""
echo "Step 3: Running pipeline..."
bash "$REPO_ROOT/scripts/run_pipeline.sh"

echo ""
echo "========================================="
echo "  Smoke run complete!"
echo "========================================="
echo ""
echo "Check outputs/ directory for results."
