#!/bin/bash
set -euo pipefail

# Smoke test for Smartslope
# Creates venv, installs package, runs minimal end-to-end pipeline

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Parse arguments
PUBLISH_FLAGS=()
USE_3D=false
for arg in "$@"; do
    case $arg in
        --publish|--force)
            PUBLISH_FLAGS+=("$arg")
            ;;
        --3d)
            USE_3D=true
            ;;
        *)
            ;;
    esac
done

echo "========================================="
echo "  Smartslope Smoke Run"
echo "========================================="
echo "Repo root: $REPO_ROOT"
if [ "$USE_3D" = true ]; then
    echo "Mode: 3D Installation Model"
fi
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
if [ "$USE_3D" = true ]; then
    if [ ${#PUBLISH_FLAGS[@]} -gt 0 ]; then
        bash "$REPO_ROOT/scripts/run_pipeline.sh" --3d "${PUBLISH_FLAGS[@]}"
    else
        bash "$REPO_ROOT/scripts/run_pipeline.sh" --3d
    fi
else
    if [ ${#PUBLISH_FLAGS[@]} -gt 0 ]; then
        bash "$REPO_ROOT/scripts/run_pipeline.sh" "${PUBLISH_FLAGS[@]}"
    else
        bash "$REPO_ROOT/scripts/run_pipeline.sh"
    fi
fi

echo ""
echo "========================================="
echo "  Smoke run complete!"
echo "========================================="
echo ""
echo "Check outputs/ directory for results."
