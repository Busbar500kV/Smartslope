#!/bin/sh
# Smartslope: Main pipeline orchestrator
# Runs synthetic generation followed by detection pipeline

set -eu

echo "=================================================="
echo "  Smartslope: Radar-Based Slope Monitoring"
echo "  Pipeline: Synthetic → Detection"
echo "=================================================="
echo ""

# Always run from repo root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Repository root: $REPO_ROOT"
echo ""

# --- Python environment setup ---
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  echo ""
fi

# shellcheck disable=SC1091
. .venv/bin/activate

echo "Python: $(python --version)"
echo ""

# Install dependencies (idempotent)
echo "Installing dependencies..."
python -m pip install -q -U pip
python -m pip install -q -e . || python -m pip install -q numpy matplotlib
echo "✓ Dependencies installed"
echo ""

# --- Stage 1: Generate synthetic data ---
echo "--------------------------------------------------"
echo "Stage 1: Synthetic Data Generation"
echo "--------------------------------------------------"
python -m smartslope.synthetic
echo ""

# --- Stage 2: Run detection pipeline ---
echo "--------------------------------------------------"
echo "Stage 2: Detection Pipeline"
echo "--------------------------------------------------"
python -m smartslope.detection
echo ""

# --- Done ---
echo "=================================================="
echo "Pipeline complete!"
echo "=================================================="
echo ""
echo "Outputs:"
echo "  - Synthetic data: data/synthetic/"
echo "  - Detection results: outputs/synthetic/"
echo ""
