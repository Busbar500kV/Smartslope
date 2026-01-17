#!/bin/bash
set -euo pipefail

# Run Smartslope pipeline
# Assumes venv is already set up and activated

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Smartslope: run_pipeline.sh ==="
echo "Repo root: $REPO_ROOT"

# Generate unique run ID based on timestamp
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
echo "Run ID: $RUN_ID"

# Ensure data and outputs directories exist
mkdir -p data/synthetic
mkdir -p outputs/"$RUN_ID"

echo ""
echo "=== Running Smartslope pipeline ==="
echo "This will:"
echo "  1. Generate synthetic coherent phase data"
echo "  2. Run baseline detection"
echo "  3. Write outputs under outputs/$RUN_ID/"
echo ""

# Run the pipeline using the smartslope CLI
smartslope pipeline

# Move outputs to run-specific directory
if [ -d "outputs/synthetic" ]; then
    echo ""
    echo "Moving outputs to outputs/$RUN_ID/..."
    mv outputs/synthetic/* "outputs/$RUN_ID/" 2>/dev/null || true
    rmdir outputs/synthetic 2>/dev/null || true
fi

echo ""
echo "=== Pipeline complete ==="
echo "Outputs written to: outputs/$RUN_ID/"
echo ""
echo "Output files:"
ls -lh "outputs/$RUN_ID/" 2>/dev/null || echo "(no files generated)"
