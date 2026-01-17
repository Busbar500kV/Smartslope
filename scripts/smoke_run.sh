#!/bin/sh
# Smoke test: Full setup and run of Smartslope pipeline
# Exits with nonzero status on any failure

set -e

echo "=== Smartslope Smoke Test ==="
echo ""

# Get repo root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Repository: $REPO_ROOT"
echo ""

# Step 1: Setup
echo "[1/2] Running setup..."
./scripts/setup_venv.sh

echo ""
echo "[2/2] Running pipeline..."
./scripts/run_pipeline.sh

echo ""
echo "=== Smoke Test PASSED ==="
echo ""

# Verify outputs exist
if [ -d "data" ] && [ -d "outputs" ]; then
    echo "Verification:"
    echo "  data/ contains $(find data -type f | wc -l) file(s)"
    echo "  outputs/ contains $(find outputs -type f | wc -l) file(s)"
    echo ""
else
    echo "Warning: Expected directories not found."
    exit 1
fi

echo "Smoke test completed successfully."
exit 0
