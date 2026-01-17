#!/bin/bash
set -euo pipefail

# Run Smartslope pipeline
# Assumes venv is already set up and activated

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Parse arguments
PUBLISH=false
FORCE=false
for arg in "$@"; do
    case $arg in
        --publish)
            PUBLISH=true
            ;;
        --force)
            FORCE=true
            ;;
        *)
            ;;
    esac
done

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

# Publish artifacts if requested
if [ "$PUBLISH" = true ]; then
    echo ""
    echo "=== Publishing artifacts ==="
    if [ "$FORCE" = true ]; then
        python3 "$REPO_ROOT/scripts/publish_run.py" "$RUN_ID" --force
    else
        python3 "$REPO_ROOT/scripts/publish_run.py" "$RUN_ID"
    fi
    PUBLISH_EXIT=$?
    if [ $PUBLISH_EXIT -ne 0 ]; then
        echo "✗ Publish failed!"
        exit $PUBLISH_EXIT
    fi
fi
