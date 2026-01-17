#!/bin/sh
# Run the Smartslope pipeline on Busbar
# Requires setup_venv.sh to have been run first

set -e

echo "=== Smartslope Pipeline Runner ==="

# Repo root (parent of scripts/)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Venv location
VENV_DIR="$HOME/venvs/smartslope"

# Check venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at: $VENV_DIR"
    echo "Please run ./scripts/setup_venv.sh first."
    exit 1
fi

# Activate venv
echo "Activating virtual environment..."
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

# Create timestamped run folder
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="outputs/run_$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "Run directory: $RUN_DIR"
echo ""

# Run the pipeline
CONFIG="${1:-code/synthetic/configs/kegalle_demo.json}"
echo "Config: $CONFIG"
echo ""

python -m smartslope.cli --config "$CONFIG" --outdir "$RUN_DIR"

echo ""
echo "=== Pipeline Complete ==="
echo "Outputs saved to: $RUN_DIR"
echo ""
