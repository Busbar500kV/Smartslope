#!/bin/sh
set -eu

echo "== Smartslope: run_all.sh =="

# Always run from repo root
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "Repo root: $REPO_ROOT"

# --- Python / venv handling ---
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate

echo "Python: $(python --version)"

# Install deps (idempotent)
python -m pip install -U pip
python -m pip install -e . || python -m pip install numpy matplotlib

# --- Make sure scripts are executable ---
chmod +x code/synthetic/scripts/*.sh

# --- Run synthetic generator ---
echo "Running synthetic data generation..."
./code/synthetic/scripts/run_synth.sh

# --- Run detection ---
echo "Running detection..."
./code/synthetic/scripts/run_detect.sh

echo "== Smartslope run_all complete =="