# Smartslope
Synthetic data and algorithms for radar-based slope movement detection - by යසස් පොන්වීර

Objective: build and calibrate algorithms for slope deformation detection using coherent radar concepts,
starting with synthetic data for any planned installation geometry.

## Quick Start (Busbar)

After a fresh clone on Linux (Busbar):

```bash
# Clone the repository
git clone https://github.com/Busbar500kV/Smartslope.git
cd Smartslope

# Run smoke test (creates venv, installs package, runs pipeline)
bash scripts/smoke_run.sh
```

This will:
1. Create a virtual environment at `.venv/` in the repo root
2. Install the `smartslope` package in editable mode
3. Generate synthetic coherent phase data
4. Run baseline detection
5. Write outputs to `outputs/run_<timestamp>/`

### Output Location

Results are written to `outputs/run_<timestamp>/`:
- `kegalle_demo_score.png` - Coherence score plot
- `kegalle_demo_summary.txt` - Summary statistics

Generated synthetic data is stored in `data/synthetic/`.

### Manual Usage

After running `scripts/smoke_run.sh` once, you can activate the venv and use the CLI directly:

```bash
source .venv/bin/activate

# Run individual commands
smartslope simulate  # Generate synthetic data
smartslope detect    # Run detection
smartslope pipeline  # Run both in sequence
```

## Workflow (Busbar)
- All code, configs, synthetic datasets, and outputs live in this repo.
- Run generation + detection on Busbar via SSH.
- Commit generated artifacts back to GitHub.

## Current scope
- Synthetic generator for coherent phase time-series per reflector (slope + reference targets)
- Drift/noise/dropouts modeling
- Baseline detection pipeline (drift removal, phase→displacement, coherence/persistence event logic)

## Repo layout
- `smartslope/` : Python package (installable via `pip install -e .`)
- `code/` : legacy scripts (being migrated)
- `scripts/` : Shell scripts for setup and running
- `data/` : generated synthetic datasets (kept small, git-ignored except .gitkeep)
- `outputs/` : plots + reports (git-ignored except .gitkeep)
