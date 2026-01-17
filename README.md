# Smartslope
Synthetic data and algorithms for radar-based slope movement detection - by යසස් පොන්වීර

Objective: build and calibrate algorithms for slope deformation detection using coherent radar concepts,
starting with synthetic data for any planned installation geometry.

## Quick Start (Busbar)

On your Ubuntu-like Linux machine with Python 3.10+:

```bash
# 1. Clone the repository
git clone https://github.com/Busbar500kV/Smartslope.git
cd Smartslope

# 2. Setup virtual environment and install package
./scripts/setup_venv.sh

# 3. Run the pipeline
./scripts/run_pipeline.sh

# Or run both setup + pipeline (smoke test)
./scripts/smoke_run.sh
```

The pipeline generates:
- Synthetic dataset in `data/` (e.g., `kegalle_demo_20260117_183045.npz`)
- Plots and reports in `outputs/run_<timestamp>/`

### Manual Execution

If you prefer manual control:

```bash
# Activate the virtual environment
source ~/venvs/smartslope/bin/activate

# Run with default config
python -m smartslope.cli

# Run with custom config and output directory
python -m smartslope.cli --config path/to/config.json --outdir outputs/custom_run
```

## Development

Install in editable mode for development:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Workflow (Busbar):
- All code, configs, synthetic datasets, and outputs live in this repo.
- Run generation + detection on Busbar via SSH.
- Commit generated artifacts back to GitHub.

Current scope:
- Synthetic generator for coherent phase time-series per reflector (slope + reference targets)
- Drift/noise/dropouts modeling
- Baseline detection pipeline (drift removal, phase→displacement, coherence/persistence event logic)

Repo layout:
- `smartslope/` : Python package (installable)
- `code/` : legacy scripts and configs
- `scripts/` : Busbar helper scripts
- `data/` : generated synthetic datasets (git-ignored except .gitkeep)
- `outputs/` : plots + reports (git-ignored except .gitkeep)