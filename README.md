# Smartslope
Synthetic data and algorithms for radar-based slope movement detection - by යසස් පොන්වීර

## Objective
Build and calibrate algorithms for slope deformation detection using coherent radar concepts, starting with synthetic data for any planned installation geometry.

## Philosophy
1. Start with physics-consistent synthetic data
2. Validate algorithms on synthetic truth first
3. Only later adapt to real radar data

The core signal is **coherent radar phase**. Displacement is derived from phase, not the other way around.

## Current Scope
- **Synthetic generator** for coherent phase time-series:
  - Slope reflectors + reference (stable) reflectors
  - Slow creep, accelerating motion, step displacements
  - Common-mode drift (instrument/atmosphere)
  - Phase noise and data dropouts
  
- **Baseline detection pipeline**:
  - Phase unwrapping
  - Drift removal using reference targets
  - Phase → displacement conversion
  - Velocity estimation
  - Persistence/coherence-based event detection

## Repository Structure
```
smartslope/          # Python package
  ├── synthetic.py   # Synthetic data generation
  ├── detection.py   # Detection pipeline
  ├── math_utils.py  # Mathematical utilities
  ├── io_utils.py    # I/O utilities
  └── configs/       # Configuration files
scripts/             # Shell orchestrators
  └── run_pipeline.sh  # Main execution entrypoint
data/                # Generated synthetic datasets (kept small)
outputs/             # Plots and reports
```

## Quick Start (on Busbar)

```bash
# Run the complete pipeline
./scripts/run_pipeline.sh
```

This will:
1. Set up Python virtual environment (if needed)
2. Install dependencies
3. Generate synthetic phase data
4. Run detection pipeline
5. Output plots and reports to `outputs/synthetic/`

## Workflow (Busbar)
- All code, configs, synthetic datasets, and outputs live in this repo
- Run generation + detection on Busbar via shell scripts
- Commit generated artifacts back to GitHub as needed
- No hidden state, no magic globals