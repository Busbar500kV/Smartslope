# Smartslope Digital Twin
**Physics-based radar slope deformation monitoring** - by යසස් පොන්වීර

Smartslope is a **Digital Twin** for coherent radar-based slope monitoring systems. It simulates the complete measurement chain from radar physics to alarm generation, enabling:

- **Pre-deployment analysis**: Validate installation geometry, beam coverage, and sensitivity
- **Algorithm development**: Test detection logic with realistic synthetic data including terrain, atmosphere, and multipath effects
- **Real-data integration**: Ingest and replay field measurements through the same analysis pipeline
- **Operational readiness**: Multi-radar redundancy, site-level status, and deployment-grade outputs

## Quick Start (Busbar Linux)

```bash
# Clone repository
git clone https://github.com/Busbar500kV/Smartslope.git
cd Smartslope

# Run 3D synthetic simulation with full outputs
bash scripts/smoke_run.sh --3d

# Publish artifacts to GitHub (creates ZIP bundle)
bash scripts/smoke_run.sh --3d --publish
```

This generates a complete run with:
- **3D scene visualizations** (before/after, plan view, elevation view)
- **Geometry & sensitivity metrics** (range, LOS gain, beam coverage)
- **Time-series plots** (phase, displacement, validity masks)
- **Alarm timeline & HMI dashboard** (latching, acknowledgment, escalation)
- **SCADA telemetry export** (CSV for external monitoring)
- **ZIP bundle** (all outputs packaged for download/sharing)

**Latest published run**: [artifacts/runs/run_20260118_151347/results.md](artifacts/runs/run_20260118_151347/results.md)

## Data Flow

```
Station Config (JSON)
    ↓
┌───────────────────────────────┐
│ Simulation or Data Ingest     │ ← DEM, Atmosphere, Beam, Multipath
│  • 3D geometry                 │
│  • Physics modeling            │
│  • CSV/NPZ import              │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│ Analysis Pipeline              │
│  • Geometry metrics            │
│  • Alarm generation            │
│  • Site-level status           │
└───────────────────────────────┘
    ↓
┌───────────────────────────────┐
│ Outputs                        │
│  • results.md (primary report) │
│  • HMI dashboard               │
│  • SCADA telemetry CSV         │
│  • Alarm log & state           │
│  • ZIP bundle                  │
└───────────────────────────────┘
```

## Station Design Parameters

Configure all aspects of your radar installation in JSON format. See `configs/site_demo_3d.json` for a complete example.

### Radar Configuration
- **Single or multi-radar**: Support for redundancy and diversity
- **Position**: `position_xyz_m` - radar location in site coordinates (East, North, Up)
- **Frequency/wavelength**: `frequency_hz` or `wavelength_m` (e.g., 24 GHz K-band)
- **Transmit power**: `tx_power_dbm` (for SNR modeling)
- **Sampling**: `sample_period_s` (measurement interval)

### Beam Pattern (NEW)
```json
"beam": {
  "enabled": true,
  "yaw_deg": 0,           // Beam pointing (0=North)
  "pitch_deg": 5,         // Elevation angle
  "beamwidth_deg": 25,    // 3dB beamwidth
  "pattern": "gaussian"   // or "cosine"
}
```
Maps beam attenuation to SNR and phase noise per reflector.

### Reflectors
- **Role**: `"ref"` (reference) or `"slope"` (monitoring target)
- **Position**: `position_xyz_m` - corner reflector location
- **Height**: `height_m` - mounting height above local ground
- **Amplitude & noise**: `amplitude`, `noise_phase_sigma_rad`, `dropout_prob`
- **Visibility**: `visible_to` - list of radar names (multi-radar only)
- **Motion model**: 
  - `"creep"`: Steady mm/hr motion
  - `"creep_plus_event"`: Creep + triggered slip event (sigmoid or triangle)
  - Direction: `direction_xyz_unit` (3D unit vector)

### Terrain / DEM (NEW)
```json
"terrain": {
  "enabled": true,
  "dem_npz_path": "data/terrain/site_dem.npz",
  "apply_occlusion_masking": true
}
```
- **NPZ format**: `dem_x_m`, `dem_y_m`, `dem_z_m` (2D elevation grid)
- **Occlusion checking**: Ray-traces LOS to detect terrain blockage
- **Height-above-ground**: Validates reflector mounting heights

### Atmosphere (NEW)
```json
"atmosphere": {
  "enabled": true,
  "common_ar1_rho": 0.995,       // Temporal correlation (slow decorrelation)
  "common_sigma_rad": 0.3,       // Common-mode phase RMS
  "local_sigma_rad": 0.08,       // Per-reflector turbulence
  "airmass_model": "secant",     // or "range_linear"
  "airmass_range_scale_m": 1000.0
}
```
Simulates atmospheric phase screen with:
- **Time-correlated noise**: AR(1) process for realistic temporal structure
- **Airmass weighting**: Longer paths through atmosphere = higher phase noise
- **Local turbulence**: Per-reflector independent component

### Multipath Bias (NEW)
```json
"multipath": {
  "enabled": true,
  "bias_sigma_rad": 0.15,   // Phase bias magnitude
  "bias_rho": 0.999          // Very slow AR(1) variation
}
```
Non-common-mode phase bias that references help suppress.

### Drift Model
- **Random walk**: `sigma_rad_per_sqrt_s` - Brownian motion
- **Sine wave**: `period_s`, `amplitude_rad` - Deterministic drift

### Alarms
```json
"alarms": {
  "enable": true,
  "ref_move_mm_threshold": 2.0,          // Reference stability alarm
  "slope_move_mm_threshold": 5.0,        // Slope movement detection
  "slope_rate_mm_per_hr_threshold": 10.0,
  "slope_k_of_n_consensus": 2,           // K-of-N logic for system event
  "slope_consensus_window_samples": 10,
  "high_noise_phase_sigma_rad_threshold": 0.6,
  "dropout_rate_threshold": 0.25
}
```

### Alarm Operations
```json
"alarm_ops": {
  "latch_severities": ["ALARM", "CRITICAL"],  // Latching behavior
  "escalation_after_minutes": 30               // Auto-escalation timeout
}
```
- **Latching**: ALARM/CRITICAL alarms persist until acknowledged
- **Acknowledgment workflow**: `alarm_ack.json` with operator notes
- **Escalation**: Unacknowledged CRITICAL alarms escalate after timeout

### Multi-Radar Redundancy (NEW)
For multi-radar configs:
```json
"site_redundancy": {
  "min_radar_confirmations": 1  // Require M radars to confirm target alarm
}
```
Site-level status aggregates per-radar observations with confirmation logic.

## Real Data Integration

### Replay Mode
Process real radar exports through the full pipeline:

```bash
# From NPZ file
smartslope replay --input data/field/export_20260118.npz --outdir outputs/replay_001

# From CSV with mapping
smartslope replay \
  --input data/field/export.csv \
  --mapping configs/mappings/example_generic_mapping.json \
  --site configs/site_demo_3d.json \
  --outdir outputs/replay_002 \
  --publish
```

Produces same outputs as synthetic runs: `results.md`, HMI, alarms, telemetry.

### File Watcher (Optional)
Auto-process new files as they arrive:

```bash
smartslope watch \
  --folder /mnt/radar_exports \
  --pattern "*.csv" \
  --mapping configs/mappings/site_mapping.json \
  --site configs/site_config.json \
  --out-root outputs/live
```

Polls folder every 30s (configurable) and creates `outputs/live/replay_<timestamp>` for each new file.

### CSV Mapping Format
Define column mapping in JSON (see `configs/mappings/example_generic_mapping.json`):

```json
{
  "timestamp_column": "timestamp",
  "timestamp_format": "iso8601",  // or "unix_epoch"
  "reflector_columns": {
    "REF_A": "phase_ref_a_rad",
    "SLOPE_01": "phase_slope_01_rad"
  },
  "reflector_roles": {
    "REF_A": "ref",
    "SLOPE_01": "slope"
  },
  "validity_columns": {
    "REF_A": "valid_ref_a"
  }
}
```

## Output Files

### Key Deliverables
- **`results.md`**: Primary human-readable report with embedded plots and status summary
- **`hmi_station.png`**: Dashboard with geometry, alarms, acknowledgment status
- **`scada_telemetry.csv`**: Time-series data for SCADA/Grafana import
- **`alarm_log.csv`**: Full alarm log with state tracking
- **`alarm_state.json`**: Current alarm snapshot (latched/acked status)
- **`site_status.json`**: Site-level status (multi-radar aggregation)
- **`geometry_metrics.csv/.json`**: Range, azimuth, elevation, LOS gain per reflector
- **`run_<timestamp>.zip`**: All outputs bundled for download

### Visualization Outputs
- **`scene_3d.png`**: 3D installation geometry with motion vectors
- **`scene_3d_before_after.png`**: Before/after displacement visualization
- **`scene_plan_view.png`**: Top-down (X-Y) site layout
- **`scene_elevation_view.png`**: Side profile (distance vs height)
- **`timeseries_<name>.png`**: Per-reflector detailed plots
- **`timeseries_grid.png`**: Compact overview of all reflectors
- **`alarm_timeline.png`**: Alarm history visualization

## Deployment Workflow

### Pre-Deployment Analysis
1. Create site config JSON with candidate radar positions and reflector locations
2. Run simulation: `bash scripts/smoke_run.sh --3d`
3. Review `geometry_metrics.csv` for LOS gain and range span
4. Iterate radar/reflector positions to optimize coverage
5. Check `scene_plan_view.png` and `scene_elevation_view.png` for access routes and clearances

### Field Commissioning
1. Install radar and reflectors per validated design
2. Collect initial phase measurements (CSV or NPZ)
3. Run replay: `smartslope replay --input <data> --site <config> --publish`
4. Verify geometry metrics match expectations
5. Calibrate alarm thresholds based on actual noise/drift

### Operational Monitoring
1. Configure file watcher or periodic replay runs
2. Monitor `results.md`, HMI dashboard, and SCADA telemetry
3. Respond to latched alarms via acknowledgment workflow
4. Archive published runs to GitHub for traceability

## Repository Structure

```
Smartslope/
├── smartslope/          # Python package (installable via pip)
│   ├── sim3d.py         # 3D physics-based simulator
│   ├── dem.py           # Terrain/DEM support (NEW)
│   ├── atmosphere.py    # Atmospheric phase (NEW)
│   ├── site.py          # Multi-radar config (NEW)
│   ├── ingest.py        # CSV/NPZ import (NEW)
│   ├── schema.py        # Dataset validation (NEW)
│   ├── site_status.py   # Redundancy logic (NEW)
│   ├── alarms.py        # Alarm generation
│   ├── hmi.py           # Dashboard rendering
│   ├── geometry_metrics.py
│   ├── scada_export.py
│   └── ...
├── scripts/
│   ├── smoke_run.sh     # End-to-end test
│   ├── run_pipeline.sh
│   ├── bundle_run.py
│   └── check_unicode_controls.py (NEW)
├── configs/
│   ├── site_demo_3d.json            # Single-radar demo
│   ├── site_demo_multi_radar.json   # Multi-radar demo (NEW)
│   ├── ci_demo_3d.json              # Fast CI config (NEW)
│   └── mappings/
│       └── example_generic_mapping.json (NEW)
├── artifacts/           # Published runs (tracked in git)
├── data/                # Synthetic datasets (git-ignored)
├── outputs/             # Run outputs (git-ignored)
└── README.md            # This file
```

## Dependencies

Minimal dependencies for maximum portability:
- **Python 3.10+**
- **NumPy** ≥ 1.24 (array operations)
- **Matplotlib** ≥ 3.7 (plotting)

No external radar libraries, web frameworks, or databases required.

## Development

```bash
# Set up development environment
bash scripts/setup_venv.sh
source .venv/bin/activate

# Run individual commands
smartslope simulate --config configs/site_demo_3d.json --outdir outputs/test
smartslope replay --input data/example.npz --outdir outputs/replay_test

# Compile and check
python -m compileall smartslope/
python scripts/check_unicode_controls.py

# Run CI checks locally
bash scripts/smoke_run.sh --3d
```

## License & Attribution

Developed by Busbar500kV (යසස් පොන්වීර) for deployment on Busbar Linux systems.

---

**Digital Twin**: A virtual replica that mirrors physical system behavior, enabling testing, optimization, and operational insights before deployment.
