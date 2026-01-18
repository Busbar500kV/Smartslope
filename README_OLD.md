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

### Publishing Artifacts to GitHub

By default, outputs are written to `outputs/` which is git-ignored. To publish curated artifacts to GitHub for traceability:

```bash
# Run smoke test and publish artifacts
bash scripts/smoke_run.sh --publish

# Or run pipeline directly with publish
source .venv/bin/activate
bash scripts/run_pipeline.sh --publish
```

The `--publish` flag will:
- Copy selected artifacts (PNG, TXT, JSON) to `artifacts/runs/<run_id>/`
- Create a manifest file with run metadata
- Update the global index at `artifacts/index.json`
- Commit and push to GitHub (only on main branch with clean working tree)

**Safety checks:**
- Requires clean git working tree (no uncommitted changes)
- Requires main branch (or use `--force` to override)
- Only commits small files (PNG plots, TXT summaries, JSON metadata)
- Does NOT commit large binary data (NPZ files remain in git-ignored `data/`)

**Examples:**
```bash
# Default: no publish (outputs remain git-ignored)
bash scripts/smoke_run.sh

# Publish to GitHub
bash scripts/smoke_run.sh --publish

# Force publish from non-main branch (use with caution)
bash scripts/smoke_run.sh --publish --force
```

### Output Location

Results are written to `outputs/run_<timestamp>/`:
- `kegalle_demo_score.png` - Coherence score plot
- `kegalle_demo_summary.txt` - Summary statistics
- `run_<timestamp>.zip` - **ZIP bundle of all outputs** (e.g., `run_20260117_123456.zip` for easy download/sharing)

For 3D runs (`--3d` flag), comprehensive deployment-grade outputs are generated:
- `scene_3d.png`, `scene_3d_before_after.png` - 3D installation geometry
- `scene_plan_view.png`, `scene_elevation_view.png` - 2D engineering views (NEW)
- `geometry_metrics.csv`, `geometry_metrics.json` - Sensitivity metrics (NEW)
- `timeseries_<name>.png` - Detailed time-series per reflector
- `timeseries_grid.png` - Compact overview of all reflectors
- `alarm_log.csv`, `alarm_state.json`, `alarm_ack_template.json` - Enhanced alarms (UPDATED)
- `scada_telemetry.csv` - SCADA-friendly telemetry export (NEW)
- `alarm_timeline.png` - Alarm timeline visualization
- `hmi_station.png` - HMI dashboard with geometry + ack tiles (UPDATED)
- `results.md` - Primary human-readable report (UPDATED)
- `report.md` - Technical report
- `manifest.json` - Machine-readable metadata

See **Deployment-Grade Features** section below for detailed documentation.

Generated synthetic data is stored in `data/synthetic/`.

### ZIP Bundle Workflow (iPhone/Mobile Friendly)

Each run automatically creates a ZIP bundle at `outputs/run_<timestamp>/run_<timestamp>.zip` containing all outputs (PNG, TXT, JSON, MD files). This makes it easy to:
1. **Download from Busbar via SCP:**
   ```bash
   scp busbar:~/Smartslope/outputs/run_20260117_123456/run_20260117_123456.zip ~/Downloads/
   ```

2. **Download from GitHub artifacts** (when using `--publish`):
   - Navigate to `artifacts/runs/<run_id>/` in GitHub
   - Download the ZIP file
   - On iPhone: Open in Files app and share to ChatGPT or other apps

3. **Share to ChatGPT or other tools** on iPhone for analysis/review

**Note:** By default, `.npz` files (large binary datasets) are excluded from the ZIP to keep file sizes small. Use `--include-npz` with the bundle script if you need them.

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

### Manual ZIP Bundling

If you need to manually create a ZIP bundle for a run:

```bash
# Bundle a specific run (excludes .npz by default)
python3 scripts/bundle_run.py --run-dir outputs/run_20260117_123456

# Include NPZ files (large binary data)
python3 scripts/bundle_run.py --run-dir outputs/run_20260117_123456 --include-npz

# Custom ZIP name
python3 scripts/bundle_run.py --run-dir outputs/run_20260117_123456 --zip-name my_bundle.zip
```

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
- `artifacts/` : curated published artifacts (tracked in git, created when using `--publish`)

## Deployment-Grade Features

### Geometry & Sensitivity Metrics

The 3D pipeline computes comprehensive geometry and sensitivity metrics for installation analysis:

**Output Files:**
- `geometry_metrics.csv` - Tabular geometry data
- `geometry_metrics.json` - Machine-readable geometry data

**Metrics Computed:**
- **Range** (m): Distance from radar to each reflector
- **LOS unit vector**: Line-of-sight direction (x, y, z)
- **Vertical angle** (degrees): Elevation angle from horizontal
- **Azimuth** (degrees): Bearing angle (0=North, 90=East)
- **Sensitivity**: Phase-to-displacement conversion factors
  - `sensitivity_mm_per_rad` = λ/(4π) × 1000 [mm/rad]
  - `sensitivity_rad_per_mm` = (4π/λ) / 1000 [rad/mm]
- **LOS gain for motion**: Projection factor of true motion onto LOS
  - `1.0` = motion directly along LOS (maximum sensitivity)
  - `0.0` = motion perpendicular to LOS (invisible)
  - `-1.0` = motion opposite to LOS (maximum sensitivity, inverted)

**Use Cases:**
- Validate installation geometry before deployment
- Assess target visibility and sensitivity
- Predict measurement accuracy for different motion directions
- Optimize radar placement for slope monitoring

### 2D Engineering Views

In addition to 3D scene plots, the pipeline generates 2D engineering views optimized for mobile viewing:

**scene_plan_view.png** - Top-down (X-Y) view:
- Shows radar and reflector positions in horizontal plane
- Motion direction arrows projected to ground plane
- Includes compass rose for orientation
- Useful for site planning and access routes

**scene_elevation_view.png** - Side profile (Distance vs Height):
- Shows radar height and reflector elevations
- Distance measured along ground (horizontal)
- Before/after displacement visualization
- Critical for assessing line-of-sight clearance

Both views are rendered at 150 DPI and sized for iPhone readability.

### Alarm Operations

The alarm system includes deployment-grade operational features:

#### Latching & Acknowledgment

**Latching Behavior:**
- Alarms with severity `ALARM` or `CRITICAL` will **latch** (persist) until acknowledged
- Latched alarms remain visible even if the condition clears
- Prevents operators from missing transient critical events

**Acknowledgment Workflow:**
1. System generates `alarm_ack_template.json` with instructions
2. Operator creates `alarm_ack.json` in run directory with acknowledgment records
3. Each acknowledgment includes: `alarm_id`, `ack_time_iso`, `ack_by`, `note`
4. Re-run pipeline to apply acknowledgments and update alarm state

**Alarm States:**
- `ACTIVE` - Condition currently true, not latched
- `LATCHED` - Condition may have cleared but alarm persists until acknowledged
- `ACKED` - Operator has acknowledged the alarm

#### K-of-N Slope Consensus

The system implements **K-of-N consensus** logic to detect coordinated slope movement:
- Alarm code: `SLOPE_EVENT_SYSTEM` (CRITICAL)
- Triggers when K or more slope targets show `SLOPE_MOVE` within a rolling time window
- Default: K=2, window=10 samples
- Helps distinguish individual target events from system-wide slope failures
- Alarm message lists all contributing targets

#### Escalation

**Automatic escalation** for unacknowledged critical alarms:
- If a `CRITICAL` alarm remains unacknowledged for >30 minutes (configurable), escalation is triggered
- Escalation status shown in HMI dashboard and `alarm_state.json`
- Provides second-level notification for critical safety events

**Configuration:**
```json
"alarm_ops": {
  "latch_severities": ["ALARM", "CRITICAL"],
  "escalation_after_minutes": 30
}
```

**Output Files:**
- `alarm_log.csv` - Full alarm log with `state`, `latched`, `acked` columns
- `alarm_state.json` - Snapshot of alarm states at end of run
- `alarm_ack_template.json` - Template for acknowledgment file

### SCADA Telemetry Export

For integration with external monitoring systems, the pipeline exports SCADA-friendly telemetry:

**scada_telemetry.csv** - Time-series data at each sample:
- `timestamp_iso` - ISO8601 timestamp
- `run_id` - Run identifier
- `radar_frequency_ghz` - Radar frequency
- `tx_power_dbm` - Transmit power
- `system_noise_metric` - Median phase noise across targets
- `system_dropout_fraction` - Fraction of invalid samples
- `ref_move_metric_mm` - Maximum reference displacement
- `slope_event_metric_mm_per_hr` - Maximum slope rate
- `alarm_counts_active` - Count of active alarms
- `alarm_counts_latched` - Count of latched alarms
- `alarm_counts_critical_active` - Count of critical alarms
- `system_status` - Overall status (OK/WARN/ALARM/CRITICAL)

This CSV can be imported into SCADA systems, Grafana dashboards, or other monitoring tools for real-time visualization and alerting.

### HMI Dashboard Updates

The `hmi_station.png` dashboard now includes:

**Geometry Sensitivity Tile:**
- Range span (min-max)
- LOS gain range for slope targets
- Quick reference for installation quality

**Acknowledgment Pending Indicator:**
- Shows count of unacknowledged ALARM/CRITICAL alarms
- Red border when acknowledgments pending
- Green when all clear

**Escalation Banner:**
- Visible red banner when escalation active
- Indicates critical alarms unacknowledged beyond threshold
- Prompts immediate operator attention

### Output Summary (3D Mode)

Complete list of files generated by `bash scripts/smoke_run.sh --3d`:

**Scene Visualizations:**
- `scene_3d.png` - 3D installation geometry
- `scene_3d_before_after.png` - Before/after displacement
- `scene_plan_view.png` - Top-down plan view (NEW)
- `scene_elevation_view.png` - Side elevation view (NEW)

**Time-Series Plots:**
- `timeseries_<name>.png` - Individual reflector plots (up to 12)
- `timeseries_grid.png` - Compact grid overview

**Geometry & Metrics:**
- `geometry_metrics.csv` - Geometry data (NEW)
- `geometry_metrics.json` - Geometry data (NEW)

**Alarms & Monitoring:**
- `alarm_log.csv` - Full alarm log with state columns (UPDATED)
- `alarm_state.json` - Alarm state snapshot (NEW)
- `alarm_ack_template.json` - Acknowledgment template (NEW)
- `alarm_timeline.png` - Alarm timeline visualization
- `scada_telemetry.csv` - SCADA-friendly telemetry (NEW)

**HMI & Reports:**
- `hmi_station.png` - HMI dashboard (UPDATED)
- `results.md` - Primary human-readable report (UPDATED)
- `report.md` - Technical report
- `manifest.json` - Machine-readable metadata
- `run_<timestamp>.zip` - ZIP bundle of all outputs

All files are automatically bundled into the ZIP for easy download and sharing.
