# Smartslope 3D Installation Simulation Report

## Radar Installation

**Name:** Demo_Radar_01

**Position (x, y, z):** (0.00, 0.00, 2.00) m

**Wavelength:** 12.50 mm (0.012500 m)

**Frequency:** 24.00 GHz

**Coordinate System:** x=East, y=North, z=Up (all in meters)

## Environment

**Duration:** 172800 s (48.00 hours)

**Sample Interval (dt):** 600 s (10.00 min)

**Number of Samples:** 288

## Drift Model

**Model:** random_walk

- sigma_rad_per_sqrt_s: 2e-05

## Reflectors

| Name | Role | Position (x, y, z) [m] | Height [m] | Range [m] | Incidence [deg] | Amplitude | Noise σ [rad] | Dropout % | Motion Model |
|------|------|------------------------|------------|-----------|-----------------|-----------|---------------|-----------|---------------|
| REF_A | ref | (200.0, -50.0, 10.0) | 10.0 | 206.3 | 92.2 | 1.00 | 0.080 | 1.0 | none |
| REF_B | ref | (210.0, 40.0, 12.0) | 12.0 | 214.0 | 92.7 | 1.20 | 0.070 | 1.0 | none |
| SLOPE_01 | slope | (1000.0, -80.0, 60.0) | 60.0 | 1004.9 | 93.3 | 0.90 | 0.100 | 2.0 | creep_plus_event |
| SLOPE_02 | slope | (1020.0, -10.0, 55.0) | 55.0 | 1021.4 | 93.0 | 0.80 | 0.120 | 3.0 | creep_plus_event |
| SLOPE_03 | slope | (990.0, 60.0, 50.0) | 50.0 | 993.0 | 92.8 | 0.85 | 0.110 | 2.0 | creep |

## Motion Models

### SLOPE_01

**Model:** creep_plus_event

**Direction (unit vector):** (0.980, 0.000, -0.200)

**Creep Rate:** 1.250 mm/hr

**Event:**
- Start time (t0): 86400 s (24.00 hr)
- Rise time: 7200 s (2.00 hr)
- Slip magnitude: 50.00 mm
- Shape: sigmoid

### SLOPE_02

**Model:** creep_plus_event

**Direction (unit vector):** (0.980, 0.000, -0.200)

**Creep Rate:** 1.250 mm/hr

**Event:**
- Start time (t0): 86400 s (24.00 hr)
- Rise time: 7200 s (2.00 hr)
- Slip magnitude: 50.00 mm
- Shape: sigmoid

### SLOPE_03

**Model:** creep

**Direction (unit vector):** (0.980, 0.000, -0.200)

**Creep Rate:** 0.500 mm/hr

## Generated Outputs

This simulation generated the following files:

- `alarm_log.csv`
- `alarm_timeline.png`
- `hmi_station.png`
- `results.md`
- `scene_3d.png`
- `scene_3d_before_after.png`
- `timeseries_REF_A.png`
- `timeseries_REF_B.png`
- `timeseries_SLOPE_01.png`
- `timeseries_SLOPE_02.png`
- `timeseries_SLOPE_03.png`
- `timeseries_grid.png`

## How to Interpret the Outputs

### Scene Plots

- **scene_3d.png**: Shows the 3D installation geometry with radar and all reflectors. Reference reflectors are blue circles, slope reflectors are orange squares. Red arrows show motion directions for slope targets.

- **scene_3d_before_after.png**: Shows original positions (faint) and displaced positions (solid) at a specific time. Green arrows indicate displacement vectors.

### Time-Series Plots

- **timeseries_<name>.png**: Detailed time-series for individual reflectors showing:
  - True displacement magnitude (3D vector norm)
  - LOS displacement (projection onto radar line-of-sight)
  - Unwrapped phase (proportional to LOS displacement)
  - Wrapped phase (actual measured phase in [-π, π])
  - Valid mask (indicates dropout events)

- **timeseries_grid.png**: Compact overview of all reflectors (LOS displacement and phase).

### Phase-Displacement Relationship

The coherent radar measures phase change proportional to LOS displacement:

```
Δφ = (4π/λ) × Δr_LOS
```

For this installation with λ = 12.50 mm:
- 1 mm LOS displacement = 1.005 radians
- 2π radians (one fringe) = 6.250 mm LOS displacement

