"""Report generation and time-series plotting for synthetic data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from smartslope.scene import compute_range, compute_incidence_angle, compute_los_vector

# Constants
MOTION_MODEL_NONE = 'none'


def plot_reflector_timeseries(
    t_s: np.ndarray,
    disp_true_xyz_m: np.ndarray,
    disp_los_m: np.ndarray,
    phi_unwrapped: np.ndarray,
    phi_wrapped: np.ndarray,
    mask_valid: np.ndarray,
    name: str,
    output_path: Path
) -> None:
    """
    Plot time-series for a single reflector.
    
    Args:
        t_s: Time array (T,)
        disp_true_xyz_m: True 3D displacement (T, 3)
        disp_los_m: LOS displacement (T,)
        phi_unwrapped: Unwrapped phase (T,)
        phi_wrapped: Wrapped phase (T,)
        mask_valid: Valid data mask (T,)
        name: Reflector name
        output_path: Output file path
    """
    t_hr = t_s / 3600.0  # Convert to hours
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    
    # 1. True displacement magnitude
    disp_mag_mm = np.linalg.norm(disp_true_xyz_m, axis=1) * 1000.0  # to mm
    axes[0].plot(t_hr, disp_mag_mm, 'b-', linewidth=1.5)
    axes[0].set_ylabel('True Disp\nMagnitude (mm)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Reflector: {name}')
    
    # 2. LOS displacement
    disp_los_mm = disp_los_m * 1000.0  # to mm
    axes[1].plot(t_hr, disp_los_mm, 'g-', linewidth=1.5)
    axes[1].set_ylabel('LOS Disp (mm)')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Unwrapped phase
    axes[2].plot(t_hr, phi_unwrapped, 'r-', linewidth=1.5)
    axes[2].set_ylabel('Unwrapped\nPhase (rad)')
    axes[2].grid(True, alpha=0.3)
    
    # 4. Wrapped phase
    axes[3].plot(t_hr, phi_wrapped, 'm.', markersize=2)
    axes[3].set_ylabel('Wrapped\nPhase (rad)')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(-np.pi - 0.5, np.pi + 0.5)
    
    # 5. Valid mask
    axes[4].fill_between(t_hr, 0, mask_valid, color='gray', alpha=0.5)
    axes[4].set_ylabel('Valid')
    axes[4].set_xlabel('Time (hours)')
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_timeseries_grid(
    t_s: np.ndarray,
    disp_los_m: np.ndarray,
    phi_unwrapped: np.ndarray,
    mask_valid: np.ndarray,
    names: List[str],
    roles: List[str],
    output_path: Path,
    max_plots: int = 12
) -> None:
    """
    Plot time-series grid for multiple reflectors (compact view).
    
    Args:
        t_s: Time array (T,)
        disp_los_m: LOS displacement (N, T)
        phi_unwrapped: Unwrapped phase (N, T)
        mask_valid: Valid mask (N, T)
        names: Reflector names
        roles: Reflector roles
        output_path: Output file path
        max_plots: Maximum number of reflectors to plot
    """
    n_reflectors = min(len(names), max_plots)
    t_hr = t_s / 3600.0
    
    fig, axes = plt.subplots(n_reflectors, 2, figsize=(14, 2.5 * n_reflectors), sharex=True)
    
    if n_reflectors == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_reflectors):
        color = 'blue' if roles[i] == 'ref' else 'orange'
        
        # LOS displacement
        disp_los_mm = disp_los_m[i] * 1000.0
        axes[i, 0].plot(t_hr, disp_los_mm, color=color, linewidth=1.0)
        axes[i, 0].set_ylabel(f'{names[i]}\nLOS (mm)', fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)
        
        # Unwrapped phase
        axes[i, 1].plot(t_hr, phi_unwrapped[i], color=color, linewidth=1.0)
        axes[i, 1].set_ylabel(f'Phase (rad)', fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)
    
    axes[-1, 0].set_xlabel('Time (hours)')
    axes[-1, 1].set_xlabel('Time (hours)')
    
    plt.suptitle('Time-Series Summary', fontsize=12, weight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()


def generate_report(
    config: Dict,
    data: Dict[str, np.ndarray],
    output_dir: Path,
    generated_files: List[str]
) -> None:
    """
    Generate human-readable report.md.
    
    Args:
        config: Configuration dictionary
        data: Simulation output data
        output_dir: Output directory
        generated_files: List of generated file names
    """
    report_path = output_dir / 'report.md'
    
    radar_cfg = config['radar']
    env_cfg = config['environment']
    reflectors_cfg = config['reflectors']
    drift_cfg = config.get('drift_model', {})
    
    wavelength_m = float(data['wavelength_m'][0])
    radar_xyz = data['radar_xyz_m']
    pos_xyz = data['pos_xyz_m']
    names = data['names']
    roles = data['roles']
    
    # Compute geometry info
    ranges_m = []
    incidence_angles_deg = []
    for i in range(len(names)):
        los_vec = compute_los_vector(radar_xyz, pos_xyz[i])
        range_m = compute_range(radar_xyz, pos_xyz[i])
        incidence_deg = compute_incidence_angle(los_vec)
        ranges_m.append(range_m)
        incidence_angles_deg.append(incidence_deg)
    
    with open(report_path, 'w') as f:
        f.write("# Smartslope 3D Installation Simulation Report\n\n")
        
        # Radar installation
        f.write("## Radar Installation\n\n")
        f.write(f"**Name:** {radar_cfg['name']}\n\n")
        f.write(f"**Position (x, y, z):** ({radar_xyz[0]:.2f}, {radar_xyz[1]:.2f}, {radar_xyz[2]:.2f}) m\n\n")
        f.write(f"**Wavelength:** {wavelength_m * 1000:.2f} mm ({wavelength_m:.6f} m)\n\n")
        if 'frequency_hz' in radar_cfg:
            freq_ghz = radar_cfg['frequency_hz'] / 1e9
            f.write(f"**Frequency:** {freq_ghz:.2f} GHz\n\n")
        
        # Coordinate system
        f.write("**Coordinate System:** x=East, y=North, z=Up (all in meters)\n\n")
        
        # Environment
        f.write("## Environment\n\n")
        f.write(f"**Duration:** {env_cfg['duration_s']} s ({env_cfg['duration_s'] / 3600.0:.2f} hours)\n\n")
        f.write(f"**Sample Interval (dt):** {env_cfg['dt_s']} s ({env_cfg['dt_s'] / 60.0:.2f} min)\n\n")
        f.write(f"**Number of Samples:** {len(data['t_s'])}\n\n")
        
        # Drift model
        f.write("## Drift Model\n\n")
        if drift_cfg.get('enabled', False):
            f.write(f"**Model:** {drift_cfg.get('model', 'N/A')}\n\n")
            params = drift_cfg.get('parameters', {})
            for key, val in params.items():
                f.write(f"- {key}: {val}\n")
            f.write("\n")
        else:
            f.write("**Drift:** Disabled\n\n")
        
        # Reflector table
        f.write("## Reflectors\n\n")
        f.write("| Name | Role | Position (x, y, z) [m] | Height [m] | Range [m] | Incidence [deg] | Amplitude | Noise σ [rad] | Dropout % | Motion Model |\n")
        f.write("|------|------|------------------------|------------|-----------|-----------------|-----------|---------------|-----------|---------------|\n")
        
        for i, refl_cfg in enumerate(reflectors_cfg):
            name = refl_cfg['name']
            role = refl_cfg['role']
            xyz = refl_cfg['position_xyz_m']
            height = refl_cfg.get('height_m', xyz[2])
            amplitude = refl_cfg.get('amplitude', 1.0)
            noise_sigma = refl_cfg.get('noise_phase_sigma_rad', 0.1)
            dropout = refl_cfg.get('dropout_prob', 0.0) * 100.0
            
            motion_cfg = refl_cfg.get('motion')
            if motion_cfg is None:
                motion_model = MOTION_MODEL_NONE
            else:
                motion_model = motion_cfg.get('model', MOTION_MODEL_NONE)
            
            f.write(f"| {name} | {role} | ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) | {height:.1f} | "
                   f"{ranges_m[i]:.1f} | {incidence_angles_deg[i]:.1f} | {amplitude:.2f} | "
                   f"{noise_sigma:.3f} | {dropout:.1f} | {motion_model} |\n")
        
        f.write("\n")
        
        # Motion details
        f.write("## Motion Models\n\n")
        for refl_cfg in reflectors_cfg:
            motion_cfg = refl_cfg.get('motion')
            if motion_cfg is not None and motion_cfg.get('model') != MOTION_MODEL_NONE:
                name = refl_cfg['name']
                model = motion_cfg['model']
                f.write(f"### {name}\n\n")
                f.write(f"**Model:** {model}\n\n")
                
                if 'direction_xyz_unit' in motion_cfg:
                    direction = motion_cfg['direction_xyz_unit']
                    f.write(f"**Direction (unit vector):** ({direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f})\n\n")
                
                if 'creep_rate_mm_per_hr' in motion_cfg:
                    f.write(f"**Creep Rate:** {motion_cfg['creep_rate_mm_per_hr']:.3f} mm/hr\n\n")
                
                event = motion_cfg.get('event', {})
                if event.get('enabled', False):
                    f.write("**Event:**\n")
                    f.write(f"- Start time (t0): {event['t0_s']} s ({event['t0_s'] / 3600.0:.2f} hr)\n")
                    f.write(f"- Rise time: {event['rise_s']} s ({event['rise_s'] / 3600.0:.2f} hr)\n")
                    f.write(f"- Slip magnitude: {event['slip_mm']:.2f} mm\n")
                    f.write(f"- Shape: {event['shape']}\n")
                    f.write("\n")
        
        # Outputs
        f.write("## Generated Outputs\n\n")
        f.write("This simulation generated the following files:\n\n")
        for fname in sorted(generated_files):
            f.write(f"- `{fname}`\n")
        f.write("\n")
        
        # Interpretation guide
        f.write("## How to Interpret the Outputs\n\n")
        f.write("### Scene Plots\n\n")
        f.write("- **scene_3d.png**: Shows the 3D installation geometry with radar and all reflectors. "
               "Reference reflectors are blue circles, slope reflectors are orange squares. "
               "Red arrows show motion directions for slope targets.\n\n")
        f.write("- **scene_3d_before_after.png**: Shows original positions (faint) and displaced positions (solid) "
               "at a specific time. Green arrows indicate displacement vectors.\n\n")
        
        f.write("### Time-Series Plots\n\n")
        f.write("- **timeseries_<name>.png**: Detailed time-series for individual reflectors showing:\n")
        f.write("  - True displacement magnitude (3D vector norm)\n")
        f.write("  - LOS displacement (projection onto radar line-of-sight)\n")
        f.write("  - Unwrapped phase (proportional to LOS displacement)\n")
        f.write("  - Wrapped phase (actual measured phase in [-π, π])\n")
        f.write("  - Valid mask (indicates dropout events)\n\n")
        
        f.write("- **timeseries_grid.png**: Compact overview of all reflectors (LOS displacement and phase).\n\n")
        
        f.write("### Phase-Displacement Relationship\n\n")
        f.write("The coherent radar measures phase change proportional to LOS displacement:\n\n")
        f.write(f"```\nΔφ = (4π/λ) × Δr_LOS\n```\n\n")
        f.write(f"For this installation with λ = {wavelength_m * 1000:.2f} mm:\n")
        f.write(f"- 1 mm LOS displacement = {4 * np.pi / (wavelength_m * 1000):.3f} radians\n")
        f.write(f"- 2π radians (one fringe) = {wavelength_m / 2 * 1000:.3f} mm LOS displacement\n\n")
    
    print(f"Wrote {report_path}")


def generate_manifest(
    config: Dict,
    data: Dict[str, np.ndarray],
    output_dir: Path,
    generated_files: List[str]
) -> None:
    """
    Generate machine-readable manifest.json.
    
    Args:
        config: Configuration dictionary
        data: Simulation output data
        output_dir: Output directory
        generated_files: List of generated file names
    """
    manifest_path = output_dir / 'manifest.json'
    
    manifest = {
        'version': '1.0',
        'simulation_type': '3d_geometry_aware',
        'radar': config['radar'],
        'environment': config['environment'],
        'drift_model': config.get('drift_model', {}),
        'reflectors': {
            'count': len(data['names']),
            'names': data['names'].tolist(),
            'roles': data['roles'].tolist(),
        },
        'data_shapes': {
            't_s': list(data['t_s'].shape),
            'pos_xyz_m': list(data['pos_xyz_m'].shape),
            'disp_true_xyz_m': list(data['disp_true_xyz_m'].shape),
            'disp_los_m': list(data['disp_los_m'].shape),
            'phi_unwrapped': list(data['phi_unwrapped'].shape),
            'phi_wrapped': list(data['phi_wrapped'].shape),
        },
        'generated_files': generated_files,
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Wrote {manifest_path}")


def generate_results_md(
    config: Dict,
    data: Dict[str, np.ndarray],
    alarms: List,
    output_dir: Path,
    run_id: str
) -> None:
    """
    Generate human-readable results.md with embedded images.
    
    This is the primary output document that should be readable directly on GitHub.
    
    Args:
        config: Configuration dictionary
        data: Simulation output data
        alarms: List of AlarmEvent objects
        output_dir: Output directory
        run_id: Run identifier
    """
    results_path = output_dir / 'results.md'
    
    # Extract configuration
    hmi_cfg = config.get('hmi', {})
    radar_station_cfg = config.get('radar_station', {})
    radar_cfg = config['radar']
    env_cfg = config['environment']
    alarm_injection_cfg = config.get('alarm_injection', {})
    
    station_name = hmi_cfg.get('station_name', 'Demo Radar Station')
    site_id = hmi_cfg.get('site_id', 'N/A')
    
    # Reflector info
    names = data['names']
    roles = data['roles']
    pos_xyz = data['pos_xyz_m']
    wavelength_m = float(data['wavelength_m'][0])
    radar_xyz = data['radar_xyz_m']
    
    # Compute geometry info
    ranges_m = []
    incidence_angles_deg = []
    for i in range(len(names)):
        los_vec = compute_los_vector(radar_xyz, pos_xyz[i])
        range_m = compute_range(radar_xyz, pos_xyz[i])
        incidence_deg = compute_incidence_angle(los_vec)
        ranges_m.append(range_m)
        incidence_angles_deg.append(incidence_deg)
    
    with open(results_path, 'w') as f:
        # Title
        f.write(f"# Smartslope HMI Results: {station_name}\n\n")
        f.write(f"**Run ID:** `{run_id}`\n\n")
        f.write(f"**Site ID:** {site_id}\n\n")
        
        # Station parameters table
        f.write("## Station Parameters\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        
        freq_hz = radar_station_cfg.get('frequency_hz', radar_cfg.get('frequency_hz', 24e9))
        freq_ghz = freq_hz / 1e9
        f.write(f"| Frequency | {freq_ghz:.2f} GHz |\n")
        f.write(f"| Wavelength | {wavelength_m * 1000:.2f} mm |\n")
        
        tx_power = radar_station_cfg.get('tx_power_dbm', 20)
        f.write(f"| Tx Power | {tx_power} dBm |\n")
        
        sample_period_s = radar_station_cfg.get('sample_period_s', env_cfg['dt_s'])
        sample_period_min = sample_period_s / 60.0
        f.write(f"| Sample Period | {sample_period_min:.1f} min ({sample_period_s} s) |\n")
        
        duration_s = env_cfg['duration_s']
        duration_hr = duration_s / 3600.0
        f.write(f"| Run Duration | {duration_hr:.1f} hours ({duration_s} s) |\n")
        
        t_iso = data['t_iso']
        f.write(f"| Start Time | {t_iso[0]} |\n")
        f.write(f"| End Time | {t_iso[-1]} |\n")
        f.write("\n")
        
        # Reflector summary table
        f.write("## Reflector Summary\n\n")
        f.write("| Name | Role | Position (x, y, z) [m] | Range [m] |\n")
        f.write("|------|------|------------------------|----------|\n")
        
        for i in range(len(names)):
            xyz = pos_xyz[i]
            f.write(f"| {names[i]} | {roles[i]} | ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) | {ranges_m[i]:.1f} |\n")
        f.write("\n")
        
        # Key figures section
        f.write("## Key Figures\n\n")
        
        f.write("### 3D Scene Geometry\n\n")
        f.write("![3D Scene](scene_3d.png)\n\n")
        
        f.write("### Scene Before/After\n\n")
        f.write("![Before/After](scene_3d_before_after.png)\n\n")
        
        f.write("### HMI Station Dashboard\n\n")
        f.write("![HMI Dashboard](hmi_station.png)\n\n")
        
        f.write("### Alarm Timeline\n\n")
        f.write("![Alarm Timeline](alarm_timeline.png)\n\n")
        
        f.write("### Time-Series Grid\n\n")
        f.write("![Time-Series](timeseries_grid.png)\n\n")
        
        # Alarm log section
        f.write("## Alarm Log\n\n")
        
        f.write("### Alarm Code Definitions\n\n")
        f.write("- **REF_MOVE** (CRITICAL): Reference target displacement exceeds threshold\n")
        f.write("- **SLOPE_MOVE** (ALARM): Slope target movement rate exceeds threshold\n")
        f.write("- **HIGH_NOISE** (WARN): High phase noise detected (possible bad weather)\n")
        f.write("- **DROPOUT_HIGH** (WARN/ALARM): High data dropout rate\n")
        f.write("- **DRIFT_HIGH** (WARN): High system drift rate\n\n")
        
        f.write("### Alarm Summary\n\n")
        f.write(f"**Total Alarms:** {len(alarms)}\n\n")
        
        # Count by severity
        severity_counts = {}
        for alarm in alarms:
            severity_counts[alarm.severity] = severity_counts.get(alarm.severity, 0) + 1
        
        for severity in ['CRITICAL', 'ALARM', 'WARN', 'INFO']:
            count = severity_counts.get(severity, 0)
            f.write(f"- **{severity}:** {count}\n")
        f.write("\n")
        
        # Show first 15 alarms in markdown table
        if alarms:
            f.write("### Recent Alarms (first 15)\n\n")
            f.write("| Time | Severity | Code | Target | Message |\n")
            f.write("|------|----------|------|--------|----------|\n")
            
            for alarm in alarms[:15]:
                # Extract time only (HH:MM:SS)
                time_str = alarm.timestamp_iso.split('T')[1][:8] if 'T' in alarm.timestamp_iso else alarm.timestamp_iso[:8]
                f.write(f"| {time_str} | {alarm.severity} | {alarm.alarm_code} | {alarm.target} | {alarm.message} |\n")
            f.write("\n")
            
            if len(alarms) > 15:
                f.write(f"*...and {len(alarms) - 15} more alarms*\n\n")
        else:
            f.write("**No alarms generated during this run.**\n\n")
        
        f.write("**Full alarm log:** See [alarm_log.csv](alarm_log.csv)\n\n")
        
        # Notes section
        f.write("## Notes\n\n")
        
        operator_note = hmi_cfg.get('operator_note', '')
        if operator_note:
            f.write(f"**Operator Note:** {operator_note}\n\n")
        
        # Document injection scenarios
        if alarm_injection_cfg.get('enable', False):
            scenarios = alarm_injection_cfg.get('scenarios', [])
            if scenarios:
                f.write("### Alarm Injection Scenarios\n\n")
                f.write("This synthetic run includes forced alarm scenarios for validation:\n\n")
                
                for scenario in scenarios:
                    name = scenario.get('name', 'unknown')
                    t0_s = scenario.get('t0_s', 0)
                    t0_hr = t0_s / 3600.0
                    duration_s = scenario.get('duration_s', 0)
                    duration_hr = duration_s / 3600.0
                    
                    f.write(f"- **{name}**:\n")
                    f.write(f"  - Start: {t0_hr:.1f} hr ({t0_s:.0f} s)\n")
                    f.write(f"  - Duration: {duration_hr:.1f} hr ({duration_s:.0f} s)\n")
                    
                    if 'extra_phase_noise_sigma_rad' in scenario:
                        f.write(f"  - Effect: Extra phase noise (σ = {scenario['extra_phase_noise_sigma_rad']:.2f} rad)\n")
                    if 'forced_disp_mm' in scenario:
                        f.write(f"  - Effect: Forced displacement on {scenario.get('target', 'N/A')} ({scenario['forced_disp_mm']:.1f} mm)\n")
                    if 'extra_dropout_prob' in scenario:
                        f.write(f"  - Effect: Extra dropout probability ({scenario['extra_dropout_prob']*100:.0f}%)\n")
                    f.write("\n")
        
        f.write("---\n\n")
        f.write("*Generated by Smartslope 3D simulation pipeline*\n")
    
    print(f"Wrote {results_path}")
