"""SCADA-friendly telemetry export for integration testing."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np


def compute_system_status(
    alarms_at_time: List,
    critical_count: int,
    alarm_count: int
) -> str:
    """
    Compute overall system status based on active alarms.
    
    Args:
        alarms_at_time: List of alarms active at this time
        critical_count: Count of CRITICAL severity alarms
        alarm_count: Count of ALARM severity alarms
    
    Returns:
        Status string: "CRITICAL", "ALARM", "WARN", or "OK"
    """
    if critical_count > 0:
        return "CRITICAL"
    elif alarm_count > 0:
        return "ALARM"
    elif any(a.severity == "WARN" for a in alarms_at_time):
        return "WARN"
    else:
        return "OK"


def write_scada_telemetry(
    data: Dict[str, np.ndarray],
    config: Dict,
    alarms: List,
    run_id: str,
    output_path: Path
) -> None:
    """
    Write SCADA-friendly telemetry CSV file with time-series data.
    
    Args:
        data: Simulation output data
        config: Configuration dictionary
        alarms: List of AlarmEvent objects
        run_id: Run identifier
        output_path: Output CSV file path
    """
    # Extract data
    t_iso = data['t_iso']
    t_s = data['t_s']
    n_samples = len(t_s)
    
    names = data['names']
    roles = data['roles']
    mask_valid = data['mask_valid']
    phi_unwrapped = data['phi_unwrapped']
    disp_los_m = data['disp_los_m']
    
    # Extract config
    radar_cfg = config.get('radar', {})
    radar_station_cfg = config.get('radar_station', {})
    alarm_cfg = config.get('alarms', {})
    
    freq_hz = radar_station_cfg.get('frequency_hz', radar_cfg.get('frequency_hz', 24e9))
    freq_ghz = freq_hz / 1e9
    tx_power_dbm = radar_station_cfg.get('tx_power_dbm', 20)
    
    # Get alarm thresholds for reference
    high_noise_threshold = alarm_cfg.get('high_noise_phase_sigma_rad_threshold', 0.6)
    dropout_threshold = alarm_cfg.get('dropout_rate_threshold', 0.25)
    noise_window = alarm_cfg.get('high_noise_window_samples', 6)
    dropout_window = alarm_cfg.get('dropout_window_samples', 6)
    
    # Find ref and slope indices
    ref_indices = np.where(roles == 'ref')[0]
    slope_indices = np.where(roles == 'slope')[0]
    
    # Pre-compute metrics per timestep
    noise_metrics = np.zeros(n_samples)
    dropout_fractions = np.zeros(n_samples)
    ref_move_metrics = np.zeros(n_samples)
    slope_event_metrics = np.zeros(n_samples)
    
    # Compute rolling system noise metric (median phase std)
    for t in range(noise_window, n_samples):
        noise_vals = []
        for i in range(len(names)):
            window_phi = phi_unwrapped[i, t-noise_window:t]
            window_valid = mask_valid[i, t-noise_window:t] == 1
            if window_valid.sum() >= noise_window // 2:
                noise_vals.append(np.nanstd(window_phi[window_valid]))
        if noise_vals:
            noise_metrics[t] = np.median(noise_vals)
    
    # Compute rolling dropout fraction
    for t in range(dropout_window, n_samples):
        window = mask_valid[:, t-dropout_window:t]
        total_samples = window.size
        invalid_samples = np.sum(window == 0)
        dropout_fractions[t] = invalid_samples / total_samples
    
    # Compute ref move metric (max abs displacement in window)
    if len(ref_indices) > 0:
        for t in range(n_samples):
            ref_disps_mm = np.abs(disp_los_m[ref_indices, t]) * 1000.0
            ref_move_metrics[t] = np.max(ref_disps_mm)
    
    # Compute slope event metric (max rate)
    if len(slope_indices) > 0 and len(t_s) > 1:
        dt_s = t_s[1] - t_s[0]
        dt_hr = dt_s / 3600.0
        
        for slope_idx in slope_indices:
            disp_mm = disp_los_m[slope_idx] * 1000.0
            rate_mm_per_hr = np.zeros(len(disp_mm))
            rate_mm_per_hr[1:] = np.abs(np.diff(disp_mm)) / dt_hr
            slope_event_metrics = np.maximum(slope_event_metrics, rate_mm_per_hr)
    
    # Build alarm lookup by sample index for efficient querying
    alarms_by_idx = {}
    for alarm in alarms:
        idx = alarm.sample_idx
        if idx not in alarms_by_idx:
            alarms_by_idx[idx] = []
        alarms_by_idx[idx].append(alarm)
    
    # Track active alarms (simple model: alarm active from sample_idx until end or cleared)
    # For simplicity, we'll just count alarms that occurred at or before this time
    # and have not been explicitly cleared (we don't have clear events in current model)
    active_alarms_history = []
    
    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'timestamp_iso',
            'run_id',
            'radar_frequency_ghz',
            'tx_power_dbm',
            'system_noise_metric',
            'system_dropout_fraction',
            'ref_move_metric_mm',
            'slope_event_metric_mm_per_hr',
            'alarm_counts_active',
            'alarm_counts_latched',
            'alarm_counts_critical_active',
            'system_status'
        ])
        
        # Rows
        for t in range(n_samples):
            timestamp = str(t_iso[t])
            
            # Get alarms that occurred up to this point
            alarms_up_to_now = [a for a in alarms if a.sample_idx <= t]
            
            # Count active alarms (simple model: all alarms up to now are "active")
            # In reality, some may have cleared, but we don't track that yet
            active_count = len(alarms_up_to_now)
            
            # Count by severity
            critical_count = sum(1 for a in alarms_up_to_now if a.severity == 'CRITICAL')
            alarm_sev_count = sum(1 for a in alarms_up_to_now if a.severity == 'ALARM')
            
            # Latched count (placeholder - will be computed by alarm state machine)
            latched_count = 0
            
            # System status
            status = compute_system_status(alarms_up_to_now, critical_count, alarm_sev_count)
            
            writer.writerow([
                timestamp,
                run_id,
                f'{freq_ghz:.6f}',
                f'{tx_power_dbm}',
                f'{noise_metrics[t]:.6f}',
                f'{dropout_fractions[t]:.6f}',
                f'{ref_move_metrics[t]:.6f}',
                f'{slope_event_metrics[t]:.6f}',
                f'{active_count}',
                f'{latched_count}',
                f'{critical_count}',
                status
            ])
    
    print(f"Wrote {output_path}")
