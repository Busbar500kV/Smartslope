"""Alarm generation and timeline visualization for radar monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


@dataclass
class AlarmEvent:
    """Represents a single alarm event."""
    alarm_id: int
    timestamp_iso: str
    sample_idx: int
    severity: str  # INFO, WARN, ALARM, CRITICAL
    alarm_code: str
    target: str
    message: str
    value: float
    threshold: float
    window_start_idx: Optional[int] = None
    window_end_idx: Optional[int] = None


def detect_ref_move_alarms(
    data: Dict[str, np.ndarray],
    config: Dict,
    alarm_id_counter: int
) -> tuple[List[AlarmEvent], int]:
    """
    Detect reference target movement alarms.
    
    Triggers CRITICAL alarm when a reference target shows displacement exceeding threshold
    for persistence samples.
    """
    alarms = []
    alarm_cfg = config.get('alarms', {})
    
    if not alarm_cfg.get('enable', False):
        return alarms, alarm_id_counter
    
    threshold_mm = alarm_cfg.get('ref_move_mm_threshold', 2.0)
    persistence = alarm_cfg.get('ref_move_persistence_samples', 3)
    
    names = data['names']
    roles = data['roles']
    t_iso = data['t_iso']
    disp_los_m = data['disp_los_m']
    mask_valid = data['mask_valid']
    
    # Find reference targets
    ref_indices = np.where(roles == 'ref')[0]
    
    for idx in ref_indices:
        name = names[idx]
        disp_mm = np.abs(disp_los_m[idx]) * 1000.0
        valid = mask_valid[idx] == 1
        
        # Check for consecutive samples exceeding threshold
        exceeds = (disp_mm > threshold_mm) & valid
        
        # Find runs of consecutive True values
        for i in range(len(exceeds) - persistence + 1):
            window = exceeds[i:i+persistence]
            if np.all(window):
                # Alarm triggered at end of persistence window
                alarm_idx = i + persistence - 1
                
                # Check if we already raised alarm for this target recently
                # (avoid duplicate alarms for same event)
                if alarms and alarms[-1].target == name and alarms[-1].sample_idx >= i - persistence:
                    continue
                
                alarm = AlarmEvent(
                    alarm_id=alarm_id_counter,
                    timestamp_iso=str(t_iso[alarm_idx]),
                    sample_idx=alarm_idx,
                    severity='CRITICAL',
                    alarm_code='REF_MOVE',
                    target=name,
                    message=f'Reference {name} moved {disp_mm[alarm_idx]:.2f} mm (threshold: {threshold_mm:.2f} mm)',
                    value=disp_mm[alarm_idx],
                    threshold=threshold_mm,
                    window_start_idx=i,
                    window_end_idx=alarm_idx
                )
                alarms.append(alarm)
                alarm_id_counter += 1
                break  # One alarm per target per continuous exceedance
    
    return alarms, alarm_id_counter


def detect_slope_move_alarms(
    data: Dict[str, np.ndarray],
    config: Dict,
    alarm_id_counter: int
) -> tuple[List[AlarmEvent], int]:
    """
    Detect slope target movement alarms.
    
    Triggers ALARM when slope target rate exceeds threshold for persistence samples.
    """
    alarms = []
    alarm_cfg = config.get('alarms', {})
    
    if not alarm_cfg.get('enable', False):
        return alarms, alarm_id_counter
    
    rate_threshold_mm_per_hr = alarm_cfg.get('slope_rate_mm_per_hr_threshold', 10.0)
    persistence = alarm_cfg.get('slope_alarm_persistence_samples', 2)
    
    names = data['names']
    roles = data['roles']
    t_iso = data['t_iso']
    t_s = data['t_s']
    disp_los_m = data['disp_los_m']
    mask_valid = data['mask_valid']
    
    # Find slope targets
    slope_indices = np.where(roles == 'slope')[0]
    
    # Compute sample rate
    if len(t_s) > 1:
        dt_s = t_s[1] - t_s[0]
        dt_hr = dt_s / 3600.0
    else:
        return alarms, alarm_id_counter
    
    for idx in slope_indices:
        name = names[idx]
        disp_mm = disp_los_m[idx] * 1000.0
        valid = mask_valid[idx] == 1
        
        # Compute rate (simple finite difference)
        rate_mm_per_hr = np.zeros(len(disp_mm))
        rate_mm_per_hr[1:] = np.diff(disp_mm) / dt_hr
        rate_mm_per_hr[0] = 0.0
        
        # Check for consecutive samples exceeding rate threshold
        exceeds = (np.abs(rate_mm_per_hr) > rate_threshold_mm_per_hr) & valid
        
        # Find runs of consecutive True values
        for i in range(len(exceeds) - persistence + 1):
            window = exceeds[i:i+persistence]
            if np.all(window):
                alarm_idx = i + persistence - 1
                
                # Avoid duplicates
                if alarms and alarms[-1].target == name and alarms[-1].sample_idx >= i - persistence:
                    continue
                
                alarm = AlarmEvent(
                    alarm_id=alarm_id_counter,
                    timestamp_iso=str(t_iso[alarm_idx]),
                    sample_idx=alarm_idx,
                    severity='ALARM',
                    alarm_code='SLOPE_MOVE',
                    target=name,
                    message=f'Slope {name} moving at {rate_mm_per_hr[alarm_idx]:.2f} mm/hr (threshold: {rate_threshold_mm_per_hr:.2f} mm/hr)',
                    value=np.abs(rate_mm_per_hr[alarm_idx]),
                    threshold=rate_threshold_mm_per_hr,
                    window_start_idx=i,
                    window_end_idx=alarm_idx
                )
                alarms.append(alarm)
                alarm_id_counter += 1
                break
    
    return alarms, alarm_id_counter


def detect_high_noise_alarms(
    data: Dict[str, np.ndarray],
    config: Dict,
    alarm_id_counter: int
) -> tuple[List[AlarmEvent], int]:
    """
    Detect high phase noise alarms.
    
    Computes rolling residual phase noise and triggers WARN if median exceeds threshold.
    """
    alarms = []
    alarm_cfg = config.get('alarms', {})
    
    if not alarm_cfg.get('enable', False):
        return alarms, alarm_id_counter
    
    threshold_rad = alarm_cfg.get('high_noise_phase_sigma_rad_threshold', 0.6)
    window_samples = alarm_cfg.get('high_noise_window_samples', 6)
    
    t_iso = data['t_iso']
    phi_unwrapped = data['phi_unwrapped']
    mask_valid = data['mask_valid']
    
    n_reflectors, n_samples = phi_unwrapped.shape
    
    # Compute residual noise per reflector (phase - moving average)
    noise_per_target = np.zeros((n_reflectors, n_samples))
    
    for i in range(n_reflectors):
        phi = phi_unwrapped[i]
        valid = mask_valid[i] == 1
        
        if valid.sum() < window_samples:
            continue
        
        # Compute moving average (simple box filter)
        phi_ma = np.full_like(phi, np.nan)
        for t in range(window_samples, n_samples):
            window = phi[t-window_samples:t]
            window_valid = valid[t-window_samples:t]
            if window_valid.sum() >= window_samples // 2:
                phi_ma[t] = np.nanmean(window[window_valid])
        
        # Residual
        residual = phi - phi_ma
        
        # Std of residual over rolling window
        for t in range(window_samples, n_samples):
            window_res = residual[t-window_samples:t]
            window_valid = valid[t-window_samples:t]
            if window_valid.sum() >= window_samples // 2:
                noise_per_target[i, t] = np.nanstd(window_res[window_valid])
    
    # Compute median noise across all targets
    for t in range(window_samples, n_samples):
        noise_vals = noise_per_target[:, t]
        noise_vals = noise_vals[noise_vals > 0]  # Filter zeros
        if len(noise_vals) < 1:
            continue
        
        median_noise = np.median(noise_vals)
        
        if median_noise > threshold_rad:
            # Check if we already raised alarm recently
            if alarms and alarms[-1].alarm_code == 'HIGH_NOISE' and alarms[-1].sample_idx >= t - window_samples:
                continue
            
            alarm = AlarmEvent(
                alarm_id=alarm_id_counter,
                timestamp_iso=str(t_iso[t]),
                sample_idx=t,
                severity='WARN',
                alarm_code='HIGH_NOISE',
                target='SYSTEM',
                message=f'High phase noise detected: {median_noise:.3f} rad (threshold: {threshold_rad:.3f} rad)',
                value=median_noise,
                threshold=threshold_rad,
                window_start_idx=t - window_samples,
                window_end_idx=t
            )
            alarms.append(alarm)
            alarm_id_counter += 1
    
    return alarms, alarm_id_counter


def detect_dropout_alarms(
    data: Dict[str, np.ndarray],
    config: Dict,
    alarm_id_counter: int
) -> tuple[List[AlarmEvent], int]:
    """
    Detect high dropout rate alarms.
    
    Triggers WARN/ALARM when fraction of invalid samples in rolling window exceeds threshold.
    """
    alarms = []
    alarm_cfg = config.get('alarms', {})
    
    if not alarm_cfg.get('enable', False):
        return alarms, alarm_id_counter
    
    threshold_rate = alarm_cfg.get('dropout_rate_threshold', 0.25)
    window_samples = alarm_cfg.get('dropout_window_samples', 6)
    
    t_iso = data['t_iso']
    mask_valid = data['mask_valid']
    
    n_reflectors, n_samples = mask_valid.shape
    
    # Compute dropout rate across all reflectors in rolling window
    for t in range(window_samples, n_samples):
        window = mask_valid[:, t-window_samples:t]
        total_samples = window.size
        invalid_samples = np.sum(window == 0)
        dropout_rate = invalid_samples / total_samples
        
        if dropout_rate > threshold_rate:
            # Check if we already raised alarm recently
            if alarms and alarms[-1].alarm_code == 'DROPOUT_HIGH' and alarms[-1].sample_idx >= t - window_samples:
                continue
            
            severity = 'ALARM' if dropout_rate > 0.5 else 'WARN'
            
            alarm = AlarmEvent(
                alarm_id=alarm_id_counter,
                timestamp_iso=str(t_iso[t]),
                sample_idx=t,
                severity=severity,
                alarm_code='DROPOUT_HIGH',
                target='SYSTEM',
                message=f'High dropout rate: {dropout_rate*100:.1f}% (threshold: {threshold_rate*100:.1f}%)',
                value=dropout_rate,
                threshold=threshold_rate,
                window_start_idx=t - window_samples,
                window_end_idx=t
            )
            alarms.append(alarm)
            alarm_id_counter += 1
    
    return alarms, alarm_id_counter


def detect_drift_alarms(
    data: Dict[str, np.ndarray],
    config: Dict,
    alarm_id_counter: int
) -> tuple[List[AlarmEvent], int]:
    """
    Detect high drift rate alarms.
    
    Triggers WARN when drift rate exceeds threshold over rolling window.
    """
    alarms = []
    alarm_cfg = config.get('alarms', {})
    
    if not alarm_cfg.get('enable', False):
        return alarms, alarm_id_counter
    
    threshold_rad_per_hr = alarm_cfg.get('drift_rate_rad_per_hr_threshold', 2.0)
    window_samples = alarm_cfg.get('drift_window_samples', 6)
    
    t_iso = data['t_iso']
    t_s = data['t_s']
    drift_rad = data['drift_rad']
    
    if len(t_s) < 2:
        return alarms, alarm_id_counter
    
    dt_s = t_s[1] - t_s[0]
    dt_hr = dt_s / 3600.0
    
    # Compute drift rate
    drift_rate = np.zeros(len(drift_rad))
    drift_rate[1:] = np.diff(drift_rad) / dt_hr
    
    # Check rolling window
    for t in range(window_samples, len(drift_rad)):
        window_rate = drift_rate[t-window_samples:t]
        avg_rate = np.mean(np.abs(window_rate))
        
        if avg_rate > threshold_rad_per_hr:
            # Check if we already raised alarm recently
            if alarms and alarms[-1].alarm_code == 'DRIFT_HIGH' and alarms[-1].sample_idx >= t - window_samples:
                continue
            
            alarm = AlarmEvent(
                alarm_id=alarm_id_counter,
                timestamp_iso=str(t_iso[t]),
                sample_idx=t,
                severity='WARN',
                alarm_code='DRIFT_HIGH',
                target='SYSTEM',
                message=f'High drift rate: {avg_rate:.3f} rad/hr (threshold: {threshold_rad_per_hr:.3f} rad/hr)',
                value=avg_rate,
                threshold=threshold_rad_per_hr,
                window_start_idx=t - window_samples,
                window_end_idx=t
            )
            alarms.append(alarm)
            alarm_id_counter += 1
    
    return alarms, alarm_id_counter


def generate_alarms(data: Dict[str, np.ndarray], config: Dict) -> List[AlarmEvent]:
    """
    Generate all alarms from synthetic data.
    
    Args:
        data: Simulation output data dictionary
        config: Configuration dictionary with alarm thresholds
    
    Returns:
        List of AlarmEvent objects, sorted by timestamp
    """
    all_alarms = []
    alarm_id = 1
    
    # Run each detector
    alarms, alarm_id = detect_ref_move_alarms(data, config, alarm_id)
    all_alarms.extend(alarms)
    
    alarms, alarm_id = detect_slope_move_alarms(data, config, alarm_id)
    all_alarms.extend(alarms)
    
    alarms, alarm_id = detect_high_noise_alarms(data, config, alarm_id)
    all_alarms.extend(alarms)
    
    alarms, alarm_id = detect_dropout_alarms(data, config, alarm_id)
    all_alarms.extend(alarms)
    
    alarms, alarm_id = detect_drift_alarms(data, config, alarm_id)
    all_alarms.extend(alarms)
    
    # Sort by sample index (time)
    all_alarms.sort(key=lambda a: a.sample_idx)
    
    return all_alarms


def write_alarm_log_csv(alarms: List[AlarmEvent], output_path: Path) -> None:
    """
    Write alarm log to CSV file.
    
    Args:
        alarms: List of AlarmEvent objects
        output_path: Output CSV file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Header
        f.write('alarm_id,timestamp_iso,sample_idx,severity,alarm_code,target,message,value,threshold\n')
        
        # Rows
        for alarm in alarms:
            f.write(f'{alarm.alarm_id},{alarm.timestamp_iso},{alarm.sample_idx},{alarm.severity},'
                   f'{alarm.alarm_code},{alarm.target},"{alarm.message}",{alarm.value:.6f},{alarm.threshold:.6f}\n')


def plot_alarm_timeline(
    data: Dict[str, np.ndarray],
    config: Dict,
    alarms: List[AlarmEvent],
    output_path: Path
) -> None:
    """
    Plot alarm timeline with key signals and alarm markers.
    
    Args:
        data: Simulation output data
        config: Configuration dictionary
        alarms: List of AlarmEvent objects
        output_path: Output PNG file path
    """
    t_s = data['t_s']
    t_hr = t_s / 3600.0
    names = data['names']
    roles = data['roles']
    disp_los_m = data['disp_los_m']
    mask_valid = data['mask_valid']
    
    # Find one slope target for plotting
    slope_indices = np.where(roles == 'slope')[0]
    if len(slope_indices) == 0:
        print("Warning: No slope targets found for alarm timeline plot")
        return
    
    slope_idx = slope_indices[0]
    slope_name = names[slope_idx]
    slope_disp_mm = disp_los_m[slope_idx] * 1000.0
    
    # Compute system noise metric (median phase std)
    phi_unwrapped = data['phi_unwrapped']
    noise_metric = np.zeros(len(t_s))
    window = 6
    for t in range(window, len(t_s)):
        noise_vals = []
        for i in range(len(names)):
            window_phi = phi_unwrapped[i, t-window:t]
            window_valid = mask_valid[i, t-window:t] == 1
            if window_valid.sum() >= window // 2:
                noise_vals.append(np.nanstd(window_phi[window_valid]))
        if noise_vals:
            noise_metric[t] = np.median(noise_vals)
    
    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot 1: Slope displacement
    ax1 = axes[0]
    ax1.plot(t_hr, slope_disp_mm, 'b-', linewidth=1.5, label=f'{slope_name} LOS Disp')
    ax1.set_ylabel('Displacement (mm)', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    
    # Plot 2: Noise metric
    ax2 = axes[1]
    ax2.plot(t_hr, noise_metric, 'orange', linewidth=1.5, label='System Noise (phase σ)')
    ax2.set_ylabel('Phase Noise (rad)', fontsize=10)
    ax2.set_xlabel('Time (hours)', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)
    
    # Overlay alarm markers
    severity_colors = {
        'INFO': 'green',
        'WARN': 'yellow',
        'ALARM': 'orange',
        'CRITICAL': 'red'
    }
    
    for alarm in alarms:
        t_alarm_hr = t_s[alarm.sample_idx] / 3600.0
        color = severity_colors.get(alarm.severity, 'gray')
        
        # Mark on both axes
        for ax in axes:
            ax.axvline(t_alarm_hr, color=color, alpha=0.5, linewidth=1.5, linestyle='--')
        
        # Add label on top axis
        y_pos = ax1.get_ylim()[1] * 0.95
        ax1.text(t_alarm_hr, y_pos, alarm.alarm_code, rotation=90, va='top', ha='right',
                fontsize=7, color=color, alpha=0.8)
    
    # Add legend for alarm severities
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=s) for s, c in severity_colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', ncol=4, fontsize=8, title='Alarm Severity')
    
    plt.suptitle('Alarm Timeline', fontsize=14, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
