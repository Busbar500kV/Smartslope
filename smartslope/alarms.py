"""Alarm generation and timeline visualization for radar monitoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json

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
    state: str = 'ACTIVE'  # ACTIVE, LATCHED, ACKED, CLEAR
    latched: bool = False
    acked: bool = False
    details: Optional[Dict] = field(default_factory=dict)


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


def detect_slope_consensus_alarms(
    data: Dict[str, np.ndarray],
    config: Dict,
    slope_alarms: List[AlarmEvent],
    alarm_id_counter: int
) -> tuple[List[AlarmEvent], int]:
    """
    Detect K-of-N slope consensus alarms (SLOPE_EVENT_SYSTEM).
    
    Triggers when K or more slope targets have SLOPE_MOVE alarms within a rolling window.
    
    Args:
        data: Simulation output data
        config: Configuration dictionary
        slope_alarms: List of SLOPE_MOVE alarms already detected
        alarm_id_counter: Current alarm ID counter
    
    Returns:
        (alarms, updated_alarm_id_counter)
    """
    alarms = []
    alarm_cfg = config.get('alarms', {})
    
    if not alarm_cfg.get('enable', False):
        return alarms, alarm_id_counter
    
    k_of_n = alarm_cfg.get('slope_k_of_n_consensus', 2)
    window_samples = alarm_cfg.get('slope_consensus_window_samples', 10)
    
    if k_of_n <= 1 or not slope_alarms:
        return alarms, alarm_id_counter
    
    t_iso = data['t_iso']
    n_samples = len(t_iso)
    
    # Group slope alarms by sample index
    alarms_by_idx = {}
    for alarm in slope_alarms:
        idx = alarm.sample_idx
        if idx not in alarms_by_idx:
            alarms_by_idx[idx] = []
        alarms_by_idx[idx].append(alarm)
    
    # Check rolling window for K-of-N consensus
    for t in range(window_samples, n_samples):
        # Count unique slope targets with alarms in window
        targets_with_alarms = set()
        contributing_alarms = []
        
        for idx in range(t - window_samples, t + 1):
            if idx in alarms_by_idx:
                for alarm in alarms_by_idx[idx]:
                    targets_with_alarms.add(alarm.target)
                    contributing_alarms.append(alarm)
        
        if len(targets_with_alarms) >= k_of_n:
            # Check if we already raised consensus alarm recently
            if alarms and alarms[-1].sample_idx >= t - window_samples:
                continue
            
            target_list = ', '.join(sorted(targets_with_alarms))
            
            alarm = AlarmEvent(
                alarm_id=alarm_id_counter,
                timestamp_iso=str(t_iso[t]),
                sample_idx=t,
                severity='CRITICAL',
                alarm_code='SLOPE_EVENT_SYSTEM',
                target='SYSTEM',
                message=f'Slope consensus: {len(targets_with_alarms)} targets moving ({target_list})',
                value=float(len(targets_with_alarms)),
                threshold=float(k_of_n),
                window_start_idx=t - window_samples,
                window_end_idx=t,
                details={'contributing_targets': list(targets_with_alarms)}
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
    
    slope_alarms, alarm_id = detect_slope_move_alarms(data, config, alarm_id)
    all_alarms.extend(slope_alarms)
    
    # Slope consensus (K-of-N)
    alarms, alarm_id = detect_slope_consensus_alarms(data, config, slope_alarms, alarm_id)
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
        f.write('alarm_id,timestamp_iso,sample_idx,severity,alarm_code,target,message,value,threshold,state,latched,acked\n')
        
        # Rows
        for alarm in alarms:
            f.write(f'{alarm.alarm_id},{alarm.timestamp_iso},{alarm.sample_idx},{alarm.severity},'
                   f'{alarm.alarm_code},{alarm.target},"{alarm.message}",{alarm.value:.6f},{alarm.threshold:.6f},'
                   f'{alarm.state},{1 if alarm.latched else 0},{1 if alarm.acked else 0}\n')


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


def load_acknowledgments(ack_path: Path) -> List[Dict]:
    """
    Load acknowledgments from JSON file if it exists.
    
    Args:
        ack_path: Path to acknowledgment JSON file
    
    Returns:
        List of acknowledgment dicts, or empty list if file doesn't exist
    """
    if not ack_path.exists():
        return []
    
    try:
        with open(ack_path, 'r') as f:
            acks = json.load(f)
            return acks if isinstance(acks, list) else []
    except (json.JSONDecodeError, IOError):
        print(f"Warning: Could not read acknowledgment file {ack_path}")
        return []


def generate_ack_template(alarms: List[AlarmEvent], output_path: Path) -> None:
    """
    Generate acknowledgment template JSON file with instructions.
    
    Args:
        alarms: List of AlarmEvent objects
        output_path: Output path for template file
    """
    # Find alarms that require acknowledgment (ALARM or CRITICAL)
    ack_required = [a for a in alarms if a.severity in ['ALARM', 'CRITICAL']]
    
    template = {
        "_instructions": {
            "description": "Use this file to acknowledge alarms. Copy this template to 'alarm_ack.json' and fill in acknowledgments.",
            "format": "Each acknowledgment should have: alarm_id (or alarm_code+target), ack_time_iso, ack_by, note",
            "example": {
                "alarm_id": 5,
                "ack_time_iso": "2026-01-18T12:00:00-08:00",
                "ack_by": "operator_name",
                "note": "Investigated and confirmed safe"
            }
        },
        "acknowledgments": []
    }
    
    # Add template entries for each alarm requiring ack
    for alarm in ack_required[:5]:  # Limit to first 5 for template
        template["acknowledgments"].append({
            "alarm_id": alarm.alarm_id,
            "alarm_code": alarm.alarm_code,
            "target": alarm.target,
            "ack_time_iso": "",
            "ack_by": "",
            "note": ""
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Generated acknowledgment template: {output_path}")


def apply_alarm_latching_and_acks(
    alarms: List[AlarmEvent],
    config: Dict,
    data: Dict[str, np.ndarray],
    ack_file: Optional[Path] = None
) -> tuple[List[AlarmEvent], Dict]:
    """
    Apply alarm latching logic and acknowledgments to alarms.
    
    Latching behavior:
    - Alarms with severity in latch_severities will remain active until acknowledged
    - State transitions: CLEAR -> ACTIVE -> LATCHED -> ACKED
    
    Args:
        alarms: List of AlarmEvent objects
        config: Configuration dictionary
        data: Simulation data (for timestamps)
        ack_file: Optional path to acknowledgment file
    
    Returns:
        (updated_alarms, alarm_state_summary)
    """
    alarm_ops_cfg = config.get('alarm_ops', {})
    latch_severities = alarm_ops_cfg.get('latch_severities', ['ALARM', 'CRITICAL'])
    escalation_after_minutes = alarm_ops_cfg.get('escalation_after_minutes', 30)
    
    # Load acknowledgments
    acks = []
    if ack_file and ack_file.exists():
        acks = load_acknowledgments(ack_file)
    
    # Build ack lookup by alarm_id
    acks_by_id = {ack.get('alarm_id'): ack for ack in acks if 'alarm_id' in ack}
    
    # Build ack lookup by (alarm_code, target)
    acks_by_code_target = {}
    for ack in acks:
        if 'alarm_code' in ack and 'target' in ack:
            key = (ack['alarm_code'], ack['target'])
            acks_by_code_target[key] = ack
    
    # Process alarms
    updated_alarms = []
    active_alarms = []
    latched_alarms = []
    acked_alarms = []
    escalated_alarms = []
    
    t_iso = data['t_iso']
    t_s = data['t_s']
    
    for alarm in alarms:
        # Check if alarm should be latched
        should_latch = alarm.severity in latch_severities
        
        # Check if acknowledged
        ack = acks_by_id.get(alarm.alarm_id)
        if not ack:
            key = (alarm.alarm_code, alarm.target)
            ack = acks_by_code_target.get(key)
        
        if ack:
            alarm.acked = True
            alarm.state = 'ACKED'
            acked_alarms.append(alarm)
        elif should_latch:
            alarm.latched = True
            alarm.state = 'LATCHED'
            latched_alarms.append(alarm)
        else:
            alarm.state = 'ACTIVE'
            active_alarms.append(alarm)
        
        updated_alarms.append(alarm)
    
    # Check for escalation (unacknowledged CRITICAL alarms)
    if escalation_after_minutes > 0:
        for alarm in latched_alarms:
            if alarm.severity == 'CRITICAL':
                # Check time since alarm
                alarm_time_s = t_s[alarm.sample_idx]
                last_time_s = t_s[-1]
                elapsed_minutes = (last_time_s - alarm_time_s) / 60.0
                
                if elapsed_minutes > escalation_after_minutes:
                    escalated_alarms.append(alarm)
    
    # Build state summary
    alarm_state = {
        'timestamp': str(t_iso[-1]),
        'total_alarms': len(alarms),
        'active_alarms': [
            {
                'alarm_id': a.alarm_id,
                'severity': a.severity,
                'alarm_code': a.alarm_code,
                'target': a.target,
                'timestamp_iso': a.timestamp_iso
            }
            for a in active_alarms
        ],
        'latched_alarms': [
            {
                'alarm_id': a.alarm_id,
                'severity': a.severity,
                'alarm_code': a.alarm_code,
                'target': a.target,
                'timestamp_iso': a.timestamp_iso
            }
            for a in latched_alarms
        ],
        'acked_alarms': [
            {
                'alarm_id': a.alarm_id,
                'severity': a.severity,
                'alarm_code': a.alarm_code,
                'target': a.target,
                'timestamp_iso': a.timestamp_iso
            }
            for a in acked_alarms
        ],
        'escalation': {
            'enabled': escalation_after_minutes > 0,
            'threshold_minutes': escalation_after_minutes,
            'escalated_count': len(escalated_alarms),
            'escalated_alarms': [
                {
                    'alarm_id': a.alarm_id,
                    'alarm_code': a.alarm_code,
                    'target': a.target
                }
                for a in escalated_alarms
            ]
        },
        'counts': {
            'active': len(active_alarms),
            'latched': len(latched_alarms),
            'acked': len(acked_alarms),
            'critical_unacked': len([a for a in latched_alarms if a.severity == 'CRITICAL']),
            'alarm_unacked': len([a for a in latched_alarms if a.severity == 'ALARM'])
        }
    }
    
    return updated_alarms, alarm_state


def write_alarm_state_json(alarm_state: Dict, output_path: Path) -> None:
    """
    Write alarm state to JSON file.
    
    Args:
        alarm_state: Alarm state dictionary
        output_path: Output JSON file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(alarm_state, f, indent=2)
    
    print(f"Wrote {output_path}")
