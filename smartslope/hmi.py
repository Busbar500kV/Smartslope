"""HMI dashboard rendering for radar station monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


def render_hmi_dashboard(
    data: Dict[str, np.ndarray],
    config: Dict,
    alarms: List,
    run_id: str,
    output_path: Path,
    geometry_metrics: Dict = None,
    alarm_state: Dict = None
) -> None:
    """
    Render HMI-style station dashboard image.
    
    Args:
        data: Simulation output data
        config: Configuration dictionary
        alarms: List of AlarmEvent objects
        run_id: Run identifier
        output_path: Output PNG file path
        geometry_metrics: Optional geometry metrics dictionary
        alarm_state: Optional alarm state dictionary
    """
    # Extract config
    hmi_cfg = config.get('hmi', {})
    radar_station_cfg = config.get('radar_station', {})
    radar_cfg = config['radar']
    
    station_name = hmi_cfg.get('station_name', 'Unknown Station')
    site_id = hmi_cfg.get('site_id', 'N/A')
    
    # Time range
    t_iso = data['t_iso']
    start_time = str(t_iso[0])
    end_time = str(t_iso[-1])
    
    # Radar parameters
    freq_hz = radar_station_cfg.get('frequency_hz', radar_cfg.get('frequency_hz', 24e9))
    freq_ghz = freq_hz / 1e9
    wavelength_m = float(data['wavelength_m'][0])
    wavelength_mm = wavelength_m * 1000.0
    tx_power_dbm = radar_station_cfg.get('tx_power_dbm', 20)
    sample_period_s = radar_station_cfg.get('sample_period_s', 600)
    sample_period_min = sample_period_s / 60.0
    
    # Compute status
    names = data['names']
    roles = data['roles']
    
    ref_indices = np.where(roles == 'ref')[0]
    slope_indices = np.where(roles == 'slope')[0]
    
    # Determine status from recent alarms (last 10% of timeline)
    recent_threshold_idx = int(len(t_iso) * 0.9)
    recent_alarms = [a for a in alarms if a.sample_idx >= recent_threshold_idx]
    
    ref_status = 'OK'
    slope_status = 'OK'
    weather_status = 'OK'
    comms_status = 'OK'
    
    for alarm in recent_alarms:
        if alarm.alarm_code == 'REF_MOVE':
            ref_status = 'ALARM'
        elif alarm.alarm_code == 'SLOPE_MOVE':
            slope_status = 'ALARM'
        elif alarm.alarm_code == 'HIGH_NOISE':
            weather_status = 'WARN'
        elif alarm.alarm_code == 'DROPOUT_HIGH':
            comms_status = 'ALARM' if alarm.severity == 'ALARM' else 'WARN'
    
    # Create figure with more rows to accommodate new tiles
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#1a1a1a')  # Dark background for HMI look
    
    gs = GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Header section
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    ax_header.text(0.5, 0.7, station_name, ha='center', va='center',
                  fontsize=24, weight='bold', color='white')
    ax_header.text(0.5, 0.3, f'Site ID: {site_id}  |  Run: {run_id}', ha='center', va='center',
                  fontsize=12, color='lightgray')
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)
    
    # Radar parameters section
    ax_params = fig.add_subplot(gs[1, :])
    ax_params.axis('off')
    ax_params.set_xlim(0, 1)
    ax_params.set_ylim(0, 1)
    
    param_text = (
        f'Frequency: {freq_ghz:.2f} GHz  |  Wavelength: {wavelength_mm:.2f} mm  |  '
        f'Tx Power: {tx_power_dbm} dBm  |  Sample Period: {sample_period_min:.1f} min\n'
        f'Start: {start_time[:19]}  |  End: {end_time[:19]}'
    )
    ax_params.text(0.5, 0.5, param_text, ha='center', va='center',
                  fontsize=10, color='lightgray', family='monospace')
    
    # Status tiles
    status_colors = {'OK': 'green', 'WARN': 'yellow', 'ALARM': 'red'}
    
    # REF Status
    ax_ref = fig.add_subplot(gs[2, 0])
    ax_ref.set_facecolor('#2a2a2a')
    ax_ref.axis('off')
    ax_ref.set_xlim(0, 1)
    ax_ref.set_ylim(0, 1)
    ref_color = status_colors[ref_status]
    rect = mpatches.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=3, edgecolor=ref_color, facecolor='none')
    ax_ref.add_patch(rect)
    ax_ref.text(0.5, 0.7, 'REFERENCE', ha='center', va='center', fontsize=14, weight='bold', color='white')
    ax_ref.text(0.5, 0.5, f'{len(ref_indices)} targets', ha='center', va='center', fontsize=10, color='lightgray')
    ax_ref.text(0.5, 0.3, ref_status, ha='center', va='center', fontsize=16, weight='bold', color=ref_color)
    
    # SLOPE Status
    ax_slope = fig.add_subplot(gs[2, 1])
    ax_slope.set_facecolor('#2a2a2a')
    ax_slope.axis('off')
    ax_slope.set_xlim(0, 1)
    ax_slope.set_ylim(0, 1)
    slope_color = status_colors[slope_status]
    rect = mpatches.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=3, edgecolor=slope_color, facecolor='none')
    ax_slope.add_patch(rect)
    ax_slope.text(0.5, 0.7, 'SLOPE', ha='center', va='center', fontsize=14, weight='bold', color='white')
    ax_slope.text(0.5, 0.5, f'{len(slope_indices)} targets', ha='center', va='center', fontsize=10, color='lightgray')
    ax_slope.text(0.5, 0.3, slope_status, ha='center', va='center', fontsize=16, weight='bold', color=slope_color)
    
    # WEATHER Status
    ax_weather = fig.add_subplot(gs[2, 2])
    ax_weather.set_facecolor('#2a2a2a')
    ax_weather.axis('off')
    ax_weather.set_xlim(0, 1)
    ax_weather.set_ylim(0, 1)
    weather_color = status_colors[weather_status]
    rect = mpatches.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=3, edgecolor=weather_color, facecolor='none')
    ax_weather.add_patch(rect)
    ax_weather.text(0.5, 0.7, 'WEATHER/NOISE', ha='center', va='center', fontsize=14, weight='bold', color='white')
    ax_weather.text(0.5, 0.3, weather_status, ha='center', va='center', fontsize=16, weight='bold', color=weather_color)
    
    # COMMS Status
    ax_comms = fig.add_subplot(gs[3, 0])
    ax_comms.set_facecolor('#2a2a2a')
    ax_comms.axis('off')
    ax_comms.set_xlim(0, 1)
    ax_comms.set_ylim(0, 1)
    comms_color = status_colors[comms_status]
    rect = mpatches.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=3, edgecolor=comms_color, facecolor='none')
    ax_comms.add_patch(rect)
    ax_comms.text(0.5, 0.7, 'COMMS/DATA', ha='center', va='center', fontsize=14, weight='bold', color='white')
    ax_comms.text(0.5, 0.3, comms_status, ha='center', va='center', fontsize=16, weight='bold', color=comms_color)
    
    # Geometry Sensitivity Tile
    ax_geom = fig.add_subplot(gs[3, 1])
    ax_geom.set_facecolor('#2a2a2a')
    ax_geom.axis('off')
    ax_geom.set_xlim(0, 1)
    ax_geom.set_ylim(0, 1)
    rect = mpatches.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=3, edgecolor='cyan', facecolor='none')
    ax_geom.add_patch(rect)
    ax_geom.text(0.5, 0.85, 'GEOMETRY', ha='center', va='center', fontsize=12, weight='bold', color='white')
    
    if geometry_metrics:
        summary = geometry_metrics.get('system_summary', {})
        min_range = summary.get('min_range_m', 0)
        max_range = summary.get('max_range_m', 0)
        min_los = summary.get('min_los_gain_slope')
        max_los = summary.get('max_los_gain_slope')
        
        ax_geom.text(0.5, 0.65, f'Range: {min_range:.0f}-{max_range:.0f}m', 
                    ha='center', va='center', fontsize=9, color='lightgray')
        if min_los is not None and max_los is not None:
            ax_geom.text(0.5, 0.50, f'LOS Gain (slope):', 
                        ha='center', va='center', fontsize=9, color='lightgray')
            ax_geom.text(0.5, 0.35, f'{min_los:.3f} to {max_los:.3f}', 
                        ha='center', va='center', fontsize=9, color='cyan')
        else:
            ax_geom.text(0.5, 0.42, 'No slope motion', 
                        ha='center', va='center', fontsize=9, color='lightgray')
    else:
        ax_geom.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=10, color='lightgray')
    
    # Alarm Acknowledgment Tile
    ax_ack = fig.add_subplot(gs[3, 2])
    ax_ack.set_facecolor('#2a2a2a')
    ax_ack.axis('off')
    ax_ack.set_xlim(0, 1)
    ax_ack.set_ylim(0, 1)
    
    unacked_count = 0
    escalation_active = False
    
    if alarm_state:
        counts = alarm_state.get('counts', {})
        unacked_count = counts.get('critical_unacked', 0) + counts.get('alarm_unacked', 0)
        escalation = alarm_state.get('escalation', {})
        escalation_active = escalation.get('escalated_count', 0) > 0
    
    ack_color = 'red' if unacked_count > 0 else 'green'
    rect = mpatches.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=3, edgecolor=ack_color, facecolor='none')
    ax_ack.add_patch(rect)
    ax_ack.text(0.5, 0.7, 'ACK PENDING', ha='center', va='center', fontsize=12, weight='bold', color='white')
    ax_ack.text(0.5, 0.45, f'{unacked_count}', ha='center', va='center', fontsize=24, weight='bold', color=ack_color)
    
    if escalation_active:
        ax_ack.text(0.5, 0.25, 'ESCALATION', ha='center', va='center', 
                   fontsize=10, weight='bold', color='red', 
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    # Latest alarms table (moved to row 4)
    ax_alarms = fig.add_subplot(gs[4, :])
    ax_alarms.set_facecolor('#2a2a2a')
    ax_alarms.axis('off')
    ax_alarms.set_xlim(0, 1)
    ax_alarms.set_ylim(0, 1)
    
    ax_alarms.text(0.5, 0.95, 'LATEST ALARMS', ha='center', va='top',
                  fontsize=12, weight='bold', color='white')
    
    # Show last 5 alarms
    latest_alarms = alarms[-5:] if len(alarms) >= 5 else alarms
    if latest_alarms:
        y_pos = 0.85
        dy = 0.15
        for alarm in reversed(latest_alarms):  # Most recent first
            time_str = alarm.timestamp_iso.split('T')[1][:8] if 'T' in alarm.timestamp_iso else alarm.timestamp_iso[:8]
            alarm_str = f'{time_str}  {alarm.severity:8s}  {alarm.alarm_code:12s}  {alarm.target:10s}'
            color = status_colors.get(alarm.severity, 'white')
            ax_alarms.text(0.05, y_pos, alarm_str, ha='left', va='top',
                          fontsize=8, family='monospace', color=color)
            y_pos -= dy
    else:
        ax_alarms.text(0.5, 0.5, 'No alarms', ha='center', va='center',
                      fontsize=10, color='lightgray')
    
    # Mini sparklines
    # Sparkline 1: Slope displacement (if available)
    if len(slope_indices) > 0:
        ax_spark1 = fig.add_subplot(gs[1, 2])
        ax_spark1.set_facecolor('#2a2a2a')
        
        slope_idx = slope_indices[0]
        disp_mm = data['disp_los_m'][slope_idx] * 1000.0
        t_hr = data['t_s'] / 3600.0
        
        ax_spark1.plot(t_hr, disp_mm, color='cyan', linewidth=1)
        ax_spark1.set_title(f'{names[slope_idx]} Disp (mm)', fontsize=8, color='white')
        ax_spark1.tick_params(labelsize=6, colors='lightgray')
        ax_spark1.grid(True, alpha=0.2, color='gray')
        ax_spark1.set_facecolor('#2a2a2a')
        for spine in ax_spark1.spines.values():
            spine.set_color('gray')
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()
