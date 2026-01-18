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
    output_path: Path
) -> None:
    """
    Render HMI-style station dashboard image.
    
    Args:
        data: Simulation output data
        config: Configuration dictionary
        alarms: List of AlarmEvent objects
        run_id: Run identifier
        output_path: Output PNG file path
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
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a1a')  # Dark background for HMI look
    
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)
    
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
    
    # Latest alarms table
    ax_alarms = fig.add_subplot(gs[3, 1:])
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
