"""Geometry and sensitivity metrics for radar-reflector configurations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from smartslope.math_utils import unit
from smartslope.scene import compute_range, compute_los_vector


def compute_vertical_angle(los_vector: np.ndarray) -> float:
    """
    Compute vertical elevation angle from horizontal plane.
    
    Args:
        los_vector: LOS unit vector (3,)
    
    Returns:
        Vertical angle in degrees (positive = above horizontal)
    """
    # Vertical angle is arcsin of the Z component
    return float(np.degrees(np.arcsin(los_vector[2])))


def compute_azimuth(los_vector: np.ndarray) -> float:
    """
    Compute azimuth angle (bearing) in horizontal plane.
    
    Args:
        los_vector: LOS unit vector (3,)
    
    Returns:
        Azimuth in degrees (0=North, 90=East, clockwise from North)
    """
    # Azimuth from North (Y-axis is North, X-axis is East)
    azimuth_rad = np.arctan2(los_vector[0], los_vector[1])  # atan2(East, North)
    azimuth_deg = np.degrees(azimuth_rad)
    
    # Normalize to [0, 360)
    if azimuth_deg < 0:
        azimuth_deg += 360.0
    
    return float(azimuth_deg)


def compute_los_gain(los_vector: np.ndarray, motion_direction: Optional[np.ndarray]) -> Optional[float]:
    """
    Compute LOS gain factor for motion sensitivity.
    
    This is the dot product of the LOS unit vector and the motion direction unit vector.
    - 1.0 = motion directly along LOS (maximum sensitivity, positive)
    - -1.0 = motion directly opposite LOS (maximum sensitivity, negative)
    - 0.0 = motion perpendicular to LOS (invisible to radar)
    
    Args:
        los_vector: LOS unit vector (3,)
        motion_direction: Motion direction unit vector (3,), or None if no motion
    
    Returns:
        LOS gain factor, or None if motion_direction is None
    """
    if motion_direction is None or np.allclose(motion_direction, 0.0):
        return None
    
    motion_unit = unit(motion_direction)
    return float(np.dot(los_vector, motion_unit))


def compute_geometry_metrics(
    radar_xyz: np.ndarray,
    reflector_xyz: np.ndarray,
    reflector_names: List[str],
    reflector_roles: List[str],
    motion_directions: Optional[List[Optional[np.ndarray]]],
    wavelength_m: float,
    config: Dict
) -> Dict:
    """
    Compute geometry and sensitivity metrics for all reflectors.
    
    Args:
        radar_xyz: Radar position (3,)
        reflector_xyz: Reflector positions (N, 3)
        reflector_names: List of reflector names
        reflector_roles: List of reflector roles ("ref" or "slope")
        motion_directions: List of motion direction vectors (or None for each)
        wavelength_m: Radar wavelength in meters
        config: Configuration dictionary
    
    Returns:
        Dictionary with:
            - system_summary: Dict with system-level metrics
            - reflectors: List of dicts with per-reflector metrics
    """
    n_reflectors = len(reflector_names)
    
    # Extract radar parameters from config
    radar_cfg = config.get('radar', {})
    radar_station_cfg = config.get('radar_station', {})
    
    freq_hz = radar_station_cfg.get('frequency_hz', radar_cfg.get('frequency_hz', 24e9))
    freq_ghz = freq_hz / 1e9
    tx_power_dbm = radar_station_cfg.get('tx_power_dbm', 20)
    
    # Compute per-reflector metrics
    reflector_metrics = []
    ranges = []
    los_gains = []
    
    for i in range(n_reflectors):
        name = reflector_names[i]
        role = reflector_roles[i]
        pos = reflector_xyz[i]
        
        # Compute LOS vector and range
        los_vec = compute_los_vector(radar_xyz, pos)
        range_m = compute_range(radar_xyz, pos)
        ranges.append(range_m)
        
        # Compute angles
        vertical_angle_deg = compute_vertical_angle(los_vec)
        azimuth_deg = compute_azimuth(los_vec)
        
        # Sensitivity metrics
        # Phase sensitivity: phi = (4*pi/lambda) * disp_los
        # Therefore: dphi/d(disp_los) = 4*pi/lambda [rad/m]
        sensitivity_rad_per_m = (4.0 * np.pi) / wavelength_m
        sensitivity_rad_per_mm = sensitivity_rad_per_m / 1000.0
        
        # Inverse: displacement per phase
        sensitivity_mm_per_rad = 1000.0 / sensitivity_rad_per_m
        
        # LOS gain for motion (only for slope targets with motion)
        motion_dir = motion_directions[i] if motion_directions else None
        los_gain = compute_los_gain(los_vec, motion_dir)
        
        if role == 'slope' and los_gain is not None:
            los_gains.append(los_gain)
        
        # Build metrics dict
        metrics = {
            'name': name,
            'role': role,
            'pos_x_m': float(pos[0]),
            'pos_y_m': float(pos[1]),
            'pos_z_m': float(pos[2]),
            'range_m': float(range_m),
            'u_los_x': float(los_vec[0]),
            'u_los_y': float(los_vec[1]),
            'u_los_z': float(los_vec[2]),
            'vertical_angle_deg': float(vertical_angle_deg),
            'azimuth_deg': float(azimuth_deg),
            'sensitivity_mm_per_rad': float(sensitivity_mm_per_rad),
            'sensitivity_rad_per_mm': float(sensitivity_rad_per_mm),
        }
        
        # Add LOS gain if available
        if los_gain is not None:
            metrics['los_gain_for_motion'] = float(los_gain)
            metrics['predicted_los_mm_per_true_mm'] = float(los_gain)
        else:
            metrics['los_gain_for_motion'] = None
            metrics['predicted_los_mm_per_true_mm'] = None
        
        reflector_metrics.append(metrics)
    
    # System summary
    ref_count = sum(1 for r in reflector_roles if r == 'ref')
    slope_count = sum(1 for r in reflector_roles if r == 'slope')
    
    system_summary = {
        'wavelength_m': float(wavelength_m),
        'wavelength_mm': float(wavelength_m * 1000.0),
        'frequency_ghz': float(freq_ghz),
        'tx_power_dbm': float(tx_power_dbm),
        'ref_count': ref_count,
        'slope_count': slope_count,
        'min_range_m': float(min(ranges)) if ranges else 0.0,
        'max_range_m': float(max(ranges)) if ranges else 0.0,
    }
    
    if los_gains:
        system_summary['min_los_gain_slope'] = float(min(los_gains))
        system_summary['max_los_gain_slope'] = float(max(los_gains))
    else:
        system_summary['min_los_gain_slope'] = None
        system_summary['max_los_gain_slope'] = None
    
    return {
        'system_summary': system_summary,
        'reflectors': reflector_metrics
    }


def write_geometry_metrics_csv(metrics: Dict, output_path: Path) -> None:
    """
    Write geometry metrics to CSV file.
    
    Args:
        metrics: Metrics dictionary from compute_geometry_metrics
        output_path: Output CSV file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write system summary header
        writer.writerow(['# System Summary'])
        summary = metrics['system_summary']
        for key, value in summary.items():
            writer.writerow([f'# {key}', value if value is not None else ''])
        writer.writerow([])
        
        # Write reflector data header
        writer.writerow([
            'name', 'role', 'pos_x_m', 'pos_y_m', 'pos_z_m', 'range_m',
            'u_los_x', 'u_los_y', 'u_los_z',
            'vertical_angle_deg', 'azimuth_deg',
            'sensitivity_mm_per_rad', 'sensitivity_rad_per_mm',
            'los_gain_for_motion', 'predicted_los_mm_per_true_mm'
        ])
        
        # Write reflector data
        for refl in metrics['reflectors']:
            writer.writerow([
                refl['name'],
                refl['role'],
                f"{refl['pos_x_m']:.3f}",
                f"{refl['pos_y_m']:.3f}",
                f"{refl['pos_z_m']:.3f}",
                f"{refl['range_m']:.3f}",
                f"{refl['u_los_x']:.6f}",
                f"{refl['u_los_y']:.6f}",
                f"{refl['u_los_z']:.6f}",
                f"{refl['vertical_angle_deg']:.3f}",
                f"{refl['azimuth_deg']:.3f}",
                f"{refl['sensitivity_mm_per_rad']:.6f}",
                f"{refl['sensitivity_rad_per_mm']:.6f}",
                f"{refl['los_gain_for_motion']:.6f}" if refl['los_gain_for_motion'] is not None else '',
                f"{refl['predicted_los_mm_per_true_mm']:.6f}" if refl['predicted_los_mm_per_true_mm'] is not None else ''
            ])


def write_geometry_metrics_json(metrics: Dict, output_path: Path) -> None:
    """
    Write geometry metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary from compute_geometry_metrics
        output_path: Output JSON file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
