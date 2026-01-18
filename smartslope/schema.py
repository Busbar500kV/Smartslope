"""Dataset schema validation for internal measurement representation.

Defines versioned schema for synthetic and imported datasets to ensure
consistent structure across ingest, simulation, and analysis pipelines.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


SCHEMA_VERSION = "1.0"


def validate_dataset_dict(data: Dict, allow_partial: bool = False) -> List[str]:
    """Validate dataset dictionary structure.
    
    Args:
        data: Dataset dictionary to validate
        allow_partial: If True, allows missing optional fields (for replay mode)
    
    Returns:
        List of validation warnings (empty if all valid)
    
    Raises:
        ValueError: If required fields are missing or invalid
    """
    warnings = []
    
    # Required fields for all datasets
    required_fields = [
        'names',           # (N,) array of reflector names
        'roles',           # (N,) array of roles ('ref', 'slope')
        't_s',             # (T,) time array in seconds
        't_iso',           # (T,) ISO8601 timestamp strings
        'phi_unwrapped',   # (N, T) unwrapped phase in radians
        'phi_wrapped',     # (N, T) wrapped phase in radians
        'mask_valid',      # (N, T) validity mask (1=valid, 0=invalid)
    ]
    
    # Optional fields (simulation-specific)
    optional_fields = [
        'radar_xyz_m',        # (3,) radar position
        'pos_xyz_m',          # (N, 3) reflector positions
        'disp_true_xyz_m',    # (N, T, 3) true 3D displacement
        'disp_los_m',         # (N, T) LOS displacement
        'wavelength_m',       # (N,) or scalar wavelength
        'frequency_hz',       # (N,) or scalar frequency
    ]
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            if allow_partial and field in ['pos_xyz_m', 'radar_xyz_m']:
                warnings.append(f"Optional field '{field}' missing (replay mode)")
                continue
            raise ValueError(f"Required field '{field}' missing from dataset")
    
    # Validate shapes
    n_reflectors = len(data['names'])
    n_times = len(data['t_s'])
    
    if len(data['roles']) != n_reflectors:
        raise ValueError(f"'roles' length {len(data['roles'])} != n_reflectors {n_reflectors}")
    
    if len(data['t_iso']) != n_times:
        raise ValueError(f"'t_iso' length {len(data['t_iso'])} != n_times {n_times}")
    
    # Check phase and mask arrays
    for field in ['phi_unwrapped', 'phi_wrapped', 'mask_valid']:
        arr = data[field]
        expected_shape = (n_reflectors, n_times)
        if arr.shape != expected_shape:
            raise ValueError(f"'{field}' shape {arr.shape} != expected {expected_shape}")
    
    # Validate optional fields if present
    if 'pos_xyz_m' in data:
        pos = data['pos_xyz_m']
        if pos.shape != (n_reflectors, 3):
            raise ValueError(f"'pos_xyz_m' shape {pos.shape} != expected ({n_reflectors}, 3)")
    
    if 'disp_true_xyz_m' in data:
        disp = data['disp_true_xyz_m']
        if disp.shape != (n_reflectors, n_times, 3):
            raise ValueError(f"'disp_true_xyz_m' shape {disp.shape} != expected ({n_reflectors}, {n_times}, 3)")
    
    if 'disp_los_m' in data:
        disp_los = data['disp_los_m']
        if disp_los.shape != (n_reflectors, n_times):
            raise ValueError(f"'disp_los_m' shape {disp_los.shape} != expected ({n_reflectors}, {n_times})")
    
    # Check radar position if present
    if 'radar_xyz_m' in data:
        radar_pos = data['radar_xyz_m']
        if not (isinstance(radar_pos, np.ndarray) and radar_pos.shape == (3,)):
            raise ValueError(f"'radar_xyz_m' must be shape (3,), got {type(radar_pos)} with shape {getattr(radar_pos, 'shape', 'N/A')}")
    
    # Check for multi-radar fields
    if 'radar_names' in data:
        # Multi-radar dataset
        n_radars = len(data['radar_names'])
        
        # Check per-radar phase arrays
        if 'phi_unwrapped_per_radar' in data:
            phi_per_radar = data['phi_unwrapped_per_radar']
            expected_shape = (n_radars, n_reflectors, n_times)
            if phi_per_radar.shape != expected_shape:
                warnings.append(f"'phi_unwrapped_per_radar' shape {phi_per_radar.shape} != expected {expected_shape}")
    
    return warnings


def get_schema_version(data: Dict) -> str:
    """Get schema version from dataset.
    
    Args:
        data: Dataset dictionary
    
    Returns:
        Schema version string (default: "1.0")
    """
    return data.get('schema_version', SCHEMA_VERSION)


def set_schema_version(data: Dict) -> None:
    """Set schema version in dataset.
    
    Args:
        data: Dataset dictionary (modified in-place)
    """
    data['schema_version'] = SCHEMA_VERSION


def is_synthetic_dataset(data: Dict) -> bool:
    """Check if dataset is synthetic (has ground truth fields).
    
    Args:
        data: Dataset dictionary
    
    Returns:
        True if dataset has synthetic ground truth fields
    """
    synthetic_fields = ['disp_true_xyz_m', 'pos_xyz_m', 'radar_xyz_m']
    return all(field in data for field in synthetic_fields)


def is_multi_radar_dataset(data: Dict) -> bool:
    """Check if dataset contains multi-radar measurements.
    
    Args:
        data: Dataset dictionary
    
    Returns:
        True if dataset has multi-radar fields
    """
    return 'radar_names' in data and len(data.get('radar_names', [])) > 1
