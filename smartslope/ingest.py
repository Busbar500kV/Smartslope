"""Data ingest tools for importing real radar measurements.

Supports multiple input formats:
- NPZ: Direct NPZ files with schema-compliant structure
- CSV: Generic CSV with configurable column mapping
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from smartslope.schema import validate_dataset_dict, set_schema_version


def ingest_npz(path: Path) -> Dict:
    """Ingest dataset from NPZ file.
    
    Args:
        path: Path to NPZ file
    
    Returns:
        Dataset dictionary validated against schema
    
    Raises:
        ValueError: If NPZ structure is invalid
    """
    data_raw = np.load(path, allow_pickle=True)
    
    # Convert to regular dict
    data = {}
    for key in data_raw.files:
        data[key] = data_raw[key]
    
    # Validate schema
    warnings = validate_dataset_dict(data, allow_partial=True)
    if warnings:
        print(f"Validation warnings for {path}:")
        for w in warnings:
            print(f"  - {w}")
    
    # Set schema version
    set_schema_version(data)
    
    return data


def ingest_csv_generic(
    csv_path: Path,
    mapping: Dict,
    site_config: Optional[Dict] = None
) -> Dict:
    """Ingest dataset from generic CSV using column mapping.
    
    Args:
        csv_path: Path to CSV file
        mapping: Column mapping configuration with keys:
            - timestamp_column: Name of timestamp column (ISO8601 or Unix epoch)
            - timestamp_format: "iso8601" or "unix_epoch"
            - reflector_columns: Dict mapping reflector_name -> column_name
                e.g., {"REF_A": "phase_ref_a_rad", "SLOPE_01": "phase_slope_01_rad"}
            - reflector_roles: Dict mapping reflector_name -> "ref" or "slope"
            - validity_columns: Optional dict mapping reflector_name -> validity_column_name
        site_config: Optional site config for geometry metadata
    
    Returns:
        Dataset dictionary validated against schema
    
    Raises:
        ValueError: If CSV or mapping is invalid
    """
    import csv as csv_module
    
    # Read CSV
    with open(csv_path, 'r') as f:
        reader = csv_module.DictReader(f)
        rows = list(reader)
    
    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")
    
    # Extract configuration
    timestamp_col = mapping.get('timestamp_column', 'timestamp')
    timestamp_format = mapping.get('timestamp_format', 'iso8601')
    reflector_columns = mapping.get('reflector_columns', {})
    reflector_roles = mapping.get('reflector_roles', {})
    validity_columns = mapping.get('validity_columns', {})
    
    if not reflector_columns:
        raise ValueError("mapping must contain 'reflector_columns'")
    
    # Parse timestamps
    n_times = len(rows)
    t_iso = []
    t_s = []
    
    for i, row in enumerate(rows):
        ts_str = row.get(timestamp_col)
        if not ts_str:
            raise ValueError(f"Row {i} missing timestamp column '{timestamp_col}'")
        
        if timestamp_format == 'iso8601':
            # Parse ISO8601
            try:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError(f"Invalid ISO8601 timestamp at row {i}: {ts_str}")
            
            t_iso.append(dt.isoformat())
            
            # Compute relative time in seconds
            if i == 0:
                t0 = dt
            delta = (dt - t0).total_seconds()
            t_s.append(delta)
        
        elif timestamp_format == 'unix_epoch':
            # Parse Unix epoch
            try:
                epoch = float(ts_str)
            except ValueError:
                raise ValueError(f"Invalid Unix epoch at row {i}: {ts_str}")
            
            dt = datetime.utcfromtimestamp(epoch)
            t_iso.append(dt.isoformat() + 'Z')
            
            if i == 0:
                t0_epoch = epoch
            t_s.append(epoch - t0_epoch)
        
        else:
            raise ValueError(f"Unknown timestamp_format: {timestamp_format}")
    
    t_s = np.array(t_s)
    t_iso = np.array(t_iso)
    
    # Extract reflector data
    reflector_names = list(reflector_columns.keys())
    n_reflectors = len(reflector_names)
    
    # Initialize arrays
    phi_wrapped = np.zeros((n_reflectors, n_times))
    mask_valid = np.ones((n_reflectors, n_times), dtype=int)
    
    for i, name in enumerate(reflector_names):
        phase_col = reflector_columns[name]
        validity_col = validity_columns.get(name)
        
        for t, row in enumerate(rows):
            # Extract phase
            phase_str = row.get(phase_col)
            if not phase_str or phase_str.strip() == '':
                phi_wrapped[i, t] = 0.0
                mask_valid[i, t] = 0
            else:
                try:
                    phi_wrapped[i, t] = float(phase_str)
                except ValueError:
                    phi_wrapped[i, t] = 0.0
                    mask_valid[i, t] = 0
            
            # Extract validity if specified
            if validity_col:
                valid_str = row.get(validity_col, '1')
                try:
                    mask_valid[i, t] = int(valid_str)
                except ValueError:
                    mask_valid[i, t] = 1  # Default to valid
    
    # Unwrap phase (simple unwrapping)
    phi_unwrapped = np.copy(phi_wrapped)
    for i in range(n_reflectors):
        phi_unwrapped[i, :] = np.unwrap(phi_wrapped[i, :])
    
    # Build roles array
    roles = np.array([reflector_roles.get(name, 'slope') for name in reflector_names])
    names = np.array(reflector_names)
    
    # Build dataset dict
    data = {
        'names': names,
        'roles': roles,
        't_s': t_s,
        't_iso': t_iso,
        'phi_unwrapped': phi_unwrapped,
        'phi_wrapped': phi_wrapped,
        'mask_valid': mask_valid,
    }
    
    # Add geometry metadata if site_config provided
    if site_config:
        # For single-radar configs, add radar position
        if 'radar' in site_config:
            radar_pos = np.array(site_config['radar']['position_xyz_m'])
            data['radar_xyz_m'] = radar_pos
            
            wavelength = site_config['radar'].get('wavelength_m', 0.0125)
            data['wavelength_m'] = np.full(n_reflectors, wavelength)
            data['frequency_hz'] = np.full(n_reflectors, site_config['radar'].get('frequency_hz', 24e9))
        
        # Extract reflector positions if available
        reflector_configs = {r['name']: r for r in site_config.get('reflectors', [])}
        pos_xyz = []
        for name in reflector_names:
            refl_cfg = reflector_configs.get(name)
            if refl_cfg and 'position_xyz_m' in refl_cfg:
                pos_xyz.append(refl_cfg['position_xyz_m'])
            else:
                pos_xyz.append([0.0, 0.0, 0.0])
        
        if any(p != [0, 0, 0] for p in pos_xyz):
            data['pos_xyz_m'] = np.array(pos_xyz)
    
    # Validate schema
    warnings = validate_dataset_dict(data, allow_partial=True)
    if warnings:
        print(f"Validation warnings for {csv_path}:")
        for w in warnings:
            print(f"  - {w}")
    
    # Set schema version
    set_schema_version(data)
    
    return data


def load_mapping_json(mapping_path: Path) -> Dict:
    """Load column mapping from JSON file.
    
    Args:
        mapping_path: Path to mapping JSON file
    
    Returns:
        Mapping dictionary
    """
    with open(mapping_path, 'r') as f:
        return json.load(f)
