"""Site configuration loader and validator for multi-radar installations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_site_config(config_path: Path) -> Dict:
    """Load site configuration from JSON file.
    
    Supports both single-radar (legacy) and multi-radar configurations.
    
    Args:
        config_path: Path to JSON configuration file
    
    Returns:
        Normalized configuration dict with 'radars' as a list
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Normalize: convert single radar to list of radars
    if 'radar' in config and 'radars' not in config:
        # Legacy single-radar config
        config['radars'] = [config['radar']]
        # Keep 'radar' for backwards compatibility
    elif 'radars' in config:
        # Multi-radar config - already in correct format
        pass
    else:
        raise ValueError("Config must contain either 'radar' or 'radars' key")
    
    return config


def validate_site_config(config: Dict) -> None:
    """Validate site configuration structure.
    
    Args:
        config: Configuration dictionary
    
    Raises:
        ValueError: If configuration is invalid
    """
    # Check radars
    if 'radars' not in config:
        raise ValueError("Config must contain 'radars' key")
    
    radars = config['radars']
    if not isinstance(radars, list) or len(radars) == 0:
        raise ValueError("'radars' must be a non-empty list")
    
    radar_names = set()
    for i, radar in enumerate(radars):
        if 'name' not in radar:
            raise ValueError(f"Radar {i} missing 'name' field")
        
        name = radar['name']
        if name in radar_names:
            raise ValueError(f"Duplicate radar name: {name}")
        radar_names.add(name)
        
        if 'position_xyz_m' not in radar:
            raise ValueError(f"Radar {name} missing 'position_xyz_m' field")
    
    # Check reflectors
    if 'reflectors' not in config:
        raise ValueError("Config must contain 'reflectors' key")
    
    reflectors = config['reflectors']
    if not isinstance(reflectors, list):
        raise ValueError("'reflectors' must be a list")
    
    reflector_names = set()
    for i, refl in enumerate(reflectors):
        if 'name' not in refl:
            raise ValueError(f"Reflector {i} missing 'name' field")
        
        name = refl['name']
        if name in reflector_names:
            raise ValueError(f"Duplicate reflector name: {name}")
        reflector_names.add(name)
        
        # Check visibility assignment (optional)
        if 'visible_to' in refl:
            visible_to = refl['visible_to']
            if not isinstance(visible_to, list):
                raise ValueError(f"Reflector {name} 'visible_to' must be a list")
            
            # Verify all radar names exist
            for radar_name in visible_to:
                if radar_name not in radar_names:
                    raise ValueError(f"Reflector {name} refers to unknown radar: {radar_name}")


def get_reflector_radar_visibility(config: Dict) -> Dict[str, List[str]]:
    """Determine which radars can see each reflector.
    
    Args:
        config: Site configuration dict
    
    Returns:
        Dictionary mapping reflector_name -> list of radar_names
        If 'visible_to' is not specified for a reflector, it's visible to all radars.
    """
    radar_names = [r['name'] for r in config['radars']]
    
    visibility = {}
    for refl in config['reflectors']:
        name = refl['name']
        if 'visible_to' in refl:
            visibility[name] = refl['visible_to']
        else:
            # Default: visible to all radars
            visibility[name] = radar_names
    
    return visibility


def is_multi_radar_config(config: Dict) -> bool:
    """Check if configuration has multiple radars.
    
    Args:
        config: Site configuration dict
    
    Returns:
        True if config has more than one radar
    """
    return len(config.get('radars', [])) > 1
