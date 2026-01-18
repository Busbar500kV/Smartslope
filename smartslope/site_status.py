"""Site-level status and redundancy logic for multi-radar installations."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def compute_site_status(
    data: Dict,
    config: Dict,
    alarms: List
) -> Dict:
    """Compute site-level status from multi-radar observations.
    
    Implements K-of-M confirmation logic:
    - Site-level slope event if K slope targets confirmed by >= M radars
    - Site-level health status based on per-radar health
    
    Args:
        data: Dataset dictionary (potentially multi-radar)
        config: Site configuration
        alarms: List of alarm events
    
    Returns:
        Site status dictionary with keys:
            - status: "OK", "WARN", "ALARM", "CRITICAL"
            - slope_event_active: bool
            - contributing_radars: list of radar names (if multi-radar)
            - contributing_targets: list of slope target names
            - per_radar_health: dict of radar_name -> status
            - site_alarms: list of site-level alarm descriptions
    """
    site_status = {
        'status': 'OK',
        'slope_event_active': False,
        'contributing_radars': [],
        'contributing_targets': [],
        'per_radar_health': {},
        'site_alarms': []
    }
    
    # Check if multi-radar
    is_multi_radar = 'radar_names' in data and len(data['radar_names']) > 1
    
    if is_multi_radar:
        # Multi-radar site-level logic
        radar_names = data['radar_names']
        
        # Per-radar health (simplified: check if radar has valid data)
        for radar_name in radar_names:
            # TODO: Add more sophisticated health checks
            site_status['per_radar_health'][radar_name] = 'OK'
        
        # Check for slope events with radar confirmation
        alarm_cfg = config.get('alarms', {})
        k_consensus = alarm_cfg.get('slope_k_of_n_consensus', 2)
        m_radar_confirm = config.get('site_redundancy', {}).get('min_radar_confirmations', 1)
        
        # Find slope event alarms
        slope_alarms = [a for a in alarms if a.alarm_code == 'SLOPE_MOVE']
        
        if slope_alarms:
            # Group by target
            target_alarms = {}
            for alarm in slope_alarms:
                target = alarm.target
                if target not in target_alarms:
                    target_alarms[target] = []
                target_alarms[target].append(alarm)
            
            # Check which targets have M+ radar confirmations
            confirmed_targets = []
            for target, target_alarm_list in target_alarms.items():
                # In single-radar mode, each target has 1 confirmation
                # In multi-radar mode, count radars
                n_confirmations = len(target_alarm_list)
                if n_confirmations >= m_radar_confirm:
                    confirmed_targets.append(target)
            
            # Check if K+ targets confirmed
            if len(confirmed_targets) >= k_consensus:
                site_status['slope_event_active'] = True
                site_status['status'] = 'CRITICAL'
                site_status['contributing_targets'] = confirmed_targets
                site_status['site_alarms'].append(
                    f"Site-level slope event: {len(confirmed_targets)} targets confirmed"
                )
    
    else:
        # Single-radar site
        site_status['per_radar_health'] = {'primary': 'OK'}
        
        # Check for critical alarms
        critical_alarms = [a for a in alarms if a.severity == 'CRITICAL']
        if critical_alarms:
            site_status['status'] = 'CRITICAL'
            
            # Check for slope event consensus
            slope_event_alarms = [a for a in alarms if a.alarm_code == 'SLOPE_EVENT_SYSTEM']
            if slope_event_alarms:
                site_status['slope_event_active'] = True
                # Extract contributing targets from alarm message
                for alarm in slope_event_alarms:
                    if hasattr(alarm, 'details') and alarm.details:
                        targets = alarm.details.get('contributing_targets', [])
                        site_status['contributing_targets'].extend(targets)
        
        elif any(a.severity == 'ALARM' for a in alarms):
            site_status['status'] = 'ALARM'
        elif any(a.severity == 'WARN' for a in alarms):
            site_status['status'] = 'WARN'
    
    return site_status


def write_site_status_json(site_status: Dict, output_path) -> None:
    """Write site status to JSON file.
    
    Args:
        site_status: Site status dictionary
        output_path: Path to output JSON file
    """
    import json
    
    with open(output_path, 'w') as f:
        json.dump(site_status, f, indent=2)
