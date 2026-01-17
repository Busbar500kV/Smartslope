"""3D geometry-aware synthetic data generator for coherent radar."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from smartslope.math_utils import unit, wrap_pi
from smartslope.scene import compute_los_vector


def load_config_3d(config_path: Path) -> Dict:
    """Load JSON configuration for 3D simulation."""
    with open(config_path, 'r') as f:
        return json.load(f)


def normalize_direction(direction: List[float]) -> np.ndarray:
    """Normalize a direction vector to unit length."""
    d = np.array(direction, dtype=float)
    return unit(d)


def compute_motion_timeseries(
    t: np.ndarray,
    motion_cfg: Optional[Dict],
    direction_unit: np.ndarray
) -> np.ndarray:
    """
    Compute 3D displacement time-series for a reflector.
    
    Args:
        t: Time array (T,)
        motion_cfg: Motion configuration dict (or None for no motion)
        direction_unit: Unit direction vector (3,)
    
    Returns:
        Displacement array (T, 3) in meters
    """
    n = len(t)
    
    if motion_cfg is None or motion_cfg.get('model') == 'none':
        return np.zeros((n, 3), dtype=float)
    
    model = motion_cfg['model']
    
    # Compute magnitude time-series
    disp_magnitude = np.zeros(n, dtype=float)
    
    if model == 'creep':
        # Steady creep
        rate_mm_per_hr = motion_cfg.get('creep_rate_mm_per_hr', 0.0)
        rate_m_per_s = rate_mm_per_hr * 1e-3 / 3600.0
        disp_magnitude = rate_m_per_s * t
    
    elif model == 'creep_plus_event':
        # Creep + event
        rate_mm_per_hr = motion_cfg.get('creep_rate_mm_per_hr', 0.0)
        rate_m_per_s = rate_mm_per_hr * 1e-3 / 3600.0
        disp_magnitude = rate_m_per_s * t
        
        # Add event if enabled
        event = motion_cfg.get('event', {})
        if event.get('enabled', False):
            t0 = event.get('t0_s', 0.0)
            rise = event.get('rise_s', 1.0)
            slip_mm = event.get('slip_mm', 0.0)
            slip_m = slip_mm * 1e-3
            shape = event.get('shape', 'sigmoid')
            
            if shape == 'sigmoid':
                # Sigmoid function: S(t) = slip / (1 + exp(-k*(t - t0)))
                # where k = 4/rise ensures ~90% transition over rise time
                k = 4.0 / rise
                event_disp = slip_m / (1.0 + np.exp(-k * (t - t0)))
                disp_magnitude += event_disp
            
            elif shape == 'triangle':
                # Triangle: linear ramp from t0 to t0+rise
                event_disp = np.zeros_like(t)
                in_ramp = (t >= t0) & (t <= t0 + rise)
                event_disp[in_ramp] = slip_m * (t[in_ramp] - t0) / rise
                after_ramp = t > t0 + rise
                event_disp[after_ramp] = slip_m
                disp_magnitude += event_disp
    
    else:
        raise ValueError(f"Unknown motion model: {model}")
    
    # Convert magnitude to 3D displacement
    disp_3d = disp_magnitude[:, None] * direction_unit[None, :]
    
    return disp_3d


def simulate_3d(config: Dict, seed: int = 123) -> Dict[str, np.ndarray]:
    """
    Generate 3D geometry-aware synthetic coherent phase data.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
    
    Returns:
        Dictionary with arrays:
            - t_s: (T,)
            - names: (N,)
            - roles: (N,)
            - pos_xyz_m: (N, 3)
            - disp_true_xyz_m: (N, T, 3)
            - disp_los_m: (N, T)
            - phi_wrapped: (N, T)
            - phi_unwrapped: (N, T)
            - mask_valid: (N, T)
            - drift_rad: (T,)
            - wavelength_m: scalar
            - radar_xyz_m: (3,)
    """
    rng = np.random.default_rng(seed)
    
    # Parse config
    radar_cfg = config['radar']
    env_cfg = config['environment']
    reflectors_cfg = config['reflectors']
    drift_cfg = config.get('drift_model', {})
    
    # Time array
    duration_s = env_cfg['duration_s']
    dt_s = env_cfg['dt_s']
    n_samples = int(duration_s / dt_s)
    t = np.arange(n_samples, dtype=float) * dt_s
    
    # Radar position
    radar_xyz = np.array(radar_cfg['position_xyz_m'], dtype=float)
    
    # Wavelength
    wavelength_m = radar_cfg.get('wavelength_m')
    if wavelength_m is None:
        freq_hz = radar_cfg.get('frequency_hz')
        if freq_hz is None:
            raise ValueError("Must provide either wavelength_m or frequency_hz")
        c = 299792458.0  # speed of light m/s
        wavelength_m = c / freq_hz
    
    # Reflectors
    n_reflectors = len(reflectors_cfg)
    names = []
    roles = []
    pos_xyz = []
    amplitudes = []
    noise_sigmas = []
    dropout_probs = []
    motion_configs = []
    
    for refl in reflectors_cfg:
        names.append(refl['name'])
        roles.append(refl['role'])
        pos_xyz.append(refl['position_xyz_m'])
        amplitudes.append(refl.get('amplitude', 1.0))
        noise_sigmas.append(refl.get('noise_phase_sigma_rad', 0.1))
        dropout_probs.append(refl.get('dropout_prob', 0.0))
        motion_configs.append(refl.get('motion'))
    
    names = np.array(names)
    roles = np.array(roles)
    pos_xyz = np.array(pos_xyz, dtype=float)  # (N, 3)
    amplitudes = np.array(amplitudes, dtype=float)
    noise_sigmas = np.array(noise_sigmas, dtype=float)
    dropout_probs = np.array(dropout_probs, dtype=float)
    
    # Compute LOS vectors
    los_vectors = np.array([compute_los_vector(radar_xyz, pos_xyz[i]) for i in range(n_reflectors)])  # (N, 3)
    
    # Generate motion for each reflector
    disp_true_xyz = np.zeros((n_reflectors, n_samples, 3), dtype=float)
    
    for i in range(n_reflectors):
        motion_cfg = motion_configs[i]
        if motion_cfg is not None:
            direction_unit = normalize_direction(motion_cfg['direction_xyz_unit'])
            disp_true_xyz[i] = compute_motion_timeseries(t, motion_cfg, direction_unit)
    
    # Project 3D displacement onto LOS to get LOS displacement
    disp_los = np.zeros((n_reflectors, n_samples), dtype=float)
    for i in range(n_reflectors):
        # disp_los[i, t] = dot(disp_true_xyz[i, t], los_vectors[i])
        disp_los[i] = np.einsum('tc,c->t', disp_true_xyz[i], los_vectors[i])
    
    # Convert LOS displacement to phase: phi = (4*pi/lambda) * disp_los
    phi_motion = (4.0 * np.pi / wavelength_m) * disp_los
    
    # Common-mode drift
    drift_rad = np.zeros(n_samples, dtype=float)
    if drift_cfg.get('enabled', False):
        model = drift_cfg.get('model', 'random_walk')
        params = drift_cfg.get('parameters', {})
        
        if model == 'random_walk':
            sigma_per_sqrt_s = params.get('sigma_rad_per_sqrt_s', 0.0)
            step_std = sigma_per_sqrt_s * np.sqrt(dt_s)
            drift_rad[1:] = np.cumsum(rng.normal(0.0, step_std, size=n_samples - 1))
        
        elif model == 'sine':
            period_s = params.get('period_s', 86400.0)
            amplitude_rad = params.get('amplitude_rad', 0.1)
            drift_rad = amplitude_rad * np.sin(2.0 * np.pi * t / period_s)
    
    # Add drift and noise to phase
    phi_true = phi_motion + drift_rad[None, :]
    
    # Add per-reflector phase noise (scaled by amplitude)
    phi_noisy = np.zeros_like(phi_true)
    for i in range(n_reflectors):
        # Prevent division by zero for weak/zero amplitude reflectors
        amplitude_safe = max(amplitudes[i], 1e-6)
        noise_std = noise_sigmas[i] / np.sqrt(amplitude_safe)
        phi_noisy[i] = phi_true[i] + rng.normal(0.0, noise_std, size=n_samples)
    
    # Apply dropouts
    mask_valid = np.ones((n_reflectors, n_samples), dtype=np.uint8)
    for i in range(n_reflectors):
        dropout_mask = rng.random(n_samples) < dropout_probs[i]
        mask_valid[i, dropout_mask] = 0
    
    phi_noisy = np.where(mask_valid, phi_noisy, np.nan)
    
    # Unwrap phase per reflector
    phi_unwrapped = np.full_like(phi_noisy, np.nan)
    for i in range(n_reflectors):
        s = phi_noisy[i].copy()
        good = np.isfinite(s)
        if good.sum() < 2:
            continue
        
        # Interpolate to fill gaps for unwrapping
        idx = np.arange(n_samples)
        s_fill = s.copy()
        s_fill[~good] = np.interp(idx[~good], idx[good], s[good])
        
        # Unwrap
        u = np.unwrap(s_fill)
        
        # Keep only valid samples in output
        phi_unwrapped[i, good] = u[good]
    
    # Wrapped phase
    phi_wrapped = wrap_pi(phi_unwrapped)
    
    return {
        't_s': t,
        'names': names,
        'roles': roles,
        'pos_xyz_m': pos_xyz,
        'disp_true_xyz_m': disp_true_xyz,
        'disp_los_m': disp_los,
        'phi_wrapped': phi_wrapped,
        'phi_unwrapped': phi_unwrapped,
        'mask_valid': mask_valid,
        'drift_rad': drift_rad,
        'wavelength_m': np.array([wavelength_m], dtype=float),  # Array for NPZ compatibility
        'radar_xyz_m': radar_xyz,
    }
