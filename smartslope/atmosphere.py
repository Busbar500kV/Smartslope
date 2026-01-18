"""Atmospheric phase screen simulation for radar measurements.

Implements time-correlated atmospheric phase effects with airmass weighting
and local turbulence components.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def compute_airmass_factor(
    radar_xyz: np.ndarray,
    reflector_xyz: np.ndarray,
    model: str = "secant",
    range_scale_m: float = 1000.0
) -> float:
    """Compute airmass factor for atmospheric phase contribution.
    
    Args:
        radar_xyz: Radar position (3,) in meters
        reflector_xyz: Reflector position (3,) in meters
        model: "secant" (elevation-based) or "range_linear" (range-based)
        range_scale_m: Scale factor for range_linear model
    
    Returns:
        Airmass factor (dimensionless, typically 1-10)
    """
    if model == "secant":
        # Elevation-based: airmass = sec(zenith_angle)
        # For radar: elevation angle from horizontal
        dx = reflector_xyz[0] - radar_xyz[0]
        dy = reflector_xyz[1] - radar_xyz[1]
        dz = reflector_xyz[2] - radar_xyz[2]
        
        range_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        if range_3d < 1e-6:
            return 1.0
        
        # Elevation angle from horizontal
        elevation_rad = np.arcsin(dz / range_3d)
        
        # Zenith angle from vertical
        zenith_rad = np.pi / 2 - elevation_rad
        
        # Secant formula (clamped to avoid singularity at horizon)
        zenith_rad = np.clip(zenith_rad, 0, np.pi / 2 - 0.01)
        airmass = 1.0 / np.cos(zenith_rad)
        
        return float(airmass)
    
    elif model == "range_linear":
        # Range-based: airmass proportional to slant range
        dx = reflector_xyz[0] - radar_xyz[0]
        dy = reflector_xyz[1] - radar_xyz[1]
        dz = reflector_xyz[2] - radar_xyz[2]
        
        range_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        airmass = range_3d / range_scale_m
        
        return float(airmass)
    
    else:
        raise ValueError(f"Unknown airmass model: {model}")


def generate_atmospheric_phase(
    n_reflectors: int,
    n_times: int,
    dt_s: float,
    config: Dict,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Generate atmospheric phase time-series for all reflectors.
    
    Implements:
        phi_atm_i(t) = common(t) * airmass_i + local_i(t)
    
    where:
        - common(t): Time-correlated common-mode phase (AR(1) process)
        - airmass_i: Airmass factor for reflector i
        - local_i(t): Local turbulence per reflector (AR(1) process)
    
    Args:
        n_reflectors: Number of reflectors
        n_times: Number of time samples
        dt_s: Time step in seconds
        config: Atmosphere configuration dict with keys:
            - common_ar1_rho: AR(1) coefficient for common mode (default: 0.995)
            - common_sigma_rad: Standard deviation of common mode (default: 0.5)
            - local_sigma_rad: Standard deviation of local turbulence (default: 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with:
            - phi_atm_rad: (n_reflectors, n_times) atmospheric phase in radians
            - atm_metric_rad: (n_times,) system-level metric (RMS common mode)
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    
    # Extract config parameters
    common_rho = config.get('common_ar1_rho', 0.995)
    common_sigma = config.get('common_sigma_rad', 0.5)
    local_sigma = config.get('local_sigma_rad', 0.1)
    
    # Generate common-mode AR(1) process
    # x[t] = rho * x[t-1] + sqrt(1 - rho^2) * sigma * noise[t]
    # This ensures stationary variance = sigma^2
    common_phase = np.zeros(n_times)
    common_phase[0] = rng.randn() * common_sigma
    
    innovation_sigma = common_sigma * np.sqrt(1 - common_rho**2)
    for t in range(1, n_times):
        common_phase[t] = common_rho * common_phase[t-1] + innovation_sigma * rng.randn()
    
    # Generate local turbulence per reflector (independent AR(1) processes)
    local_phase = np.zeros((n_reflectors, n_times))
    local_innovation_sigma = local_sigma * np.sqrt(1 - common_rho**2)
    
    for i in range(n_reflectors):
        local_phase[i, 0] = rng.randn() * local_sigma
        for t in range(1, n_times):
            local_phase[i, t] = common_rho * local_phase[i, t-1] + local_innovation_sigma * rng.randn()
    
    # Combine: phi_atm[i, t] = common[t] * airmass[i] + local[i, t]
    # Note: airmass factors are applied externally during simulation
    # Here we just return the common and local components
    
    # For simplicity, we'll return the combined phase assuming airmass=1
    # The actual airmass scaling happens in sim3d.py
    phi_atm_rad = common_phase[None, :] + local_phase
    
    # System-level metric: RMS of common mode
    atm_metric_rad = np.abs(common_phase)
    
    return {
        'phi_atm_rad': phi_atm_rad,
        'atm_metric_rad': atm_metric_rad,
        'common_phase_rad': common_phase,
        'local_phase_rad': local_phase
    }


def apply_airmass_scaling(
    phi_atm_rad: np.ndarray,
    common_phase_rad: np.ndarray,
    local_phase_rad: np.ndarray,
    airmass_factors: np.ndarray
) -> np.ndarray:
    """Apply airmass scaling to atmospheric phase.
    
    Computes: phi_atm_i(t) = common(t) * airmass_i + local_i(t)
    
    Args:
        phi_atm_rad: Placeholder phase array (n_reflectors, n_times)
        common_phase_rad: Common mode phase (n_times,)
        local_phase_rad: Local turbulence (n_reflectors, n_times)
        airmass_factors: Airmass for each reflector (n_reflectors,)
    
    Returns:
        Scaled atmospheric phase (n_reflectors, n_times)
    """
    n_reflectors, n_times = phi_atm_rad.shape
    
    # Apply airmass scaling
    phi_scaled = np.zeros((n_reflectors, n_times))
    for i in range(n_reflectors):
        phi_scaled[i, :] = common_phase_rad * airmass_factors[i] + local_phase_rad[i, :]
    
    return phi_scaled
