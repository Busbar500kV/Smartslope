"""3D geometry-aware synthetic data generator for coherent radar."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def apply_alarm_injection(
    config: Dict,
    t: np.ndarray,
    names: np.ndarray,
    disp_true_xyz: np.ndarray,
    noise_sigmas: np.ndarray,
    dropout_probs: np.ndarray,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply alarm injection scenarios from config.
    
    Modifies displacement, noise, and dropout arrays to trigger alarms at known times.
    
    Args:
        config: Configuration dictionary
        t: Time array (T,)
        names: Reflector names (N,)
        disp_true_xyz: True 3D displacement (N, T, 3)
        noise_sigmas: Per-reflector phase noise sigmas (N,)
        dropout_probs: Per-reflector dropout probabilities (N,)
        rng: Random number generator
    
    Returns:
        Modified (disp_true_xyz, noise_sigmas_modified, dropout_probs_modified)
    """
    injection_cfg = config.get('alarm_injection', {})
    if not injection_cfg.get('enable', False):
        return disp_true_xyz, noise_sigmas, dropout_probs
    
    scenarios = injection_cfg.get('scenarios', [])
    if not scenarios:
        return disp_true_xyz, noise_sigmas, dropout_probs
    
    dt_s = config['environment']['dt_s']
    n_samples = len(t)
    n_reflectors = len(names)
    
    # Create modifiable copies
    disp_mod = disp_true_xyz.copy()
    noise_mod = np.tile(noise_sigmas[:, None], (1, n_samples))
    dropout_mod = np.tile(dropout_probs[:, None], (1, n_samples))
    
    for scenario in scenarios:
        name = scenario.get('name', 'unknown')
        t0_s = scenario.get('t0_s', 0.0)
        duration_s = scenario.get('duration_s', 0.0)
        
        # Find time window
        t1_s = t0_s + duration_s
        mask = (t >= t0_s) & (t < t1_s)
        
        if scenario.get('extra_phase_noise_sigma_rad') is not None:
            # Add extra noise to all reflectors
            extra_noise = scenario['extra_phase_noise_sigma_rad']
            noise_mod[:, mask] += extra_noise
            print(f"  Alarm injection: {name} at t={t0_s:.0f}s, duration={duration_s:.0f}s (extra noise)")
        
        if scenario.get('forced_disp_mm') is not None and scenario.get('target') is not None:
            # Force displacement on specific target
            target_name = scenario['target']
            target_idx = np.where(names == target_name)[0]
            if len(target_idx) == 0:
                print(f"  Warning: target {target_name} not found for injection {name}")
                continue
            target_idx = target_idx[0]
            
            forced_disp_m = scenario['forced_disp_mm'] * 1e-3
            direction = np.array(scenario.get('direction_xyz_unit', [1.0, 0.0, 0.0]), dtype=float)
            direction = unit(direction)
            
            # Add forced displacement
            disp_mod[target_idx, mask, :] += forced_disp_m * direction[None, :]
            print(f"  Alarm injection: {name} at t={t0_s:.0f}s, duration={duration_s:.0f}s (ref instability on {target_name})")
        
        if scenario.get('extra_dropout_prob') is not None:
            # Increase dropout probability for all reflectors
            extra_dropout = scenario['extra_dropout_prob']
            dropout_mod[:, mask] = np.minimum(dropout_mod[:, mask] + extra_dropout, 1.0)
            print(f"  Alarm injection: {name} at t={t0_s:.0f}s, duration={duration_s:.0f}s (extra dropouts)")
    
    # Convert noise_mod back to per-sample usage (will be applied later)
    # For now, return noise sigmas that will be used per-timestep
    return disp_mod, noise_mod, dropout_mod


def compute_beam_attenuation(
    radar_xyz: np.ndarray,
    reflector_xyz: np.ndarray,
    beam_cfg: Dict
) -> Tuple[float, float]:
    """Compute beam pattern attenuation factor for a reflector.
    
    Args:
        radar_xyz: Radar position (3,)
        reflector_xyz: Reflector position (3,)
        beam_cfg: Beam configuration with keys:
            - yaw_deg: Beam yaw angle (degrees, 0=North)
            - pitch_deg: Beam pitch angle (degrees, 0=horizontal)
            - beamwidth_deg: 3dB beamwidth (degrees)
            - pattern: "gaussian" or "cosine"
    
    Returns:
        (attenuation_factor, off_boresight_deg)
        attenuation_factor: 0.0 to 1.0 (1.0 = on boresight)
        off_boresight_deg: Off-boresight angle in degrees
    """
    # Compute LOS vector
    los = reflector_xyz - radar_xyz
    los_range = np.linalg.norm(los)
    if los_range < 1e-6:
        return 1.0, 0.0
    
    los_unit = los / los_range
    
    # Compute beam boresight direction
    yaw_rad = np.deg2rad(beam_cfg.get('yaw_deg', 0.0))
    pitch_rad = np.deg2rad(beam_cfg.get('pitch_deg', 0.0))
    
    # Boresight unit vector (yaw from North, pitch from horizontal)
    boresight = np.array([
        np.sin(yaw_rad) * np.cos(pitch_rad),  # East
        np.cos(yaw_rad) * np.cos(pitch_rad),  # North
        np.sin(pitch_rad)                      # Up
    ])
    
    # Off-boresight angle
    cos_angle = np.dot(los_unit, boresight)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    off_boresight_rad = np.arccos(cos_angle)
    off_boresight_deg = np.rad2deg(off_boresight_rad)
    
    # Compute attenuation based on pattern
    beamwidth_deg = beam_cfg.get('beamwidth_deg', 20.0)
    pattern = beam_cfg.get('pattern', 'gaussian')
    
    if pattern == 'gaussian':
        # Gaussian pattern: exp(-4*ln(2) * (theta/theta_3db)^2)
        # This gives 0.5 (-3dB) at theta = theta_3db
        sigma_deg = beamwidth_deg / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        attenuation = np.exp(-0.5 * (off_boresight_deg / sigma_deg)**2)
    
    elif pattern == 'cosine':
        # Cosine pattern: max(0, cos(theta))^n
        # where n is chosen to give ~3dB at beamwidth_deg
        n = np.log(0.5) / np.log(np.cos(np.deg2rad(beamwidth_deg / 2)))
        attenuation = max(0.0, np.cos(off_boresight_rad))**n
    
    else:
        # Unknown pattern - assume uniform
        attenuation = 1.0
    
    return float(attenuation), float(off_boresight_deg)


def apply_multipath_bias(
    n_reflectors: int,
    n_times: int,
    multipath_cfg: Dict,
    rng: np.random.Generator
) -> np.ndarray:
    """Generate slowly varying multipath phase bias per reflector.
    
    Args:
        n_reflectors: Number of reflectors
        n_times: Number of time samples
        multipath_cfg: Configuration with keys:
            - bias_sigma_rad: Standard deviation of bias
            - bias_rho: AR(1) coefficient (0.999 = very slow)
    
    Returns:
        Phase bias array (n_reflectors, n_times)
    """
    bias_sigma = multipath_cfg.get('bias_sigma_rad', 0.2)
    bias_rho = multipath_cfg.get('bias_rho', 0.999)
    
    # Generate AR(1) process for each reflector
    bias = np.zeros((n_reflectors, n_times))
    innovation_sigma = bias_sigma * np.sqrt(1 - bias_rho**2)
    
    for i in range(n_reflectors):
        bias[i, 0] = rng.normal(0.0, bias_sigma)
        for t in range(1, n_times):
            bias[i, t] = bias_rho * bias[i, t-1] + innovation_sigma * rng.normal()
    
    return bias


def generate_timestamps(config: Dict, t_s: np.ndarray) -> tuple[list[str], np.ndarray]:
    """
    Generate ISO8601 timestamp strings and unix timestamps.
    
    Args:
        config: Configuration dictionary with 'hmi' section
        t_s: Time array in seconds (T,)
    
    Returns:
        (t_iso, t_unix_s) where t_iso is list of ISO strings and t_unix_s is unix timestamps
    """
    hmi_cfg = config.get('hmi', {})
    start_time_iso = hmi_cfg.get('start_time_iso')
    
    if start_time_iso is None:
        # Fallback: use current time
        start_dt = datetime.now(timezone.utc)
    else:
        # Parse ISO8601 with timezone
        start_dt = datetime.fromisoformat(start_time_iso)
    
    # Generate timestamps
    t_iso = []
    t_unix_s = np.zeros(len(t_s), dtype=float)
    
    for i, t_offset in enumerate(t_s):
        dt = start_dt + timedelta(seconds=float(t_offset))
        t_iso.append(dt.isoformat())
        t_unix_s[i] = dt.timestamp()
    
    return t_iso, t_unix_s


def simulate_3d(config: Dict, seed: int = 123) -> Dict[str, np.ndarray]:
    """
    Generate 3D geometry-aware synthetic coherent phase data.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
    
    Returns:
        Dictionary with arrays:
            - t_s: (T,)
            - t_iso: list of ISO8601 strings (T,)
            - t_unix_s: (T,) unix timestamps
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
    
    # Apply alarm injection scenarios (modifies displacement, noise, dropout)
    print("Checking for alarm injection scenarios...")
    disp_true_xyz, noise_sigmas_per_t, dropout_probs_per_t = apply_alarm_injection(
        config, t, names, disp_true_xyz, noise_sigmas, dropout_probs, rng
    )
    
    # === NEW PHYSICS FEATURES ===
    
    # A1: DEM/Terrain support
    terrain_cfg = config.get('terrain', {})
    dem = None
    height_above_ground = np.full(n_reflectors, np.nan)
    occluded_fraction = np.zeros(n_reflectors)
    
    if terrain_cfg.get('enabled', False):
        print("Processing terrain/DEM...")
        from smartslope.dem import load_dem_npz, compute_height_above_ground, check_los_occlusion
        
        dem_path = terrain_cfg.get('dem_npz_path')
        if dem_path:
            from pathlib import Path as PathLib
            dem_path_obj = PathLib(dem_path)
            if not dem_path_obj.is_absolute():
                repo_root = PathLib(__file__).resolve().parents[1]
                dem_path_obj = repo_root / dem_path
            
            try:
                dem = load_dem_npz(dem_path_obj)
                
                # Compute height above ground for each reflector
                for i in range(n_reflectors):
                    height_above_ground[i] = compute_height_above_ground(dem, pos_xyz[i])
                
                # Check LOS occlusion
                for i in range(n_reflectors):
                    is_occluded, frac = check_los_occlusion(dem, radar_xyz, pos_xyz[i])
                    occluded_fraction[i] = frac
                    
                    # If occluded, reduce amplitude or mask out
                    if is_occluded and terrain_cfg.get('apply_occlusion_masking', True):
                        # Reduce validity based on occlusion fraction
                        mask_valid[i, :] = (rng.random(n_samples) > frac).astype(np.uint8)
                
                print(f"  DEM loaded: {dem_path}")
                print(f"  Height above ground range: {np.nanmin(height_above_ground):.1f} to {np.nanmax(height_above_ground):.1f} m")
                print(f"  Max occlusion fraction: {np.max(occluded_fraction):.2%}")
            
            except Exception as e:
                print(f"  Warning: Failed to load DEM: {e}")
    
    # A2: Atmospheric phase screen
    atm_cfg = config.get('atmosphere', {})
    phi_atm = np.zeros((n_reflectors, n_samples))
    atm_metric_rad = np.zeros(n_samples)
    
    if atm_cfg.get('enabled', False):
        print("Generating atmospheric phase screen...")
        from smartslope.atmosphere import (
            generate_atmospheric_phase,
            compute_airmass_factor,
            apply_airmass_scaling
        )
        
        # Generate base atmospheric phase
        atm_data = generate_atmospheric_phase(
            n_reflectors, n_samples, dt_s, atm_cfg, seed=seed+1
        )
        
        # Compute airmass factors
        airmass_model = atm_cfg.get('airmass_model', 'secant')
        range_scale = atm_cfg.get('airmass_range_scale_m', 1000.0)
        airmass_factors = np.array([
            compute_airmass_factor(radar_xyz, pos_xyz[i], airmass_model, range_scale)
            for i in range(n_reflectors)
        ])
        
        # Apply airmass scaling
        phi_atm = apply_airmass_scaling(
            atm_data['phi_atm_rad'],
            atm_data['common_phase_rad'],
            atm_data['local_phase_rad'],
            airmass_factors
        )
        
        atm_metric_rad = atm_data['atm_metric_rad']
        
        print(f"  Atmospheric phase: common RMS = {np.std(atm_data['common_phase_rad']):.3f} rad")
        print(f"  Airmass factors: {np.min(airmass_factors):.2f} to {np.max(airmass_factors):.2f}")
    
    # A3: Beam pattern + RCS proxy
    beam_cfg = config.get('beam', {})
    snr_proxy = np.ones(n_reflectors)  # 1.0 = nominal
    beam_attenuation = np.ones(n_reflectors)
    
    if beam_cfg.get('enabled', False):
        print("Computing beam pattern attenuation...")
        
        for i in range(n_reflectors):
            atten, off_boresight = compute_beam_attenuation(radar_xyz, pos_xyz[i], beam_cfg)
            beam_attenuation[i] = atten
            snr_proxy[i] = atten  # Simplified: SNR proxy = beam attenuation
            
            # Map attenuation to increased phase noise
            # Lower attenuation → higher noise
            # noise_increase = 1 / sqrt(attenuation) to model SNR
            if atten > 0.1:
                noise_factor = 1.0 / np.sqrt(atten)
                noise_sigmas_per_t[i, :] *= noise_factor
        
        print(f"  Beam attenuation range: {np.min(beam_attenuation):.3f} to {np.max(beam_attenuation):.3f}")
        print(f"  SNR proxy range: {np.min(snr_proxy):.3f} to {np.max(snr_proxy):.3f}")
    
    # A4: Multipath bias
    multipath_cfg = config.get('multipath', {})
    phi_multipath = np.zeros((n_reflectors, n_samples))
    
    if multipath_cfg.get('enabled', False):
        print("Generating multipath phase bias...")
        phi_multipath = apply_multipath_bias(n_reflectors, n_samples, multipath_cfg, rng)
        print(f"  Multipath bias RMS: {np.std(phi_multipath):.3f} rad")
    
    # === END NEW PHYSICS FEATURES ===
    
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
    
    # Add atmospheric phase
    phi_true += phi_atm
    
    # Add multipath bias
    phi_true += phi_multipath
    
    # Add per-reflector phase noise (scaled by amplitude, with injection)
    phi_noisy = np.zeros_like(phi_true)
    for i in range(n_reflectors):
        # Prevent division by zero for weak/zero amplitude reflectors
        amplitude_safe = max(amplitudes[i], 1e-6)
        for t_idx in range(n_samples):
            noise_std = noise_sigmas_per_t[i, t_idx] / np.sqrt(amplitude_safe)
            phi_noisy[i, t_idx] = phi_true[i, t_idx] + rng.normal(0.0, noise_std)
    
    # Apply dropouts (with injection)
    mask_valid = np.ones((n_reflectors, n_samples), dtype=np.uint8)
    for i in range(n_reflectors):
        for t_idx in range(n_samples):
            if rng.random() < dropout_probs_per_t[i, t_idx]:
                mask_valid[i, t_idx] = 0
    
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
    
    # Generate timestamps
    print("Generating timestamps...")
    t_iso, t_unix_s = generate_timestamps(config, t)
    
    # Build return dictionary
    result = {
        't_s': t,
        't_iso': np.array(t_iso, dtype=object),  # Array of strings
        't_unix_s': t_unix_s,
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
    
    # Add new physics outputs
    if atm_cfg.get('enabled', False):
        result['atm_metric_rad'] = atm_metric_rad
    
    if beam_cfg.get('enabled', False):
        result['snr_proxy'] = snr_proxy
        result['beam_attenuation'] = beam_attenuation
    
    if terrain_cfg.get('enabled', False) and dem is not None:
        result['height_above_ground_m'] = height_above_ground
        result['occluded_fraction'] = occluded_fraction
    
    return result
