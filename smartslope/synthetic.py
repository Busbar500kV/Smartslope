"""
Synthetic coherent phase time-series generator.

Generates realistic radar phase data including:
- Slow creep motion
- Accelerating motion
- Step displacements
- Common-mode drift (instrument/atmosphere)
- Phase noise
- Data dropouts
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .io_utils import save_npz
from .math_utils import unit, wrap_pi


@dataclass
class RadarCfg:
    """Radar system configuration."""
    wavelength_m: float
    sample_period_s: float
    n_samples: int
    seed: int = 123


@dataclass
class ReflectorCfg:
    """Individual reflector configuration."""
    name: str
    xyz_m: Tuple[float, float, float]
    role: str  # "reference" or "slope"


@dataclass
class NoiseCfg:
    """Noise and data quality parameters."""
    phase_noise_std_rad: float
    drift_std_rad_per_s: float
    dropout_prob: float


@dataclass
class MotionCfg:
    """Motion model configuration."""
    model: str  # "none" | "creep" | "accelerating" | "step"
    direction_xyz: Tuple[float, float, float]
    creep_mm_per_day: float = 0.0
    accel_mm_per_day2: float = 0.0
    step_mm: float = 0.0
    step_at_sample: int = 0


@dataclass
class InstallCfg:
    """Complete installation configuration."""
    radar: RadarCfg
    reflectors: List[ReflectorCfg]
    noise: NoiseCfg
    motion: MotionCfg


def load_cfg(path: Path) -> InstallCfg:
    """Load configuration from JSON file."""
    d = json.loads(path.read_text())
    radar = RadarCfg(**d["radar"])
    noise = NoiseCfg(**d["noise"])
    motion = MotionCfg(**d["motion"])
    refl = [
        ReflectorCfg(name=r["name"], xyz_m=tuple(r["xyz_m"]), role=r["role"])
        for r in d["reflectors"]
    ]
    return InstallCfg(radar=radar, reflectors=refl, noise=noise, motion=motion)


def simulate(cfg: InstallCfg) -> Dict[str, np.ndarray]:
    """
    Generate synthetic coherent phase time-series.
    
    Physics model:
      φ_measured = (4π/λ) * Δr_LOS + drift + noise
      
    where:
      - Δr_LOS is line-of-sight displacement from motion model
      - drift is common-mode phase drift (random walk)
      - noise is independent phase measurement noise
      - dropouts are modeled as NaN values
    
    Returns:
        Dictionary with arrays:
        - t_s: time in seconds
        - names: reflector names
        - roles: reflector roles ("reference" or "slope")
        - phi_unwrapped: unwrapped phase (rad)
        - phi_wrapped: wrapped phase (rad)
        - disp_true_m: true LOS displacement (m)
        - drift_rad: common drift component (rad)
        - mask_valid: validity mask (1=valid, 0=dropout)
        - wavelength_m: radar wavelength
    """
    rng = np.random.default_rng(cfg.radar.seed)

    n = cfg.radar.n_samples
    dt = cfg.radar.sample_period_s
    t = np.arange(n, dtype=float) * dt

    names = np.array([r.name for r in cfg.reflectors])
    roles = np.array([r.role for r in cfg.reflectors])

    xyz = np.array([r.xyz_m for r in cfg.reflectors], dtype=float)  # (R,3)
    los = np.array([unit(xyz[i]) for i in range(xyz.shape[0])])     # (R,3)

    direction = unit(np.array(cfg.motion.direction_xyz, dtype=float))

    # Initialize 3D displacement array (reflectors, time, xyz)
    disp3 = np.zeros((xyz.shape[0], n, 3), dtype=float)

    # Conversion factors
    MM_TO_M = 1e-3
    SECONDS_PER_DAY = 24.0 * 3600.0
    mm_per_day_to_m_per_s = MM_TO_M / SECONDS_PER_DAY
    
    if cfg.motion.model == "none":
        pass

    elif cfg.motion.model == "creep":
        v = cfg.motion.creep_mm_per_day * mm_per_day_to_m_per_s
        disp_mag = v * t
        for i in range(xyz.shape[0]):
            if roles[i] == "slope":
                disp3[i] = disp_mag[:, None] * direction[None, :]

    elif cfg.motion.model == "accelerating":
        v0 = cfg.motion.creep_mm_per_day * mm_per_day_to_m_per_s
        a = cfg.motion.accel_mm_per_day2 * (MM_TO_M / (SECONDS_PER_DAY ** 2))
        disp_mag = v0 * t + 0.5 * a * t * t
        for i in range(xyz.shape[0]):
            if roles[i] == "slope":
                disp3[i] = disp_mag[:, None] * direction[None, :]

    elif cfg.motion.model == "step":
        step_m = cfg.motion.step_mm * MM_TO_M
        k = int(cfg.motion.step_at_sample)
        for i in range(xyz.shape[0]):
            if roles[i] == "slope":
                disp3[i, k:] = step_m * direction[None, :]

    else:
        raise ValueError(f"Unknown motion model: {cfg.motion.model}")

    # Project 3D displacement to LOS (R,T)
    disp_true_m = np.einsum("rti,ri->rt", disp3, los)

    # Common phase drift (random walk)
    drift = np.zeros(n, dtype=float)
    step_std = cfg.noise.drift_std_rad_per_s * np.sqrt(dt)
    drift[1:] = np.cumsum(rng.normal(0.0, step_std, size=n - 1))

    lam = cfg.radar.wavelength_m
    phi_true = (4.0 * np.pi / lam) * disp_true_m  # (R,T)

    # Add drift and noise
    phi = phi_true + drift[None, :] + rng.normal(
        0.0, cfg.noise.phase_noise_std_rad, size=phi_true.shape
    )

    # Apply dropouts
    mask_valid = rng.random(phi.shape) > cfg.noise.dropout_prob
    phi = np.where(mask_valid, phi, np.nan)

    # Unwrap phase per reflector (fill gaps for unwrap; keep NaNs in output)
    phi_unwrapped = np.full_like(phi, np.nan)
    for i in range(phi.shape[0]):
        s = phi[i].copy()
        good = np.isfinite(s)
        if good.sum() < 2:
            continue
        idx = np.arange(n)
        s_fill = s.copy()
        s_fill[~good] = np.interp(idx[~good], idx[good], s[good])
        u = np.unwrap(s_fill)
        phi_unwrapped[i, good] = u[good]

    phi_wrapped = wrap_pi(phi_unwrapped)

    return {
        "t_s": t,
        "names": names,
        "roles": roles,
        "phi_unwrapped": phi_unwrapped,
        "phi_wrapped": phi_wrapped,
        "disp_true_m": disp_true_m,
        "drift_rad": drift,
        "mask_valid": mask_valid.astype(np.uint8),
        "wavelength_m": np.array([cfg.radar.wavelength_m], dtype=float),
    }


def repo_root() -> Path:
    """Get repository root directory (2 levels up from smartslope/synthetic.py)."""
    # smartslope/synthetic.py -> smartslope -> repo_root
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """Generate synthetic data from default config."""
    root = repo_root()
    cfg_path = root / "smartslope" / "configs" / "kegalle_demo.json"
    out_path = root / "data" / "synthetic" / "kegalle_demo_run.npz"

    print(f"Loading config: {cfg_path}")
    cfg = load_cfg(cfg_path)
    
    print(f"Generating synthetic data...")
    print(f"  - {len(cfg.reflectors)} reflectors")
    print(f"  - {cfg.radar.n_samples} samples @ {cfg.radar.sample_period_s}s")
    print(f"  - Motion model: {cfg.motion.model}")
    
    arrays = simulate(cfg)
    save_npz(out_path, arrays)
    print(f"✓ Wrote {out_path}")


if __name__ == "__main__":
    main()
