from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from smartslope.io_npz import save_npz
from smartslope.math_utils import unit, wrap_pi


@dataclass
class RadarCfg:
    wavelength_m: float
    sample_period_s: float
    n_samples: int
    seed: int = 123


@dataclass
class ReflectorCfg:
    name: str
    xyz_m: Tuple[float, float, float]
    role: str  # "reference" or "slope"


@dataclass
class NoiseCfg:
    phase_noise_std_rad: float
    drift_std_rad_per_s: float
    dropout_prob: float


@dataclass
class MotionCfg:
    model: str  # "none" | "creep" | "accelerating" | "step"
    direction_xyz: Tuple[float, float, float]
    creep_mm_per_day: float = 0.0
    accel_mm_per_day2: float = 0.0
    step_mm: float = 0.0
    step_at_sample: int = 0


@dataclass
class InstallCfg:
    radar: RadarCfg
    reflectors: List[ReflectorCfg]
    noise: NoiseCfg
    motion: MotionCfg


def load_cfg(path: Path) -> InstallCfg:
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
    Coherent model:
      phi_true = (4π/λ) * Δr_LOS
    with:
      Δr_LOS from motion projected to LOS,
      common drift as phase random walk,
      iid phase noise,
      dropouts.
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

    disp3 = np.zeros((xyz.shape[0], n, 3), dtype=float)

    mm_per_day_to_m_per_s = 1e-3 / (24.0 * 3600.0)
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
        a = cfg.motion.accel_mm_per_day2 * (1e-3 / (24.0 * 3600.0) ** 2)
        disp_mag = v0 * t + 0.5 * a * t * t
        for i in range(xyz.shape[0]):
            if roles[i] == "slope":
                disp3[i] = disp_mag[:, None] * direction[None, :]

    elif cfg.motion.model == "step":
        step_m = cfg.motion.step_mm * 1e-3
        k = int(cfg.motion.step_at_sample)
        for i in range(xyz.shape[0]):
            if roles[i] == "slope":
                disp3[i, k:] = step_m * direction[None, :]

    else:
        raise ValueError(f"Unknown motion model: {cfg.motion.model}")

    # LOS displacement (R,T)
    disp_true_m = np.einsum("rti,ri->rt", disp3, los)

    # Common phase drift (random walk)
    drift = np.zeros(n, dtype=float)
    step_std = cfg.noise.drift_std_rad_per_s * np.sqrt(dt)
    drift[1:] = np.cumsum(rng.normal(0.0, step_std, size=n - 1))

    lam = cfg.radar.wavelength_m
    phi_true = (4.0 * np.pi / lam) * disp_true_m  # (R,T)

    phi = phi_true + drift[None, :] + rng.normal(0.0, cfg.noise.phase_noise_std_rad, size=phi_true.shape)

    mask_valid = rng.random(phi.shape) > cfg.noise.dropout_prob
    phi = np.where(mask_valid, phi, np.nan)

    # unwrap per reflector (fill gaps for unwrap; keep NaNs in output)
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
    # .../Smartslope/smartslope/simulate.py -> root is 2 levels up
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = repo_root()
    cfg_path = root / "code" / "synthetic" / "configs" / "kegalle_demo.json"
    out_path = root / "data" / "synthetic" / "kegalle_demo_run.npz"

    cfg = load_cfg(cfg_path)
    arrays = simulate(cfg)
    save_npz(out_path, arrays)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()