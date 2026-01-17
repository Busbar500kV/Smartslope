"""
Baseline detection pipeline for slope deformation events.

Pipeline stages:
1. Phase unwrapping (done in synthetic module)
2. Drift removal using reference reflectors
3. Phase → displacement conversion
4. Velocity estimation
5. Event detection using persistence/coherence logic
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from .io_utils import load_npz
from .math_utils import nanmedian_1d


def phase_to_disp(phi_rad: np.ndarray, wavelength_m: float) -> np.ndarray:
    """
    Convert phase to displacement.
    
    Δd = (λ / 4π) * Δφ
    
    Args:
        phi_rad: Phase in radians (R, T)
        wavelength_m: Radar wavelength in meters
        
    Returns:
        Displacement in meters (R, T)
    """
    return (wavelength_m / (4.0 * np.pi)) * phi_rad


def estimate_common_drift(phi_unwrapped: np.ndarray, roles: np.ndarray) -> np.ndarray:
    """
    Estimate common-mode drift using reference reflectors.
    
    Strategy:
    - Reference reflectors are assumed stable (no motion)
    - Median of reference phases at each time = common drift
    - Remove DC offset so drift starts at zero
    
    Args:
        phi_unwrapped: Unwrapped phase (R, T)
        roles: Reflector roles array
        
    Returns:
        Estimated drift in radians (T,)
    """
    ref_idx = np.where(roles == "reference")[0]
    if ref_idx.size == 0:
        raise ValueError("No reference reflectors found (role='reference').")

    drift_hat = np.full(phi_unwrapped.shape[1], np.nan)
    for t in range(phi_unwrapped.shape[1]):
        drift_hat[t] = nanmedian_1d(phi_unwrapped[ref_idx, t])

    # Remove offset so drift_hat[0] = 0 (if possible)
    if np.isfinite(drift_hat[0]):
        drift_hat = drift_hat - drift_hat[0]

    return drift_hat


def estimate_velocity(
    disp: np.ndarray, t: np.ndarray, window_samples: int = 10
) -> np.ndarray:
    """
    Estimate velocity using finite differences over a sliding window.
    
    Args:
        disp: Displacement (R, T)
        t: Time array (T,)
        window_samples: Window size for velocity estimation
        
    Returns:
        Velocity in m/s (R, T)
    """
    vel = np.full_like(disp, np.nan)
    
    for i in range(disp.shape[0]):
        for t_idx in range(window_samples, disp.shape[1]):
            start = t_idx - window_samples
            d_slice = disp[i, start:t_idx+1]
            t_slice = t[start:t_idx+1]
            
            valid = np.isfinite(d_slice)
            if valid.sum() >= 2:
                # Linear fit
                t_valid = t_slice[valid]
                d_valid = d_slice[valid]
                vel[i, t_idx] = np.polyfit(t_valid, d_valid, 1)[0]
    
    return vel


def detect_events(
    disp_corr: np.ndarray,
    vel: np.ndarray,
    roles: np.ndarray,
    disp_threshold_m: float = 0.02,
    vel_threshold_m_per_s: float = 1e-6,
    min_reflectors: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect deformation events using displacement and velocity thresholds.
    
    Event detection logic:
    - Count slope reflectors exceeding displacement threshold
    - Count slope reflectors exceeding velocity threshold
    - Event is flagged when counts exceed minimum
    
    Args:
        disp_corr: Drift-corrected displacement (R, T)
        vel: Velocity estimate (R, T)
        roles: Reflector roles
        disp_threshold_m: Displacement threshold (default 2cm)
        vel_threshold_m_per_s: Velocity threshold
        min_reflectors: Minimum reflectors to flag event
        
    Returns:
        Tuple of (disp_score, vel_score) arrays (T,)
    """
    slope_idx = np.where(roles == "slope")[0]
    
    disp_score = np.zeros(disp_corr.shape[1], dtype=int)
    vel_score = np.zeros(vel.shape[1], dtype=int)
    
    for t in range(disp_corr.shape[1]):
        # Displacement score
        d_vals = disp_corr[slope_idx, t]
        disp_score[t] = int(
            np.sum(np.isfinite(d_vals) & (np.abs(d_vals) >= disp_threshold_m))
        )
        
        # Velocity score
        v_vals = vel[slope_idx, t]
        vel_score[t] = int(
            np.sum(np.isfinite(v_vals) & (np.abs(v_vals) >= vel_threshold_m_per_s))
        )
    
    return disp_score, vel_score


def repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).resolve().parents[1]


def main() -> None:
    """Run baseline detection pipeline."""
    root = repo_root()
    in_path = root / "data" / "synthetic" / "kegalle_demo_run.npz"
    out_dir = root / "outputs" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data: {in_path}")
    z = load_npz(in_path)

    t = z["t_s"]
    names = z["names"].astype(str)
    roles = z["roles"].astype(str)
    phi_u = z["phi_unwrapped"]
    wavelength_m = float(z["wavelength_m"][0])
    
    n_refs = int(np.sum(roles == "reference"))
    n_slope = int(np.sum(roles == "slope"))
    print(f"Reflectors: {len(names)} ({n_refs} reference, {n_slope} slope)")

    # Stage 1: Estimate and remove common drift
    print("Estimating common drift...")
    drift_hat = estimate_common_drift(phi_u, roles)
    phi_corr = phi_u - drift_hat[None, :]

    # Stage 2: Convert phase to displacement
    print("Converting phase to displacement...")
    disp_corr = phase_to_disp(phi_corr, wavelength_m)

    # Stage 3: Estimate velocity
    print("Estimating velocity...")
    vel = estimate_velocity(disp_corr, t, window_samples=10)

    # Stage 4: Detect events
    print("Detecting events...")
    disp_threshold_m = 0.02  # 2 cm
    vel_threshold_m_per_s = 1e-6  # ~3.6 mm/hr
    disp_score, vel_score = detect_events(
        disp_corr, vel, roles,
        disp_threshold_m=disp_threshold_m,
        vel_threshold_m_per_s=vel_threshold_m_per_s,
    )

    # Generate plots
    print("Generating plots...")
    
    # Plot 1: Displacement score
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    t_hours = t / 3600.0
    disp_threshold_mm = disp_threshold_m * 1000
    vel_threshold_mm_per_hr = vel_threshold_m_per_s * 3600 * 1000
    
    ax1.plot(t_hours, disp_score, 'b-', linewidth=2)
    ax1.set_ylabel(f"Displacement score\n(count |d| ≥ {disp_threshold_mm:.0f}cm)")
    ax1.set_title("Smartslope Detection Pipeline: Event Scores")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t_hours, vel_score, 'r-', linewidth=2)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel(f"Velocity score\n(count |v| ≥ {vel_threshold_mm_per_hr:.1f}mm/hr)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    p1 = out_dir / "detection_scores.png"
    plt.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"✓ Wrote {p1}")

    # Plot 2: Displacement time series for slope reflectors
    fig, ax = plt.subplots(figsize=(10, 5))
    slope_idx = np.where(roles == "slope")[0]
    for i in slope_idx:
        d_mm = disp_corr[i, :] * 1000  # Convert to mm
        ax.plot(t_hours, d_mm, label=names[i], alpha=0.7)
    
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("LOS Displacement (mm)")
    ax.set_title("Drift-Corrected Displacement (Slope Reflectors)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    p2 = out_dir / "displacement_timeseries.png"
    plt.savefig(p2, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"✓ Wrote {p2}")

    # Write summary report
    p3 = out_dir / "detection_summary.txt"
    with p3.open("w") as f:
        f.write("=" * 60 + "\n")
        f.write("Smartslope Detection Pipeline Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Input: {in_path}\n")
        f.write(f"Wavelength: {wavelength_m*1000:.2f} mm\n")
        f.write(f"Duration: {t[-1]/3600:.1f} hours\n")
        f.write(f"Sample period: {t[1]-t[0]:.0f} seconds\n\n")
        f.write(f"Reflectors:\n")
        f.write(f"  - Reference: {n_refs}\n")
        f.write(f"  - Slope: {n_slope}\n\n")
        f.write(f"Detection Results:\n")
        f.write(f"  - Max displacement score: {int(np.nanmax(disp_score))}\n")
        f.write(f"  - Max velocity score: {int(np.nanmax(vel_score))}\n")
        f.write(f"  - Samples with events (disp): {int(np.sum(disp_score > 0))}\n")
        f.write(f"  - Samples with events (vel): {int(np.sum(vel_score > 0))}\n\n")
        
        # Statistics on slope reflectors
        f.write(f"Slope Reflector Statistics:\n")
        for i in slope_idx:
            d_vals = disp_corr[i, :]
            valid = np.isfinite(d_vals)
            if valid.sum() > 0:
                max_disp = np.max(np.abs(d_vals[valid])) * 1000  # mm
                f.write(f"  - {names[i]}: max |d| = {max_disp:.2f} mm\n")

    print(f"✓ Wrote {p3}")
    print("\n" + "=" * 60)
    print("Detection pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
