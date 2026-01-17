from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .io_npz import load_npz
from .math_utils import nanmedian_1d


def repo_root() -> Path:
    # .../Smartslope/smartslope/detect.py -> root is 1 level up
    return Path(__file__).resolve().parents[1]


def phase_to_disp(phi_rad: np.ndarray, wavelength_m: float) -> np.ndarray:
    # Δd = (λ / 4π) * Δφ
    return (wavelength_m / (4.0 * np.pi)) * phi_rad


def estimate_common_drift(phi_unwrapped: np.ndarray, roles: np.ndarray) -> np.ndarray:
    """
    Estimate common-mode drift using the median of reference reflector phases.
    Assumes references are stable; net reference phase change is drift.
    """
    ref_idx = np.where(roles == "reference")[0]
    if ref_idx.size == 0:
        raise ValueError("No reference reflectors found (role='reference').")

    drift_hat = np.full(phi_unwrapped.shape[1], np.nan)
    for t in range(phi_unwrapped.shape[1]):
        drift_hat[t] = nanmedian_1d(phi_unwrapped[ref_idx, t])

    # remove offset so drift_hat[0] = 0 (if possible)
    if np.isfinite(drift_hat[0]):
        drift_hat = drift_hat - drift_hat[0]

    return drift_hat


def simple_score(disp_corr: np.ndarray, roles: np.ndarray, threshold_m: float) -> np.ndarray:
    """
    v0 score: number of slope reflectors above abs displacement threshold at each time.
    """
    slope_idx = np.where(roles == "slope")[0]
    s = np.zeros(disp_corr.shape[1], dtype=int)
    for t in range(disp_corr.shape[1]):
        v = disp_corr[slope_idx, t]
        s[t] = int(np.sum(np.isfinite(v) & (np.abs(v) >= threshold_m)))
    return s


def main() -> None:
    root = repo_root()
    in_path = root / "data" / "synthetic" / "kegalle_demo_run.npz"
    out_dir = root / "outputs" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    z = load_npz(in_path)

    t = z["t_s"]
    names = z["names"].astype(str)
    roles = z["roles"].astype(str)
    phi_u = z["phi_unwrapped"]
    wavelength_m = float(z["wavelength_m"][0])

    drift_hat = estimate_common_drift(phi_u, roles)
    phi_corr = phi_u - drift_hat[None, :]

    disp_corr = phase_to_disp(phi_corr, wavelength_m)

    # v0: 2 cm LOS displacement threshold
    score = simple_score(disp_corr, roles, threshold_m=0.02)

    # Plot score
    plt.figure()
    plt.plot(t / 3600.0, score)
    plt.xlabel("time (hours)")
    plt.ylabel("count(|disp| >= 2cm) among slope reflectors")
    plt.title("Smartslope v0: simple coherence score")
    p1 = out_dir / "kegalle_demo_score.png"
    plt.savefig(p1, dpi=160, bbox_inches="tight")
    plt.close()

    # Write small summary
    p2 = out_dir / "kegalle_demo_summary.txt"
    with p2.open("w") as f:
        f.write(f"Input: {in_path}\n")
        f.write(f"Wavelength (m): {wavelength_m}\n")
        f.write(f"Reflectors: {len(names)} (refs={int(np.sum(roles=='reference'))}, slope={int(np.sum(roles=='slope'))})\n")
        f.write(f"Max score: {int(np.nanmax(score))}\n")

    print(f"Wrote {p1}")
    print(f"Wrote {p2}")


if __name__ == "__main__":
    main()