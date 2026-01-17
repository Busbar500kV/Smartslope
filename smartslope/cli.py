#!/usr/bin/env python3
"""CLI entrypoint for Smartslope pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from .simulate import load_cfg, simulate
from .io_npz import save_npz, load_npz
from .detect import (
    phase_to_disp,
    estimate_common_drift,
    simple_score,
)


def run_pipeline(config_path: Path, outdir: Path) -> None:
    """
    Run the full Smartslope pipeline:
    1. Load config
    2. Generate synthetic data
    3. Run baseline detection
    4. Save outputs
    """
    print(f"=== Smartslope Pipeline ===")
    print(f"Config: {config_path}")
    print(f"Output directory: {outdir}")
    print()

    # Load configuration
    cfg = load_cfg(config_path)
    print(f"Loaded config: {cfg.radar.n_samples} samples @ {cfg.radar.sample_period_s}s period")
    print(f"Reflectors: {len(cfg.reflectors)} total")
    
    # Generate synthetic data
    print("\n[1/3] Generating synthetic phase data...")
    arrays = simulate(cfg)
    
    # Save synthetic dataset
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Use config filename as base for output names
    base_name = config_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = data_dir / f"{base_name}_{timestamp}.npz"
    
    save_npz(data_file, arrays)
    print(f"  → Saved synthetic data: {data_file}")
    
    # Run detection
    print("\n[2/3] Running baseline detection...")
    t = arrays["t_s"]
    names = arrays["names"].astype(str)
    roles = arrays["roles"].astype(str)
    phi_u = arrays["phi_unwrapped"]
    wavelength_m = float(arrays["wavelength_m"][0])
    
    drift_hat = estimate_common_drift(phi_u, roles)
    phi_corr = phi_u - drift_hat[None, :]
    disp_corr = phase_to_disp(phi_corr, wavelength_m)
    
    # Simple scoring (v0: 2 cm threshold)
    score = simple_score(disp_corr, roles, threshold_m=0.02)
    
    # Save outputs
    print("\n[3/3] Saving outputs...")
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Plot score
    plt.figure(figsize=(10, 4))
    plt.plot(t / 3600.0, score, linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Count (|disp| >= 2cm) among slope reflectors")
    plt.title("Smartslope: Simple Coherence Score")
    plt.grid(True, alpha=0.3)
    
    score_plot = outdir / f"{base_name}_score_{timestamp}.png"
    plt.savefig(score_plot, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  → Saved plot: {score_plot}")
    
    # Write summary report
    summary_file = outdir / f"{base_name}_summary_{timestamp}.txt"
    with summary_file.open("w") as f:
        f.write(f"Smartslope Pipeline Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Run timestamp: {timestamp}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Data file: {data_file}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Wavelength: {wavelength_m} m\n")
        f.write(f"  Sample period: {cfg.radar.sample_period_s} s\n")
        f.write(f"  Number of samples: {cfg.radar.n_samples}\n")
        f.write(f"  Total duration: {t[-1]/3600.0:.2f} hours\n\n")
        f.write(f"Reflectors:\n")
        f.write(f"  Total: {len(names)}\n")
        f.write(f"  References: {int(np.sum(roles=='reference'))}\n")
        f.write(f"  Slope targets: {int(np.sum(roles=='slope'))}\n\n")
        f.write(f"Detection Results:\n")
        f.write(f"  Max score: {int(np.nanmax(score))}\n")
        f.write(f"  Mean score: {np.nanmean(score):.2f}\n\n")
        f.write(f"Outputs:\n")
        f.write(f"  Score plot: {score_plot}\n")
        f.write(f"  Summary: {summary_file}\n")
    
    print(f"  → Saved summary: {summary_file}")
    
    # Print summary to console
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    print(f"Generated dataset: {data_file}")
    print(f"Output directory: {outdir}")
    print(f"  - Score plot: {score_plot.name}")
    print(f"  - Summary report: {summary_file.name}")
    print()
    print(f"Reflectors: {len(names)} ({int(np.sum(roles=='reference'))} ref, {int(np.sum(roles=='slope'))} slope)")
    print(f"Max detection score: {int(np.nanmax(score))} / {int(np.sum(roles=='slope'))}")
    print("="*50)


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Smartslope: Synthetic radar slope deformation detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("code/synthetic/configs/kegalle_demo.json"),
        help="Path to JSON configuration file (default: code/synthetic/configs/kegalle_demo.json)",
    )
    
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for plots and reports (default: outputs)",
    )
    
    args = parser.parse_args()
    
    # Validate config exists
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1
    
    try:
        run_pipeline(args.config, args.outdir)
        return 0
    except Exception as e:
        print(f"Error: Pipeline failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
