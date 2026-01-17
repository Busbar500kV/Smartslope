"""CLI entrypoint for smartslope package."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

from smartslope.simulate import main as simulate_main
from smartslope.detect import main as detect_main


def simulate_3d_main(config_path: str, outdir: str) -> int:
    """Run 3D simulation with config file."""
    import numpy as np
    from datetime import datetime
    from smartslope.sim3d import load_config_3d, simulate_3d
    from smartslope.scene import plot_scene_3d, plot_scene_before_after
    from smartslope.report import (
        plot_reflector_timeseries, 
        plot_timeseries_grid,
        generate_report,
        generate_manifest
    )
    from smartslope.io_npz import save_npz
    from smartslope.math_utils import unit
    
    config_path_obj = Path(config_path)
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading config: {config_path}")
    config = load_config_3d(config_path_obj)
    
    print("Running 3D simulation...")
    data = simulate_3d(config, seed=123)
    
    print(f"Output directory: {outdir_path}")
    
    generated_files = []
    
    # Save NPZ if requested
    output_cfg = config.get('output', {})
    if output_cfg.get('write_npz', True):
        npz_path = Path(output_cfg.get('npz_path', 'data/synthetic/site_demo_3d.npz'))
        if not npz_path.is_absolute():
            # Resolve relative to repo root
            repo_root = Path(__file__).resolve().parents[1]
            npz_path = repo_root / npz_path
        save_npz(npz_path, data)
        print(f"Wrote NPZ: {npz_path}")
    
    # Extract data for plotting
    radar_xyz = data['radar_xyz_m']
    reflector_xyz = data['pos_xyz_m']
    reflector_names = data['names'].tolist()
    reflector_roles = data['roles'].tolist()
    t_s = data['t_s']
    disp_true_xyz = data['disp_true_xyz_m']
    disp_los = data['disp_los_m']
    phi_unwrapped = data['phi_unwrapped']
    phi_wrapped = data['phi_wrapped']
    mask_valid = data['mask_valid']
    
    # Compute motion vectors for visualization
    motion_vectors = []
    for refl_cfg in config['reflectors']:
        motion_cfg = refl_cfg.get('motion')
        if motion_cfg is not None and motion_cfg.get('model') != 'none':
            direction = np.array(motion_cfg['direction_xyz_unit'], dtype=float)
            motion_vectors.append(unit(direction))
        else:
            motion_vectors.append(np.array([0.0, 0.0, 0.0]))
    
    # Plot 3D scene
    print("Generating scene_3d.png...")
    scene_path = outdir_path / 'scene_3d.png'
    plot_scene_3d(
        radar_xyz, reflector_xyz, reflector_names, reflector_roles,
        motion_vectors=motion_vectors, output_path=scene_path
    )
    generated_files.append('scene_3d.png')
    
    # Plot before/after
    print("Generating scene_3d_before_after.png...")
    before_after_time = output_cfg.get('before_after_time_s', 'end')
    if before_after_time == 'end':
        time_idx = -1
    else:
        time_idx = int(before_after_time / config['environment']['dt_s'])
    
    reflector_xyz_displaced = reflector_xyz + disp_true_xyz[:, time_idx, :]
    before_after_path = outdir_path / 'scene_3d_before_after.png'
    plot_scene_before_after(
        radar_xyz, reflector_xyz, reflector_xyz_displaced,
        reflector_names, reflector_roles, output_path=before_after_path
    )
    generated_files.append('scene_3d_before_after.png')
    
    # Plot time-series
    max_plots = output_cfg.get('max_reflector_plots', 12)
    n_reflectors = len(reflector_names)
    
    if n_reflectors <= max_plots:
        # Plot individual time-series for each reflector
        print(f"Generating individual time-series plots for {n_reflectors} reflectors...")
        for i in range(n_reflectors):
            ts_path = outdir_path / f'timeseries_{reflector_names[i]}.png'
            plot_reflector_timeseries(
                t_s, disp_true_xyz[i], disp_los[i],
                phi_unwrapped[i], phi_wrapped[i], mask_valid[i],
                reflector_names[i], ts_path
            )
            generated_files.append(f'timeseries_{reflector_names[i]}.png')
    
    # Always generate grid view
    print("Generating timeseries_grid.png...")
    grid_path = outdir_path / 'timeseries_grid.png'
    plot_timeseries_grid(
        t_s, disp_los, phi_unwrapped, mask_valid,
        reflector_names, reflector_roles, grid_path, max_plots=max_plots
    )
    generated_files.append('timeseries_grid.png')
    
    # Generate report
    print("Generating report.md...")
    generate_report(config, data, outdir_path, generated_files)
    generated_files.append('report.md')
    
    # Generate manifest
    print("Generating manifest.json...")
    generate_manifest(config, data, outdir_path, generated_files)
    generated_files.append('manifest.json')
    
    print(f"\n=== Simulation complete ===")
    print(f"Generated {len(generated_files)} files in {outdir_path}")
    
    return 0


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="smartslope",
        description="Smartslope: radar-based slope deformation detection"
    )
    parser.add_argument(
        "command",
        choices=["simulate", "detect", "pipeline"],
        help="Command to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON config file (for simulate command with 3D mode)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Output directory (for simulate command with 3D mode)"
    )
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        # Check if using new 3D mode (--config and --outdir)
        if args.config and args.outdir:
            return simulate_3d_main(args.config, args.outdir)
        else:
            # Legacy mode
            simulate_main()
            return 0
    elif args.command == "detect":
        detect_main()
        return 0
    elif args.command == "pipeline":
        # Run both simulate and detect
        print("=== Running simulation ===")
        simulate_main()
        print("\n=== Running detection ===")
        detect_main()
        print("\n=== Pipeline complete ===")
        return 0
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
