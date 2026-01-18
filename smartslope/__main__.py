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
    from smartslope.scene_2d import plot_plan_view, plot_elevation_view
    from smartslope.geometry_metrics import (
        compute_geometry_metrics,
        write_geometry_metrics_csv,
        write_geometry_metrics_json
    )
    from smartslope.report import (
        plot_reflector_timeseries, 
        plot_timeseries_grid,
        generate_report,
        generate_manifest,
        generate_results_md
    )
    from smartslope.alarms import (
        generate_alarms,
        write_alarm_log_csv,
        plot_alarm_timeline,
        apply_alarm_latching_and_acks,
        write_alarm_state_json,
        generate_ack_template
    )
    from smartslope.scada_export import write_scada_telemetry
    from smartslope.hmi import render_hmi_dashboard
    from smartslope.io_npz import save_npz
    from smartslope.math_utils import unit
    
    config_path_obj = Path(config_path)
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    
    # Extract run_id from output directory
    run_id = outdir_path.name
    
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
    
    # Generate geometry metrics
    print("Computing geometry metrics...")
    wavelength_m = float(data['wavelength_m'][0])
    motion_directions_for_metrics = []
    for i, refl_cfg in enumerate(config['reflectors']):
        motion_cfg = refl_cfg.get('motion')
        if motion_cfg is not None and motion_cfg.get('model') != 'none':
            direction = np.array(motion_cfg['direction_xyz_unit'], dtype=float)
            motion_directions_for_metrics.append(direction)
        else:
            motion_directions_for_metrics.append(None)
    
    from smartslope.geometry_metrics import compute_geometry_metrics
    geometry_metrics = compute_geometry_metrics(
        radar_xyz, reflector_xyz, reflector_names, reflector_roles,
        motion_directions_for_metrics, wavelength_m, config
    )
    
    print("Writing geometry_metrics.csv...")
    geom_csv_path = outdir_path / 'geometry_metrics.csv'
    write_geometry_metrics_csv(geometry_metrics, geom_csv_path)
    generated_files.append('geometry_metrics.csv')
    
    print("Writing geometry_metrics.json...")
    geom_json_path = outdir_path / 'geometry_metrics.json'
    write_geometry_metrics_json(geometry_metrics, geom_json_path)
    generated_files.append('geometry_metrics.json')
    
    # Generate 2D engineering views
    print("Generating scene_plan_view.png...")
    plan_view_path = outdir_path / 'scene_plan_view.png'
    plot_plan_view(
        radar_xyz, reflector_xyz, reflector_names, reflector_roles,
        motion_vectors=np.array(motion_vectors), output_path=plan_view_path
    )
    generated_files.append('scene_plan_view.png')
    
    print("Generating scene_elevation_view.png...")
    elevation_view_path = outdir_path / 'scene_elevation_view.png'
    plot_elevation_view(
        radar_xyz, reflector_xyz, reflector_xyz_displaced,
        reflector_names, reflector_roles, output_path=elevation_view_path
    )
    generated_files.append('scene_elevation_view.png')
    
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
    
    # Generate alarms
    print("Generating alarms...")
    alarms = generate_alarms(data, config)
    print(f"Generated {len(alarms)} alarms")
    
    # Apply alarm latching and acknowledgments
    print("Processing alarm states (latching/ack)...")
    ack_file = outdir_path / 'alarm_ack.json'
    alarms, alarm_state = apply_alarm_latching_and_acks(alarms, config, data, ack_file)
    
    # Generate ack template if no ack file exists
    if not ack_file.exists():
        ack_template_path = outdir_path / 'alarm_ack_template.json'
        generate_ack_template(alarms, ack_template_path)
        generated_files.append('alarm_ack_template.json')
    
    # Write alarm state JSON
    print("Writing alarm_state.json...")
    alarm_state_path = outdir_path / 'alarm_state.json'
    write_alarm_state_json(alarm_state, alarm_state_path)
    generated_files.append('alarm_state.json')
    
    # Write alarm log CSV
    print("Writing alarm_log.csv...")
    alarm_log_path = outdir_path / 'alarm_log.csv'
    write_alarm_log_csv(alarms, alarm_log_path)
    generated_files.append('alarm_log.csv')
    
    # Write SCADA telemetry
    print("Writing scada_telemetry.csv...")
    scada_path = outdir_path / 'scada_telemetry.csv'
    write_scada_telemetry(data, config, alarms, run_id, scada_path)
    generated_files.append('scada_telemetry.csv')
    
    # Plot alarm timeline
    print("Generating alarm_timeline.png...")
    alarm_timeline_path = outdir_path / 'alarm_timeline.png'
    plot_alarm_timeline(data, config, alarms, alarm_timeline_path)
    generated_files.append('alarm_timeline.png')
    
    # Render HMI dashboard
    print("Generating hmi_station.png...")
    hmi_path = outdir_path / 'hmi_station.png'
    render_hmi_dashboard(data, config, alarms, run_id, hmi_path, geometry_metrics, alarm_state)
    generated_files.append('hmi_station.png')
    
    # Generate results.md (primary human-readable output)
    print("Generating results.md...")
    generate_results_md(config, data, alarms, outdir_path, run_id, geometry_metrics, alarm_state)
    generated_files.append('results.md')
    
    # Generate report (legacy)
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
        help="Path to JSON config file (for 3D mode: requires --outdir)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Output directory (for 3D mode: requires --config)"
    )
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        # Check if using new 3D mode (both --config and --outdir required)
        if args.config and args.outdir:
            return simulate_3d_main(args.config, args.outdir)
        elif args.config or args.outdir:
            # Error: both flags must be provided together
            print("Error: 3D mode requires both --config and --outdir flags", file=sys.stderr)
            return 1
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
