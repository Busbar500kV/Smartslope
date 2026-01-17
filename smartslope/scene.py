"""3D scene geometry computations and visualization for radar installations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from smartslope.math_utils import unit


def compute_los_vector(radar_xyz: np.ndarray, reflector_xyz: np.ndarray) -> np.ndarray:
    """
    Compute line-of-sight unit vector from radar to reflector.
    
    Args:
        radar_xyz: Radar position (3,)
        reflector_xyz: Reflector position (3,)
    
    Returns:
        LOS unit vector (3,)
    """
    delta = reflector_xyz - radar_xyz
    return unit(delta)


def compute_range(radar_xyz: np.ndarray, reflector_xyz: np.ndarray) -> float:
    """
    Compute range (distance) from radar to reflector.
    
    Args:
        radar_xyz: Radar position (3,)
        reflector_xyz: Reflector position (3,)
    
    Returns:
        Range in meters
    """
    return float(np.linalg.norm(reflector_xyz - radar_xyz))


def compute_incidence_angle(los: np.ndarray, surface_normal: np.ndarray = None) -> float:
    """
    Compute incidence angle between LOS and surface normal.
    
    Args:
        los: LOS unit vector (3,)
        surface_normal: Surface normal unit vector (3,), defaults to vertical [0,0,1]
    
    Returns:
        Incidence angle in degrees
    """
    if surface_normal is None:
        surface_normal = np.array([0.0, 0.0, 1.0])
    
    cos_angle = np.dot(-los, surface_normal)  # negative because LOS points away from radar
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(angle_rad))


def plot_scene_3d(
    radar_xyz: np.ndarray,
    reflector_xyz: np.ndarray,
    reflector_names: List[str],
    reflector_roles: List[str],
    motion_vectors: np.ndarray = None,
    output_path: Path = None,
    title: str = "3D Radar Installation Scene"
) -> None:
    """
    Plot 3D scene showing radar and reflectors.
    
    Args:
        radar_xyz: Radar position (3,)
        reflector_xyz: Reflector positions (N, 3)
        reflector_names: List of reflector names
        reflector_roles: List of reflector roles ("ref" or "slope")
        motion_vectors: Optional motion direction vectors for slope targets (N, 3)
        output_path: Path to save plot
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot radar
    ax.scatter(*radar_xyz, c='red', marker='^', s=200, label='Radar', edgecolors='black', linewidths=2)
    ax.text(radar_xyz[0], radar_xyz[1], radar_xyz[2] + 5, 'RADAR', fontsize=10, weight='bold')
    
    # Plot reflectors
    ref_indices = [i for i, r in enumerate(reflector_roles) if r == "ref"]
    slope_indices = [i for i, r in enumerate(reflector_roles) if r == "slope"]
    
    if ref_indices:
        ref_xyz = reflector_xyz[ref_indices]
        ax.scatter(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2], 
                   c='blue', marker='o', s=100, label='Reference', alpha=0.8)
        for i in ref_indices:
            ax.text(reflector_xyz[i, 0], reflector_xyz[i, 1], reflector_xyz[i, 2] + 3,
                   reflector_names[i], fontsize=8, color='blue')
    
    if slope_indices:
        slope_xyz = reflector_xyz[slope_indices]
        ax.scatter(slope_xyz[:, 0], slope_xyz[:, 1], slope_xyz[:, 2],
                   c='orange', marker='s', s=100, label='Slope', alpha=0.8)
        for i in slope_indices:
            ax.text(reflector_xyz[i, 0], reflector_xyz[i, 1], reflector_xyz[i, 2] + 3,
                   reflector_names[i], fontsize=8, color='orange')
    
    # Plot motion vectors for slope targets
    if motion_vectors is not None:
        for i in slope_indices:
            # Check if motion vector is non-zero using norm for efficiency
            if np.linalg.norm(motion_vectors[i]) > 1e-10:
                # Scale vector for visibility (50m length)
                vec_scaled = motion_vectors[i] * 50
                ax.quiver(reflector_xyz[i, 0], reflector_xyz[i, 1], reflector_xyz[i, 2],
                         vec_scaled[0], vec_scaled[1], vec_scaled[2],
                         color='red', arrow_length_ratio=0.2, linewidth=2, alpha=0.7)
    
    # Draw LOS lines from radar to reflectors
    for i in range(len(reflector_names)):
        ax.plot([radar_xyz[0], reflector_xyz[i, 0]],
               [radar_xyz[1], reflector_xyz[i, 1]],
               [radar_xyz[2], reflector_xyz[i, 2]],
               'k--', alpha=0.3, linewidth=0.5)
    
    ax.set_xlabel('X (East) [m]')
    ax.set_ylabel('Y (North) [m]')
    ax.set_zlabel('Z (Up) [m]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_range = np.array([
        reflector_xyz[:, 0].max() - reflector_xyz[:, 0].min(),
        reflector_xyz[:, 1].max() - reflector_xyz[:, 1].min(),
        reflector_xyz[:, 2].max() - reflector_xyz[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (reflector_xyz[:, 0].max() + reflector_xyz[:, 0].min()) * 0.5
    mid_y = (reflector_xyz[:, 1].max() + reflector_xyz[:, 1].min()) * 0.5
    mid_z = (reflector_xyz[:, 2].max() + reflector_xyz[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Wrote {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_scene_before_after(
    radar_xyz: np.ndarray,
    reflector_xyz: np.ndarray,
    reflector_xyz_displaced: np.ndarray,
    reflector_names: List[str],
    reflector_roles: List[str],
    output_path: Path = None,
    title: str = "3D Scene: Before vs After Displacement"
) -> None:
    """
    Plot before/after scene showing original and displaced positions.
    
    Args:
        radar_xyz: Radar position (3,)
        reflector_xyz: Original reflector positions (N, 3)
        reflector_xyz_displaced: Displaced reflector positions (N, 3)
        reflector_names: List of reflector names
        reflector_roles: List of reflector roles
        output_path: Path to save plot
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot radar
    ax.scatter(*radar_xyz, c='red', marker='^', s=200, label='Radar', edgecolors='black', linewidths=2)
    
    # Plot original and displaced positions
    for i, name in enumerate(reflector_names):
        color = 'blue' if reflector_roles[i] == 'ref' else 'orange'
        marker = 'o' if reflector_roles[i] == 'ref' else 's'
        
        # Original position
        ax.scatter(*reflector_xyz[i], c=color, marker=marker, s=100, alpha=0.3)
        
        # Displaced position
        ax.scatter(*reflector_xyz_displaced[i], c=color, marker=marker, s=100, alpha=1.0)
        
        # Draw displacement vector
        if np.linalg.norm(reflector_xyz[i] - reflector_xyz_displaced[i]) > 1e-10:
            ax.plot([reflector_xyz[i, 0], reflector_xyz_displaced[i, 0]],
                   [reflector_xyz[i, 1], reflector_xyz_displaced[i, 1]],
                   [reflector_xyz[i, 2], reflector_xyz_displaced[i, 2]],
                   'g-', linewidth=2, alpha=0.7)
            # Add arrow head
            disp_vec = reflector_xyz_displaced[i] - reflector_xyz[i]
            ax.quiver(reflector_xyz[i, 0], reflector_xyz[i, 1], reflector_xyz[i, 2],
                     disp_vec[0], disp_vec[1], disp_vec[2],
                     color='green', arrow_length_ratio=0.3, linewidth=2, alpha=0.7)
        
        # Label
        ax.text(reflector_xyz_displaced[i, 0], reflector_xyz_displaced[i, 1], 
               reflector_xyz_displaced[i, 2] + 3, name, fontsize=8, color=color)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Radar'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Reference'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='Slope'),
        Line2D([0], [0], color='green', linewidth=2, label='Displacement')
    ]
    ax.legend(handles=legend_elements)
    
    ax.set_xlabel('X (East) [m]')
    ax.set_ylabel('Y (North) [m]')
    ax.set_zlabel('Z (Up) [m]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Wrote {output_path}")
    else:
        plt.show()
    
    plt.close()
