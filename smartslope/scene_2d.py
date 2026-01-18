"""2D engineering views for radar installation geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from smartslope.math_utils import unit


def plot_plan_view(
    radar_xyz: np.ndarray,
    reflector_xyz: np.ndarray,
    reflector_names: List[str],
    reflector_roles: List[str],
    motion_vectors: Optional[np.ndarray] = None,
    output_path: Path = None,
    title: str = "Plan View (X-Y)"
) -> None:
    """
    Plot plan view (top-down, X-Y projection) of radar installation.
    
    Args:
        radar_xyz: Radar position (3,)
        reflector_xyz: Reflector positions (N, 3)
        reflector_names: List of reflector names
        reflector_roles: List of reflector roles ("ref" or "slope")
        motion_vectors: Optional motion direction vectors for slope targets (N, 3)
        output_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot radar
    ax.scatter(radar_xyz[0], radar_xyz[1], c='red', marker='^', s=300, 
               label='Radar', edgecolors='black', linewidths=2, zorder=10)
    ax.text(radar_xyz[0], radar_xyz[1] + 5, 'RADAR', 
            fontsize=11, weight='bold', ha='center', va='bottom')
    
    # Plot reflectors
    ref_indices = [i for i, r in enumerate(reflector_roles) if r == "ref"]
    slope_indices = [i for i, r in enumerate(reflector_roles) if r == "slope"]
    
    if ref_indices:
        ref_xy = reflector_xyz[ref_indices, :2]
        ax.scatter(ref_xy[:, 0], ref_xy[:, 1], 
                   c='blue', marker='o', s=150, label='Reference', 
                   alpha=0.8, edgecolors='black', linewidths=1, zorder=8)
        for i in ref_indices:
            ax.text(reflector_xyz[i, 0], reflector_xyz[i, 1] + 3,
                   reflector_names[i], fontsize=9, color='blue', 
                   ha='center', va='bottom', weight='bold')
    
    if slope_indices:
        slope_xy = reflector_xyz[slope_indices, :2]
        ax.scatter(slope_xy[:, 0], slope_xy[:, 1],
                   c='orange', marker='s', s=150, label='Slope', 
                   alpha=0.8, edgecolors='black', linewidths=1, zorder=8)
        for i in slope_indices:
            ax.text(reflector_xyz[i, 0], reflector_xyz[i, 1] + 3,
                   reflector_names[i], fontsize=9, color='orange',
                   ha='center', va='bottom', weight='bold')
    
    # Draw LOS lines from radar to reflectors
    for i in range(len(reflector_names)):
        ax.plot([radar_xyz[0], reflector_xyz[i, 0]],
               [radar_xyz[1], reflector_xyz[i, 1]],
               'k--', alpha=0.3, linewidth=1, zorder=1)
    
    # Plot motion vectors for slope targets (projected to XY plane)
    if motion_vectors is not None:
        for i in slope_indices:
            if np.linalg.norm(motion_vectors[i]) > 1e-10:
                # Project motion vector to XY plane
                motion_xy = motion_vectors[i, :2]
                if np.linalg.norm(motion_xy) > 1e-10:
                    # Scale for visibility (30m length)
                    vec_scaled = unit(motion_xy) * 30
                    ax.arrow(reflector_xyz[i, 0], reflector_xyz[i, 1],
                            vec_scaled[0], vec_scaled[1],
                            color='red', width=2, head_width=8, head_length=6,
                            alpha=0.7, zorder=9, length_includes_head=True)
    
    ax.set_xlabel('X - East (m)', fontsize=12, weight='bold')
    ax.set_ylabel('Y - North (m)', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Add compass rose
    compass_x = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.9
    compass_y = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.9
    ax.annotate('N', xy=(compass_x, compass_y + 15), fontsize=10, weight='bold', ha='center')
    ax.arrow(compass_x, compass_y, 0, 10, head_width=3, head_length=2, fc='black', ec='black')
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Wrote {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_elevation_view(
    radar_xyz: np.ndarray,
    reflector_xyz: np.ndarray,
    reflector_xyz_displaced: Optional[np.ndarray],
    reflector_names: List[str],
    reflector_roles: List[str],
    output_path: Path = None,
    title: str = "Elevation View (Distance vs Height)"
) -> None:
    """
    Plot elevation view showing distance from radar vs height.
    
    This creates a side-view showing the radar height and reflector heights
    as a function of horizontal distance from radar.
    
    Args:
        radar_xyz: Radar position (3,)
        reflector_xyz: Reflector positions (N, 3)
        reflector_xyz_displaced: Optional displaced positions for before/after view (N, 3)
        reflector_names: List of reflector names
        reflector_roles: List of reflector roles
        output_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Compute horizontal distance from radar (in XY plane)
    n_reflectors = len(reflector_names)
    distances = np.zeros(n_reflectors)
    heights = reflector_xyz[:, 2]  # Z coordinate
    
    for i in range(n_reflectors):
        xy_dist = np.sqrt((reflector_xyz[i, 0] - radar_xyz[0])**2 + 
                         (reflector_xyz[i, 1] - radar_xyz[1])**2)
        distances[i] = xy_dist
    
    # Plot radar
    ax.scatter(0, radar_xyz[2], c='red', marker='^', s=300, 
               label='Radar', edgecolors='black', linewidths=2, zorder=10)
    ax.text(0, radar_xyz[2] + 2, 'RADAR', 
            fontsize=11, weight='bold', ha='center', va='bottom')
    
    # Plot reflectors (original positions)
    ref_indices = [i for i, r in enumerate(reflector_roles) if r == "ref"]
    slope_indices = [i for i, r in enumerate(reflector_roles) if r == "slope"]
    
    if ref_indices:
        ref_dist = distances[ref_indices]
        ref_height = heights[ref_indices]
        ax.scatter(ref_dist, ref_height, 
                   c='blue', marker='o', s=150, label='Reference', 
                   alpha=0.8, edgecolors='black', linewidths=1, zorder=8)
        for i in ref_indices:
            ax.text(distances[i], heights[i] + 2,
                   reflector_names[i], fontsize=9, color='blue', 
                   ha='center', va='bottom', weight='bold')
    
    if slope_indices:
        slope_dist = distances[slope_indices]
        slope_height = heights[slope_indices]
        ax.scatter(slope_dist, slope_height,
                   c='orange', marker='s', s=150, label='Slope', 
                   alpha=0.8, edgecolors='black', linewidths=1, zorder=8)
        for i in slope_indices:
            ax.text(distances[i], heights[i] + 2,
                   reflector_names[i], fontsize=9, color='orange',
                   ha='center', va='bottom', weight='bold')
    
    # Draw LOS lines from radar to reflectors
    for i in range(n_reflectors):
        ax.plot([0, distances[i]], [radar_xyz[2], heights[i]],
               'k--', alpha=0.3, linewidth=1, zorder=1)
    
    # If displaced positions provided, show before/after
    if reflector_xyz_displaced is not None:
        heights_displaced = reflector_xyz_displaced[:, 2]
        
        # Compute displaced distances
        distances_displaced = np.zeros(n_reflectors)
        for i in range(n_reflectors):
            xy_dist = np.sqrt((reflector_xyz_displaced[i, 0] - radar_xyz[0])**2 + 
                             (reflector_xyz_displaced[i, 1] - radar_xyz[1])**2)
            distances_displaced[i] = xy_dist
        
        # Plot displaced positions with different style
        for i in range(n_reflectors):
            if np.linalg.norm(reflector_xyz[i] - reflector_xyz_displaced[i]) > 1e-6:
                color = 'blue' if reflector_roles[i] == 'ref' else 'orange'
                marker = 'o' if reflector_roles[i] == 'ref' else 's'
                
                # Draw displacement vector
                ax.plot([distances[i], distances_displaced[i]],
                       [heights[i], heights_displaced[i]],
                       'g-', linewidth=2, alpha=0.7, zorder=7)
                
                # Plot displaced position
                ax.scatter(distances_displaced[i], heights_displaced[i],
                          c=color, marker=marker, s=100, alpha=0.5, 
                          edgecolors='green', linewidths=2, zorder=7)
    
    ax.set_xlabel('Horizontal Distance from Radar (m)', fontsize=12, weight='bold')
    ax.set_ylabel('Height / Elevation (m)', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add ground reference line at z=0
    ax.axhline(y=0, color='brown', linestyle='-', linewidth=2, alpha=0.3, label='Ground Level')
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Wrote {output_path}")
    else:
        plt.show()
    
    plt.close()
