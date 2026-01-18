"""Digital Elevation Model (DEM) support for terrain occlusion analysis.

Lightweight DEM module supporting NPZ format with bilinear interpolation
and terrain occlusion detection for line-of-sight calculations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def load_dem_npz(path: Path) -> dict:
    """Load DEM from NPZ file.
    
    Expected keys:
        dem_x_m: (Nx,) float - X coordinates in meters
        dem_y_m: (Ny,) float - Y coordinates in meters
        dem_z_m: (Ny, Nx) float - Elevation (Z) in meters
    
    Args:
        path: Path to NPZ file
    
    Returns:
        Dictionary with dem_x_m, dem_y_m, dem_z_m arrays
    """
    data = np.load(path)
    
    required_keys = ['dem_x_m', 'dem_y_m', 'dem_z_m']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"DEM NPZ missing required key: {key}")
    
    dem_x = data['dem_x_m']
    dem_y = data['dem_y_m']
    dem_z = data['dem_z_m']
    
    # Validate shapes
    if dem_z.ndim != 2:
        raise ValueError(f"dem_z_m must be 2D, got shape {dem_z.shape}")
    
    ny, nx = dem_z.shape
    if len(dem_x) != nx:
        raise ValueError(f"dem_x_m length {len(dem_x)} != dem_z width {nx}")
    if len(dem_y) != ny:
        raise ValueError(f"dem_y_m length {len(dem_y)} != dem_z height {ny}")
    
    return {
        'dem_x_m': dem_x,
        'dem_y_m': dem_y,
        'dem_z_m': dem_z
    }


def sample_dem_z(dem: dict, x: float, y: float) -> float:
    """Sample DEM elevation at (x, y) using bilinear interpolation.
    
    Args:
        dem: DEM dict from load_dem_npz
        x: X coordinate in meters
        y: Y coordinate in meters
    
    Returns:
        Interpolated elevation in meters
        Returns NaN if point is outside DEM bounds
    """
    dem_x = dem['dem_x_m']
    dem_y = dem['dem_y_m']
    dem_z = dem['dem_z_m']
    
    # Check bounds
    if x < dem_x[0] or x > dem_x[-1] or y < dem_y[0] or y > dem_y[-1]:
        return np.nan
    
    # Find grid cell
    ix = np.searchsorted(dem_x, x) - 1
    iy = np.searchsorted(dem_y, y) - 1
    
    # Clamp to valid range
    ix = max(0, min(ix, len(dem_x) - 2))
    iy = max(0, min(iy, len(dem_y) - 2))
    
    # Bilinear interpolation weights
    x0, x1 = dem_x[ix], dem_x[ix + 1]
    y0, y1 = dem_y[iy], dem_y[iy + 1]
    
    wx = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
    wy = (y - y0) / (y1 - y0) if y1 > y0 else 0.0
    
    # Four corner elevations
    z00 = dem_z[iy, ix]
    z10 = dem_z[iy, ix + 1]
    z01 = dem_z[iy + 1, ix]
    z11 = dem_z[iy + 1, ix + 1]
    
    # Bilinear interpolation
    z0 = z00 * (1 - wx) + z10 * wx
    z1 = z01 * (1 - wx) + z11 * wx
    z = z0 * (1 - wy) + z1 * wy
    
    return z


def generate_synthetic_dem(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    resolution_m: float = 10.0,
    slope_x: float = 0.0,
    slope_y: float = 0.0,
    base_elevation: float = 0.0,
    bumps: Optional[list] = None
) -> dict:
    """Generate a synthetic DEM for testing and demos.
    
    Args:
        x_min, x_max, y_min, y_max: DEM bounds in meters
        resolution_m: Grid resolution in meters
        slope_x: Linear slope in X direction (dz/dx)
        slope_y: Linear slope in Y direction (dz/dy)
        base_elevation: Base elevation in meters
        bumps: Optional list of bump dicts with keys:
            - center_x, center_y: Bump center position
            - radius_m: Bump radius
            - height_m: Bump height
    
    Returns:
        DEM dict compatible with load_dem_npz
    """
    # Generate grid
    x = np.arange(x_min, x_max + resolution_m, resolution_m)
    y = np.arange(y_min, y_max + resolution_m, resolution_m)
    
    xx, yy = np.meshgrid(x, y)
    
    # Start with planar slope
    z = base_elevation + slope_x * xx + slope_y * yy
    
    # Add bumps if requested
    if bumps:
        for bump in bumps:
            cx = bump['center_x']
            cy = bump['center_y']
            radius = bump['radius_m']
            height = bump['height_m']
            
            # Gaussian bump
            dist_sq = (xx - cx)**2 + (yy - cy)**2
            bump_z = height * np.exp(-dist_sq / (2 * (radius / 2.5)**2))
            z += bump_z
    
    return {
        'dem_x_m': x,
        'dem_y_m': y,
        'dem_z_m': z
    }


def check_los_occlusion(
    dem: dict,
    radar_xyz: np.ndarray,
    reflector_xyz: np.ndarray,
    n_samples: int = 200
) -> Tuple[bool, float]:
    """Check if line-of-sight from radar to reflector is occluded by terrain.
    
    Args:
        dem: DEM dict from load_dem_npz
        radar_xyz: Radar position (3,) in meters
        reflector_xyz: Reflector position (3,) in meters
        n_samples: Number of samples along the ray
    
    Returns:
        (is_occluded, fraction_blocked)
        is_occluded: True if any ray sample is below terrain
        fraction_blocked: Fraction of ray samples below terrain (0.0 to 1.0)
    """
    # Sample points along line-of-sight
    t = np.linspace(0, 1, n_samples)
    ray_x = radar_xyz[0] + t * (reflector_xyz[0] - radar_xyz[0])
    ray_y = radar_xyz[1] + t * (reflector_xyz[1] - radar_xyz[1])
    ray_z = radar_xyz[2] + t * (reflector_xyz[2] - radar_xyz[2])
    
    # Sample DEM elevation at each ray point
    blocked_count = 0
    for i in range(n_samples):
        terrain_z = sample_dem_z(dem, ray_x[i], ray_y[i])
        
        # Skip if outside DEM bounds
        if np.isnan(terrain_z):
            continue
        
        # Check if ray is below terrain
        if ray_z[i] < terrain_z:
            blocked_count += 1
    
    fraction_blocked = blocked_count / n_samples
    is_occluded = blocked_count > 0
    
    return is_occluded, fraction_blocked


def compute_height_above_ground(
    dem: dict,
    reflector_xyz: np.ndarray
) -> float:
    """Compute reflector height above ground using DEM.
    
    Args:
        dem: DEM dict from load_dem_npz
        reflector_xyz: Reflector position (3,) in meters
    
    Returns:
        Height above ground in meters
        Returns NaN if reflector is outside DEM bounds
    """
    terrain_z = sample_dem_z(dem, reflector_xyz[0], reflector_xyz[1])
    
    if np.isnan(terrain_z):
        return np.nan
    
    return reflector_xyz[2] - terrain_z
