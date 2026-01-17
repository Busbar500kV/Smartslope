from __future__ import annotations
import numpy as np

def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v

def wrap_pi(phi: np.ndarray) -> np.ndarray:
    return (phi + np.pi) % (2 * np.pi) - np.pi

def nanmedian_1d(x: np.ndarray) -> float:
    if np.isfinite(x).any():
        return float(np.nanmedian(x))
    return float("nan")