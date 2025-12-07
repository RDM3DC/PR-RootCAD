"""
AdaptiveCAD πₐ kernel
- Single source of truth for Adaptive-π (πₐ) geometry across plugins and tools.
- Treat πₐ as a slowly varying function of local curvature and scale.

Public API:
- PiAParams: dataclass with (beta, s0, clamp)
- pi_a(kappa, scale, params) -> float
- adaptive_arc_length(radius, angle_rad, kappa, scale, params) -> float
- make_adaptive_circle(radius, n, kappa, scale, params) -> np.ndarray[(n,2)]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["PiAParams", "pi_a", "adaptive_arc_length", "make_adaptive_circle", "__version__"]
__version__ = "0.1.0"


@dataclass
class PiAParams:
    """
    Parameters for an example πₐ model.
    beta: curvature sensitivity
    s0:   reference scale (meters) separating macro/micro regimes
    clamp: max fractional deviation from π (>=0), keeps model bounded
    """

    beta: float = 0.2
    s0: float = 1.0
    clamp: float = 0.3


def _clamp(x: float, lo: float, hi: float) -> float:
    return hi if x > hi else lo if x < lo else x


def pi_a(kappa: float, scale: float = 1.0, params: PiAParams = PiAParams()) -> float:
    """
    Adaptive π model (baseline exemplar):
        πₐ = π * (1 + beta * (kappa * scale / s0)^2), clamped to ±clamp.
    - kappa: curvature [1/m]
    - scale: characteristic length [m]
    """
    base = math.pi
    s0 = max(params.s0, 1e-9)
    frac = params.beta * (kappa * (scale / s0)) ** 2
    frac = _clamp(frac, -params.clamp, params.clamp)
    return base * (1.0 + frac)


def adaptive_arc_length(
    radius: float, angle_rad: float, kappa: float, scale: float, params: PiAParams = PiAParams()
) -> float:
    """Arc length with πₐ scaling: for small angles, L = angle * r; we modulate by πₐ/π."""
    pa = pi_a(kappa, scale, params)
    return float(angle_rad * radius * (pa / math.pi))


def make_adaptive_circle(
    radius: float,
    n: int = 128,
    kappa: float = 0.0,
    scale: float = 1.0,
    params: PiAParams = PiAParams(),
) -> np.ndarray:
    """
    Generate a polyline approximating a circle with adaptive π scaling (uniform over the loop).
    Returns Nx2 numpy array in meters.
    """
    pts = np.zeros((n, 2), dtype=float)
    pa = pi_a(kappa, scale, params)
    # scale the full angle 2π by pa/π so circumference stretches/shrinks coherently
    total_angle = 2.0 * math.pi * (pa / math.pi)
    for i in range(n):
        theta = total_angle * (i / float(n))
        pts[i, 0] = radius * math.cos(theta)
        pts[i, 1] = radius * math.sin(theta)
    return pts
