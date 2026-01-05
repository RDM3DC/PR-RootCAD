from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class PRFieldConfig:
    """Configuration for a grid-based Phase‑Resolved field."""

    size: int = 64
    steps: int = 200
    dt: float = 0.15
    diffusion: float = 0.35
    coupling: float = 0.25
    coupling_mode: Literal["none", "geom_target"] = "geom_target"
    phase_dim: int = 2
    phase_space: Literal["unwrapped", "wrapped"] = "unwrapped"
    seed: int | None = 0


@dataclass
class PRFieldState:
    """State for a grid-based Phase‑Resolved field."""

    phi: np.ndarray  # (N,N,phase_dim)
    geom: np.ndarray  # (N,N) scalar geometry proxy
    falsifier_residual: np.ndarray | None = None  # (N,N) plaquette holonomy/branch residual


__all__ = ["PRFieldConfig", "PRFieldState"]
