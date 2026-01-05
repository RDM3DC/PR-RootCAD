from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Tuple

import numpy as np

from .types import PRFieldConfig, PRFieldState


def _wrap_to_pi(delta: np.ndarray) -> np.ndarray:
    """Map values to (-pi, pi] elementwise.

    This is the MVP operationalization of PR-Root branch bookkeeping:
    angle differences must choose a consistent principal branch.
    """

    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _laplacian_2d(arr: np.ndarray) -> np.ndarray:
    return (
        np.roll(arr, 1, axis=0)
        + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1)
        + np.roll(arr, -1, axis=1)
        - 4.0 * arr
    )


def _plaquette_falsifier_residual(phi: np.ndarray, *, phase_space: str) -> np.ndarray:
    """Compute a plaquette loop-closure residual for phase branch consistency.

    For each cell, we compute the signed sum of edge differences around a
    unit plaquette. In a consistent (Abelian / branch-consistent) field this
    should be ~0; wrapped spaces can exhibit non-zero residuals near branch
    cuts / topological defects.

    Returns:
        (N,N) float residual magnitude.
    """

    # Forward differences (periodic).
    dx = phi - np.roll(phi, -1, axis=1)
    dy = phi - np.roll(phi, -1, axis=0)
    if phase_space == "wrapped":
        dx = _wrap_to_pi(dx)
        dy = _wrap_to_pi(dy)

    # Plaquette sum: dx(i,j) + dy(i,j+1) - dx(i+1,j) - dy(i,j)
    dy_x = np.roll(dy, -1, axis=1)
    dx_y = np.roll(dx, -1, axis=0)
    loop = dx + dy_x - dx_y - dy
    if phase_space == "wrapped":
        loop = _wrap_to_pi(loop)

    if loop.ndim == 2:
        return np.abs(loop).astype(np.float32)
    return np.linalg.norm(loop, axis=2).astype(np.float32)


def _phase_smoothness_energy(phi: np.ndarray, *, phase_space: str) -> float:
    dx = phi - np.roll(phi, -1, axis=1)
    dy = phi - np.roll(phi, -1, axis=0)
    if phase_space == "wrapped":
        dx = _wrap_to_pi(dx)
        dy = _wrap_to_pi(dy)
    return float(np.mean(dx * dx) + np.mean(dy * dy))


def _coupling_target(geom: np.ndarray, *, phase_dim: int, phase_space: str) -> np.ndarray:
    if phase_space == "wrapped":
        # Map geom to a bounded angle target in (-pi, pi].
        theta = np.pi * np.tanh(geom)
        if phase_dim == 1:
            return theta[:, :, None]
        target = np.zeros((geom.shape[0], geom.shape[1], phase_dim), dtype=float)
        target[:, :, 0] = theta
        return target

    # Unwrapped: track geom in the first component.
    target = np.zeros((geom.shape[0], geom.shape[1], phase_dim), dtype=float)
    target[:, :, 0] = geom
    return target


def _coupling_energy(phi: np.ndarray, target: np.ndarray, *, phase_space: str) -> float:
    diff = phi - target
    if phase_space == "wrapped":
        diff = _wrap_to_pi(diff)
    return float(np.mean(diff * diff))


def _coherence_metric(phi: np.ndarray) -> float:
    # A simple bounded coherence proxy: higher when gradients are small.
    # Default assumes unwrapped, but callers should prefer the new energy-based metrics.
    e = float(np.mean(phi * phi))
    return float(1.0 / (1.0 + e))


def _make_geom_proxy(N: int, rng: np.random.Generator) -> np.ndarray:
    # A smooth-ish scalar field: low-frequency noise via repeated averaging.
    g = rng.standard_normal((N, N)).astype(float)
    for _ in range(6):
        g = 0.5 * g + 0.5 * (
            np.roll(g, 1, axis=0)
            + np.roll(g, -1, axis=0)
            + np.roll(g, 1, axis=1)
            + np.roll(g, -1, axis=1)
        ) / 4.0
    g -= float(np.mean(g))
    g /= float(np.std(g) + 1e-9)
    return g


def relax_phase_field(cfg: PRFieldConfig) -> Tuple[Dict[str, Any], PRFieldState]:
    """Relax a phase field on a grid and return metrics + final state.

    This implements the math-complete MVP from `PHASE_RESOLVED_MODELING.md`.

    Phase spaces:
    - unwrapped: phi in R^n
    - wrapped:   phi in (S^1)^n, with principal-branch wrap-to-pi differences

    Energies:
    - E_phase: Dirichlet smoothness using wrapped/unwrapped differences
    - E_couple (optional): squared deviation from a target T(geom)

    Update rule (relaxation / descent):
    - diffusion step:   phi <- phi + dt * diffusion * Laplacian(phi)
    - coupling step:    phi <- phi - dt * coupling * (phi - T(geom))
      (wrapped mode uses wrap(phi - T) as the descent direction)
    """

    N = int(cfg.size)
    if N < 8:
        raise ValueError("size must be >= 8")

    phase_dim = int(cfg.phase_dim)
    if phase_dim < 1 or phase_dim > 8:
        raise ValueError("phase_dim must be in [1,8]")

    phase_space = str(cfg.phase_space)
    if phase_space not in {"unwrapped", "wrapped"}:
        raise ValueError("phase_space must be 'unwrapped' or 'wrapped'")

    rng = np.random.default_rng(cfg.seed)

    phi = rng.standard_normal((N, N, phase_dim)).astype(float)
    if phase_space == "wrapped":
        phi = _wrap_to_pi(phi)
    geom = _make_geom_proxy(N, rng)

    coupling_mode = str(cfg.coupling_mode)
    if coupling_mode not in {"none", "geom_target"}:
        raise ValueError("coupling_mode must be 'none' or 'geom_target'")

    target = _coupling_target(geom, phase_dim=phase_dim, phase_space=phase_space)

    history_phase_energy: list[float] = []
    history_coupling_energy: list[float] = []
    history_coherence: list[float] = []

    dt = float(cfg.dt)
    diffusion = float(cfg.diffusion)
    coupling = float(cfg.coupling)

    for _ in range(int(cfg.steps)):
        lap = _laplacian_2d(phi)

        # Diffusion step (minimizes Dirichlet energy)
        phi = phi + dt * (diffusion * lap)

        # Coupling-to-geometry target (optional)
        if coupling_mode != "none" and coupling != 0.0:
            diff = phi - target
            if phase_space == "wrapped":
                diff = _wrap_to_pi(diff)
            phi = phi - dt * (coupling * diff)

        if phase_space == "wrapped":
            phi = _wrap_to_pi(phi)

        e_phase = _phase_smoothness_energy(phi, phase_space=phase_space)
        history_phase_energy.append(e_phase)

        e_couple = 0.0
        if coupling_mode != "none" and coupling != 0.0:
            e_couple = _coupling_energy(phi, target, phase_space=phase_space)
        history_coupling_energy.append(float(e_couple))

        history_coherence.append(float(1.0 / (1.0 + e_phase)))

    falsifier = _plaquette_falsifier_residual(phi, phase_space=phase_space)

    metrics: Dict[str, Any] = {
        "config": asdict(cfg),
        "phase_energy": float(
            history_phase_energy[-1]
            if history_phase_energy
            else _phase_smoothness_energy(phi, phase_space=phase_space)
        ),
        "coupling_energy": float(
            history_coupling_energy[-1]
            if history_coupling_energy
            else _coupling_energy(phi, target, phase_space=phase_space)
        ),
        "total_energy": float(
            (history_phase_energy[-1] if history_phase_energy else 0.0)
            + (history_coupling_energy[-1] if history_coupling_energy else 0.0)
        ),
        "coherence": float(history_coherence[-1] if history_coherence else 1.0 / (1.0 + 0.0)),
        "phi_mean": float(np.mean(phi)),
        "phi_std": float(np.std(phi)),
        "geom_mean": float(np.mean(geom)),
        "geom_std": float(np.std(geom)),
        "falsifier_mean": float(np.mean(falsifier)),
        "falsifier_max": float(np.max(falsifier)),
        "history": {
            "phase_energy": history_phase_energy,
            "coupling_energy": history_coupling_energy,
            "coherence": history_coherence,
        },
    }

    return metrics, PRFieldState(phi=phi, geom=geom, falsifier_residual=falsifier)


__all__ = ["relax_phase_field"]
