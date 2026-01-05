"""
PR Volume — 3D phase-resolved voxel grid with diffusion.

Extends the 2D PR field to a full volumetric domain:
  - 3D Laplacian diffusion on phase field φ(x,y,z)
  - Optional geometry proxy coupling (3D noise target)
  - Isosurface extraction via marching cubes
  - Branch-safe wrapped or unwrapped phase space
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class PRVolumeConfig:
    """Configuration for a 3D Phase-Resolved voxel field."""
    
    size: int = 32  # NxNxN grid
    steps: int = 100
    dt: float = 0.15
    diffusion: float = 0.35
    coupling: float = 0.20
    coupling_mode: Literal["none", "geom_target"] = "geom_target"
    
    # Geometry proxy synthesis
    geom_mode: Literal["smooth_noise", "spectral_band", "torus", "torus_knot", "gyroid"] = "smooth_noise"
    geom_smooth_iters: int = 4
    geom_freq_low: float = 0.02
    geom_freq_high: float = 0.15
    geom_freq_power: float = 1.0
    
    # Torus parameters (for geom_mode="torus" or "torus_knot")
    torus_R: float = 0.35  # major radius (fraction of grid)
    torus_r: float = 0.15  # minor radius (fraction of grid)
    torus_p: int = 2       # knot winding (longitudinal)
    torus_q: int = 3       # knot winding (meridional)
    
    # Noise overlay (adds noise on top of any geometry)
    noise_amplitude: float = 0.0  # 0 = no noise overlay
    noise_smoothing: int = 2      # smoothing iterations for noise
    
    # Phase space
    phase_dim: int = 1  # scalar phase for volume (keeps it simple)
    phase_space: Literal["unwrapped", "wrapped"] = "unwrapped"
    
    # Isosurface extraction
    iso_level: float = 0.0  # extract surface where φ = iso_level
    
    seed: int | None = 0


@dataclass
class PRVolumeState:
    """State for a 3D Phase-Resolved voxel field."""
    
    phi: np.ndarray  # (N, N, N) or (N, N, N, phase_dim)
    geom: np.ndarray  # (N, N, N) scalar geometry proxy
    falsifier_residual: np.ndarray | None = None  # (N, N, N) gradient magnitude


def _laplacian_3d(phi: np.ndarray) -> np.ndarray:
    """Compute 3D Laplacian with periodic boundaries."""
    lap = (
        np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) +
        np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) +
        np.roll(phi, 1, axis=2) + np.roll(phi, -1, axis=2) -
        6.0 * phi
    )
    return lap


def _gradient_magnitude_3d(phi: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude |∇φ| with periodic boundaries."""
    dx = np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)
    dy = np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)
    dz = np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)
    return np.sqrt(dx**2 + dy**2 + dz**2) * 0.5


def _make_geom_proxy_smooth_3d(N: int, rng: np.random.Generator, iters: int) -> np.ndarray:
    """Generate smooth 3D geometry proxy via repeated averaging."""
    geom = rng.uniform(-1, 1, (N, N, N)).astype(np.float32)
    for _ in range(iters):
        geom = (
            np.roll(geom, 1, axis=0) + np.roll(geom, -1, axis=0) +
            np.roll(geom, 1, axis=1) + np.roll(geom, -1, axis=1) +
            np.roll(geom, 1, axis=2) + np.roll(geom, -1, axis=2) +
            geom
        ) / 7.0
    geom = (geom - geom.min()) / (geom.max() - geom.min() + 1e-9)
    return geom.astype(np.float32)


def _make_geom_proxy_spectral_3d(
    N: int, rng: np.random.Generator,
    f_low: float, f_high: float, power: float
) -> np.ndarray:
    """Generate 3D geometry proxy via FFT band-pass filtering."""
    noise = rng.standard_normal((N, N, N)).astype(np.float32)
    F = np.fft.fftn(noise)
    
    # Build 3D frequency radius grid
    freq_x = np.fft.fftfreq(N)
    freq_y = np.fft.fftfreq(N)
    freq_z = np.fft.fftfreq(N)
    FX, FY, FZ = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
    R = np.sqrt(FX**2 + FY**2 + FZ**2)
    
    # Band-pass filter
    mask = ((R >= f_low) & (R <= f_high)).astype(np.float32)
    if power != 1.0:
        weight = np.where(R > 1e-9, R ** (power - 1), 0.0)
        mask = mask * weight
    
    F_filtered = F * mask
    geom = np.fft.ifftn(F_filtered).real.astype(np.float32)
    geom = (geom - geom.min()) / (geom.max() - geom.min() + 1e-9)
    return geom


def _make_geom_proxy_torus_3d(N: int, R: float, r: float) -> np.ndarray:
    """
    Generate torus-shaped SDF as geometry proxy.
    
    The torus is centered in the grid with:
    - R: major radius (distance from center to tube center)
    - r: minor radius (tube radius)
    
    Values are normalized [0,1] with 1 inside the torus shell.
    """
    # Create coordinate grid [-0.5, 0.5]^3
    x = np.linspace(-0.5, 0.5, N, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Distance from z-axis
    rho = np.sqrt(X**2 + Y**2)
    
    # Torus SDF: distance to tube surface
    # d = sqrt((rho - R)^2 + z^2) - r
    sdf = np.sqrt((rho - R)**2 + Z**2) - r
    
    # Convert SDF to smooth interior mask (1 inside, 0 outside)
    # Using sigmoid-like smooth transition
    thickness = 0.02
    geom = 0.5 - 0.5 * np.tanh(sdf / thickness)
    
    return geom.astype(np.float32)


def _make_geom_proxy_torus_knot_3d(N: int, R: float, r: float, p: int, q: int) -> np.ndarray:
    """
    Generate torus knot SDF as geometry proxy.
    
    A (p,q)-torus knot winds p times around the torus longitudinally
    and q times meridionally.
    """
    x = np.linspace(-0.5, 0.5, N, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Sample points along the knot curve
    n_samples = 512
    t = np.linspace(0, 2 * np.pi, n_samples, dtype=np.float32)
    
    # Torus knot parametric curve
    # x(t) = (R + r*cos(q*t)) * cos(p*t)
    # y(t) = (R + r*cos(q*t)) * sin(p*t)
    # z(t) = r * sin(q*t)
    curve_x = (R + r * 0.5 * np.cos(q * t)) * np.cos(p * t)
    curve_y = (R + r * 0.5 * np.cos(q * t)) * np.sin(p * t)
    curve_z = r * 0.5 * np.sin(q * t)
    
    # Compute minimum distance from each grid point to the curve
    # (vectorized over curve samples)
    sdf = np.full((N, N, N), np.inf, dtype=np.float32)
    tube_r = r * 0.4  # tube thickness
    
    for i in range(n_samples):
        dist = np.sqrt((X - curve_x[i])**2 + (Y - curve_y[i])**2 + (Z - curve_z[i])**2)
        sdf = np.minimum(sdf, dist)
    
    sdf = sdf - tube_r  # offset to tube surface
    
    # Convert to smooth mask
    thickness = 0.02
    geom = 0.5 - 0.5 * np.tanh(sdf / thickness)
    
    return geom.astype(np.float32)


def _make_geom_proxy_gyroid_3d(N: int, scale: float = 4.0) -> np.ndarray:
    """
    Generate gyroid minimal surface as geometry proxy.
    
    Gyroid: sin(x)cos(y) + sin(y)cos(z) + sin(z)cos(x) ≈ 0
    """
    t = np.linspace(0, scale * 2 * np.pi, N, dtype=np.float32)
    X, Y, Z = np.meshgrid(t, t, t, indexing='ij')
    
    # Gyroid implicit surface
    gyroid = np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)
    
    # Normalize to [0, 1]
    geom = (gyroid - gyroid.min()) / (gyroid.max() - gyroid.min() + 1e-9)
    
    return geom.astype(np.float32)


def _add_noise_overlay(geom: np.ndarray, rng: np.random.Generator, amplitude: float, smooth_iters: int) -> np.ndarray:
    """Add smoothed noise overlay to existing geometry."""
    if amplitude <= 0:
        return geom
    
    N = geom.shape[0]
    noise = rng.uniform(-1, 1, (N, N, N)).astype(np.float32)
    
    # Smooth the noise
    for _ in range(smooth_iters):
        noise = (
            np.roll(noise, 1, axis=0) + np.roll(noise, -1, axis=0) +
            np.roll(noise, 1, axis=1) + np.roll(noise, -1, axis=1) +
            np.roll(noise, 1, axis=2) + np.roll(noise, -1, axis=2) +
            noise
        ) / 7.0
    
    # Normalize and blend
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-9) - 0.5
    result = geom + amplitude * noise
    result = np.clip(result, 0, 1)
    
    return result.astype(np.float32)


def relax_phase_volume(cfg: PRVolumeConfig) -> tuple[dict, PRVolumeState]:
    """
    Run 3D phase field relaxation.
    
    Returns:
        metrics: dict with convergence info
        state: PRVolumeState with final phi, geom, residual
    """
    N = cfg.size
    rng = np.random.default_rng(cfg.seed)
    
    # Initialize phase field
    if cfg.phase_dim == 1:
        phi = rng.uniform(-1, 1, (N, N, N)).astype(np.float32)
    else:
        phi = rng.uniform(-1, 1, (N, N, N, cfg.phase_dim)).astype(np.float32)
    
    # Build geometry proxy
    if cfg.geom_mode == "spectral_band":
        geom = _make_geom_proxy_spectral_3d(N, rng, cfg.geom_freq_low, cfg.geom_freq_high, cfg.geom_freq_power)
    elif cfg.geom_mode == "torus":
        geom = _make_geom_proxy_torus_3d(N, cfg.torus_R, cfg.torus_r)
    elif cfg.geom_mode == "torus_knot":
        geom = _make_geom_proxy_torus_knot_3d(N, cfg.torus_R, cfg.torus_r, cfg.torus_p, cfg.torus_q)
    elif cfg.geom_mode == "gyroid":
        geom = _make_geom_proxy_gyroid_3d(N)
    else:
        geom = _make_geom_proxy_smooth_3d(N, rng, cfg.geom_smooth_iters)
    
    # Add noise overlay if requested
    if cfg.noise_amplitude > 0:
        geom = _add_noise_overlay(geom, rng, cfg.noise_amplitude, cfg.noise_smoothing)
    
    # Relaxation loop
    history_delta = []
    
    for step in range(cfg.steps):
        if cfg.phase_dim == 1:
            lap = _laplacian_3d(phi)
        else:
            lap = np.stack([_laplacian_3d(phi[..., d]) for d in range(cfg.phase_dim)], axis=-1)
        
        # Diffusion term
        dphi = cfg.dt * cfg.diffusion * lap
        
        # Coupling to geometry
        if cfg.coupling_mode == "geom_target" and cfg.coupling > 0:
            if cfg.phase_dim == 1:
                target = geom * 2.0 - 1.0  # map [0,1] → [-1,1]
                dphi += cfg.dt * cfg.coupling * (target - phi)
            else:
                target = (geom * 2.0 - 1.0)[..., None]
                dphi += cfg.dt * cfg.coupling * (target - phi)
        
        phi_new = phi + dphi
        
        # Wrap if needed
        if cfg.phase_space == "wrapped":
            phi_new = np.arctan2(np.sin(phi_new * np.pi), np.cos(phi_new * np.pi)) / np.pi
        
        delta = float(np.abs(phi_new - phi).mean())
        history_delta.append(delta)
        phi = phi_new
    
    # Compute residual (gradient magnitude as "branch tension" proxy)
    if cfg.phase_dim == 1:
        residual = _gradient_magnitude_3d(phi)
    else:
        residual = np.sqrt(sum(_gradient_magnitude_3d(phi[..., d])**2 for d in range(cfg.phase_dim)))
    
    metrics = {
        "steps": cfg.steps,
        "final_delta": history_delta[-1] if history_delta else 0.0,
        "phi_min": float(phi.min()),
        "phi_max": float(phi.max()),
        "phi_mean": float(phi.mean()),
        "residual_mean": float(residual.mean()),
        "residual_max": float(residual.max()),
        "geom_mode": cfg.geom_mode,
    }
    
    state = PRVolumeState(phi=phi, geom=geom, falsifier_residual=residual)
    return metrics, state


def extract_isosurface(
    phi: np.ndarray,
    level: float = 0.0,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract isosurface from 3D scalar field using marching cubes.
    
    Returns:
        vertices: (M, 3) array
        faces: (F, 3) array of triangle indices
    """
    try:
        from skimage import measure
    except ImportError:
        # Fallback: simple threshold-based point cloud
        mask = phi > level
        coords = np.argwhere(mask).astype(np.float32) * scale
        # Return as degenerate mesh (points only)
        return coords, np.zeros((0, 3), dtype=np.int32)
    
    try:
        verts, faces, normals, values = measure.marching_cubes(
            phi, level=level, spacing=(scale, scale, scale)
        )
        # Center the mesh around origin
        if len(verts) > 0:
            center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
            verts = verts - center
        return verts.astype(np.float32), faces.astype(np.int32)
    except Exception:
        # Field might be entirely above/below level
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)


def export_volume_as_ama(
    state: PRVolumeState,
    cfg: PRVolumeConfig,
    iso_level: float | None = None,
    scale: float = 1.0,
    units: str = "mm",
) -> bytes:
    """
    Export a 3D PR volume as an .ama archive.
    
    Contains:
      - mesh/isosurface.stl (extracted surface)
      - fields/phi.npy (3D phase field)
      - fields/geom.npy (geometry proxy)
      - fields/residual.npy (gradient magnitude)
      - analytic/scene.json, manifest.json, provenance
    """
    import io
    import json
    import zipfile
    import hashlib
    from datetime import datetime, timezone
    
    if iso_level is None:
        iso_level = cfg.iso_level
    
    phi = state.phi if state.phi.ndim == 3 else state.phi[..., 0]
    
    # Extract isosurface
    verts, faces = extract_isosurface(phi, level=iso_level, scale=scale)
    
    # Build STL
    if len(verts) > 0 and len(faces) > 0:
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            stl_io = io.BytesIO()
            mesh.export(stl_io, file_type="stl")
            stl_bytes = stl_io.getvalue()
        except ImportError:
            # ASCII STL fallback
            lines = ["solid isosurface"]
            for f in faces:
                v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
                e1, e2 = v1 - v0, v2 - v0
                n = np.cross(e1, e2)
                n_len = np.linalg.norm(n)
                n = n / n_len if n_len > 1e-9 else np.array([0, 0, 1])
                lines.append(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
                lines.append("    outer loop")
                lines.append(f"      vertex {v0[0]:.6f} {v0[1]:.6f} {v0[2]:.6f}")
                lines.append(f"      vertex {v1[0]:.6f} {v1[1]:.6f} {v1[2]:.6f}")
                lines.append(f"      vertex {v2[0]:.6f} {v2[1]:.6f} {v2[2]:.6f}")
                lines.append("    endloop")
                lines.append("  endfacet")
            lines.append("endsolid isosurface")
            stl_bytes = "\n".join(lines).encode("utf-8")
    else:
        stl_bytes = b"solid empty\nendsolid empty"
    
    # Scene JSON
    scene = {
        "layers": [
            {"name": "phi", "field": "fields/phi.npy", "colormap": "plasma", "enabled": True},
            {"name": "geom", "field": "fields/geom.npy", "colormap": "viridis", "enabled": False},
            {"name": "residual", "field": "fields/residual.npy", "colormap": "inferno", "enabled": False},
        ],
        "mesh": "mesh/isosurface.stl",
        "render": {"wireframe": False, "shading": "smooth", "iso_level": iso_level},
        "volume": {
            "shape": list(phi.shape),
            "iso_level": iso_level,
            "scale": scale,
        },
    }
    
    # Manifest
    manifest = {
        "type": "pr_volume",
        "fields": {
            "phi": {"shape": list(phi.shape), "dtype": "float32", "path": "fields/phi.npy"},
            "geom": {"shape": list(state.geom.shape), "dtype": "float32", "path": "fields/geom.npy"},
            "residual": {"shape": list(state.falsifier_residual.shape), "dtype": "float32", "path": "fields/residual.npy"},
        },
        "config": {
            "size": cfg.size,
            "steps": cfg.steps,
            "dt": cfg.dt,
            "diffusion": cfg.diffusion,
            "coupling": cfg.coupling,
            "coupling_mode": cfg.coupling_mode,
            "geom_mode": cfg.geom_mode,
            "phase_space": cfg.phase_space,
            "iso_level": iso_level,
        },
    }
    
    # Provenance
    provenance = {
        "generator": "adaptivecad.pr.volume",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "units": units,
        "scale": scale,
        "checksum_phi": hashlib.sha256(phi.tobytes()).hexdigest()[:16],
    }
    
    # Pack archive
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mesh/isosurface.stl", stl_bytes)
        
        for name, arr in [("phi", phi), ("geom", state.geom), ("residual", state.falsifier_residual)]:
            arr_io = io.BytesIO()
            np.save(arr_io, arr.astype(np.float32))
            zf.writestr(f"fields/{name}.npy", arr_io.getvalue())
        
        zf.writestr("analytic/scene.json", json.dumps(scene, indent=2))
        zf.writestr("analytic/manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("meta/provenance.json", json.dumps(provenance, indent=2))
    
    return buf.getvalue()


def save_volume_ama(
    state: PRVolumeState,
    cfg: PRVolumeConfig,
    path: str,
    iso_level: float | None = None,
    scale: float = 1.0,
    units: str = "mm",
) -> str:
    """Save volume .ama to disk."""
    data = export_volume_as_ama(state, cfg, iso_level=iso_level, scale=scale, units=units)
    with open(path, "wb") as f:
        f.write(data)
    return path


__all__ = [
    "PRVolumeConfig",
    "PRVolumeState",
    "relax_phase_volume",
    "extract_isosurface",
    "export_volume_as_ama",
    "save_volume_ama",
]
