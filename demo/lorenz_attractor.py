#!/usr/bin/env python3
"""
Lorenz Attractor Demo
- Renders a rotating GIF of the Lorenz trajectory.
- Exports a 3D-printable tube mesh (ASCII STL) of the trajectory.

Usage:
  python demo/lorenz_attractor.py \
      --out-gif lorenz_attractor.gif \
      --out-stl lorenz_tube.stl \
      --frames 180 --size 640 --duration 60

Args:
  --sigma, --rho, --beta : Lorenz parameters (default 10, 28, 8/3)
  --dt                   : time step for RK4 (default 0.005)
  --steps                : integration steps (default 25000)
  --burn-in              : discard initial steps before recording (default 1000)
  --frames               : GIF frames (default 180)
  --size                 : GIF width/height in px (default 640)
  --duration             : ms per frame (default 60)
  --tube-radius          : tube radius as fraction of bbox (default 0.02)
  --ring-segments        : radial segments of tube (default 16)
"""
import argparse
import io
import math
from typing import Tuple

import numpy as np
from PIL import Image


# ---------- Lorenz system and integrator ----------
def lorenz_deriv(
    p: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0
) -> np.ndarray:
    x, y, z = p
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dtype=np.float64)


def rk4_step(p: np.ndarray, dt: float, fn) -> np.ndarray:
    k1 = fn(p)
    k2 = fn(p + 0.5 * dt * k1)
    k3 = fn(p + 0.5 * dt * k2)
    k4 = fn(p + dt * k3)
    return p + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_lorenz(
    p0: Tuple[float, float, float],
    dt: float = 0.005,
    steps: int = 25000,
    burn_in: int = 1000,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
) -> np.ndarray:
    p = np.array(p0, dtype=np.float64)
    out = []

    def f(q: np.ndarray) -> np.ndarray:
        return lorenz_deriv(q, sigma=sigma, rho=rho, beta=beta)

    for i in range(steps + burn_in):
        p = rk4_step(p, dt, f)
        if i >= burn_in:
            out.append(p.copy())
    return np.asarray(out, dtype=np.float64)


# ---------- Tube builder and STL writer ----------
def build_tube(traj_pts: np.ndarray, radius: float = 0.25, ring_segments: int = 16) -> np.ndarray:
    """Return triangles (N, 3, 3) for a tube following traj_pts."""
    P = traj_pts[::10]
    N = len(P)

    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12

    up = np.array([0.0, 0.0, 1.0])
    rings = []
    for i in range(N):
        t = T[i]
        ref = up if abs(np.dot(t, up)) < 0.95 else np.array([0.0, 1.0, 0.0])
        n = np.cross(t, ref)
        n /= np.linalg.norm(n) + 1e-12
        b = np.cross(t, n)
        b /= np.linalg.norm(b) + 1e-12
        angles = np.linspace(0, 2 * np.pi, ring_segments, endpoint=False)
        ring = np.array([P[i] + radius * (np.cos(a) * n + np.sin(a) * b) for a in angles])
        rings.append(ring)
    rings = np.asarray(rings)

    tris = []
    M = ring_segments
    for i in range(N - 1):
        r0 = rings[i]
        r1 = rings[i + 1]
        for j in range(M):
            a0 = r0[j]
            a1 = r0[(j + 1) % M]
            b0 = r1[j]
            b1 = r1[(j + 1) % M]
            tris.append((a0, b0, a1))
            tris.append((a1, b0, b1))
    return np.asarray(tris, dtype=np.float64)


def write_ascii_stl(path: str, triangles: np.ndarray, name: str = "lorenz_tube") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"solid {name}\n")
        for tri in triangles:
            v0, v1, v2 = tri
            n = np.cross(v1 - v0, v2 - v0)
            n /= np.linalg.norm(n) + 1e-18
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {name}\n")


# ---------- GIF rendering via matplotlib ----------
def render_gif(
    traj: np.ndarray, out_path: str, frames: int = 180, size: int = 640, duration_ms: int = 60
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    sub = traj[::5]
    mins = sub.min(axis=0)
    maxs = sub.max(axis=0)
    center = (mins + maxs) / 2.0
    rng = (maxs - mins).max() * 0.55
    lims = np.vstack([center - rng, center + rng])

    dpi = 128
    fig_inch = (size / dpi, size / dpi)
    images = []

    for i, n in enumerate(np.linspace(50, len(sub), frames, dtype=int)):
        fig = plt.figure(figsize=fig_inch, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(lims[0, 0], lims[1, 0])
        ax.set_ylim(lims[0, 1], lims[1, 1])
        ax.set_zlim(lims[0, 2], lims[1, 2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)

        az = 30 + 200.0 * (i / max(1, frames - 1))
        el = 25 + 5.0 * math.sin(2 * math.pi * i / max(1, frames - 1))
        ax.view_init(elev=el, azim=az)

        seg = sub[:n]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=1.0)

        fig.subplots_adjust(0, 0, 1, 1)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("P"))

    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )


# ---------- CLI ----------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=10.0)
    ap.add_argument("--rho", type=float, default=28.0)
    ap.add_argument("--beta", type=float, default=8.0 / 3.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--burn-in", type=int, default=1000)
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--out-gif", type=str, default="lorenz_attractor.gif")
    ap.add_argument("--out-stl", type=str, default="lorenz_tube.stl")
    ap.add_argument(
        "--tube-radius", type=float, default=0.02, help="Tube radius as fraction of bbox"
    )
    ap.add_argument("--ring-segments", type=int, default=16)
    args = ap.parse_args()

    traj = integrate_lorenz(
        [0.01, 0.0, 0.0],
        dt=args.dt,
        steps=args.steps,
        burn_in=args.burn_in,
        sigma=args.sigma,
        rho=args.rho,
        beta=args.beta,
    )

    render_gif(
        traj,
        out_path=args.out_gif,
        frames=args.frames,
        size=args.size,
        duration_ms=args.duration,
    )

    mins = traj.min(axis=0)
    maxs = traj.max(axis=0)
    bbox_scale = (maxs - mins).max()
    tube_r = args.tube_radius * bbox_scale
    tris = build_tube(traj, radius=tube_r, ring_segments=args.ring_segments)
    write_ascii_stl(args.out_stl, tris)

    print(f"Wrote: {args.out_gif}")
    print(f"Wrote: {args.out_stl}")


if __name__ == "__main__":
    main()
