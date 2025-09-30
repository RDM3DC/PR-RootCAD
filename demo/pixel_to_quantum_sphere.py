#!/usr/bin/env python3
"""
Pixel-to-Quantum Sphere Zoom (procedural)
Generates a GIF that starts as a ~1-pixel sphere and reveals structure as it grows.

Usage:
  python demo/pixel_to_quantum_sphere.py --out AdaptiveCAD_pixel_to_quantum_sphere.gif \
      --size 512 --frames 96 --duration 80
"""
import argparse
import math

import numpy as np
from PIL import Image


def smoothstep(a: float, b: float, x: float) -> float:
    if b == a:
        return 0.0
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return float(t * t * (3 - 2 * t))


def make_frame(R: float, size: int = 384) -> np.ndarray:
    s = size
    lin = np.linspace(-1.0, 1.0, s, dtype=np.float32)
    xv, yv = np.meshgrid(lin, lin)
    r_norm = R / (s / 2.0)
    r2 = xv ** 2 + yv ** 2
    mask = r2 <= r_norm ** 2 + 1e-9

    zv = np.zeros_like(xv)
    zv[mask] = np.sqrt(np.maximum(r_norm ** 2 - r2[mask], 0.0))

    nx, ny, nz = xv.copy(), yv.copy(), zv.copy()
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-9
    nx /= norm
    ny /= norm
    nz /= norm

    L = np.array([0.4, -0.2, 0.9], dtype=np.float32)
    L = L / np.linalg.norm(L)
    intensity = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0.0, 1.0)
    base = 0.12 + 0.88 * intensity
    img = np.zeros((s, s, 3), dtype=np.float32)
    img[...] = base[..., None]

    theta = np.arccos(np.clip(nz, -1.0, 1.0))
    phi = np.arctan2(ny, nx)
    u = (phi + math.pi) / (2 * math.pi)
    v = theta / math.pi

    w_micro = smoothstep(10, 40, R)
    w_meso = smoothstep(50, 100, R)
    w_molecular = smoothstep(110, 150, R)
    w_atomic = smoothstep(160, 190, R)
    w_nuclear = smoothstep(200, 215, R)
    w_sub = smoothstep(215, 230, R)

    if w_micro > 1e-6:
        noise = (np.sin(30 * u + 18 * v) + np.sin(25 * u - 22 * v) + np.sin(40 * u + 12 * v)) / 3.0
        img += (w_micro * 0.05 * noise)[..., None]

    if w_meso > 1e-6:
        f = 6.0
        g1 = 0.5 * (np.cos(2 * np.pi * f * u) + np.cos(2 * np.pi * f * v))
        g2 = 0.5 * (
            np.cos(2 * np.pi * f * (0.5 * u + (np.sqrt(3) / 2) * v))
            + np.cos(2 * np.pi * f * (u - 0.5 * v))
        )
        grains = np.maximum(g1, g2)
        boundaries = np.clip(1.0 - (grains + 1.0) / 2.0, 0.0, 1.0)
        img[..., 0] += w_meso * 0.08 * boundaries
        img[..., 1] += w_meso * 0.04 * boundaries
        img[..., 2] += w_meso * 0.02 * boundaries

    if w_molecular > 1e-6:
        f = 18.0
        a1 = np.cos(2 * np.pi * f * u)
        a2 = np.cos(2 * np.pi * f * (0.5 * u + (np.sqrt(3) / 2) * v))
        a3 = np.cos(2 * np.pi * f * ((-0.5) * u + (np.sqrt(3) / 2) * v))
        hex_lines = np.abs(a1 * a2 * a3)
        lines = hex_lines ** 8
        img[..., 1] += w_molecular * 0.15 * lines
        img[..., 0] += w_molecular * 0.05 * lines

    if w_atomic > 1e-6:
        Y10 = np.cos(theta)
        Y11c = np.sin(theta) * np.cos(phi)
        Y11s = np.sin(theta) * np.sin(phi)
        density = Y10 ** 2 + 0.5 * (Y11c ** 2 + Y11s ** 2)
        lobes = density / (density.max() + 1e-9)
        img[..., 2] += w_atomic * 0.25 * lobes
        img[..., 0] += w_atomic * 0.05 * lobes

    if w_nuclear > 1e-6:
        rr = np.sqrt(xv ** 2 + yv ** 2)
        core = np.exp(- (rr / (r_norm * 0.2 + 1e-6)) ** 2)
        img[..., 0] += w_nuclear * 0.25 * core
        img[..., 1] += w_nuclear * 0.05 * core
        img[..., 2] += w_nuclear * 0.05 * core

    if w_sub > 1e-6:
        sub = (np.sin(200 * u) + np.sin(220 * v) + np.sin(140 * (u + v))) / 3.0
        img += (w_sub * 0.05 * sub)[..., None]

    img = np.clip(img, 0.0, 1.0)
    out = np.zeros_like(img)
    out[mask] = img[mask]
    return (out * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=384, help="Frame width/height in pixels")
    ap.add_argument("--frames", type=int, default=72, help="Number of frames")
    ap.add_argument("--duration", type=int, default=80, help="ms per frame")
    ap.add_argument("--out", type=str, default="AdaptiveCAD_pixel_to_quantum_sphere.gif")
    args = ap.parse_args()

    radii = np.linspace(0.8, args.size * 0.46, args.frames)
    frames = [Image.fromarray(make_frame(R, size=args.size)) for R in radii]
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=args.duration,
        optimize=False,
    )
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
