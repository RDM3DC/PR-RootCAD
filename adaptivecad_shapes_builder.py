#!/usr/bin/env python3
"""
AdaptiveCAD Shape Builder — offline (copy‑paste friendly)

Generates STL files locally so you don’t need to download from chat.

Shapes:
  • rt_seam_hex        — single hex seam frame (ring + 6 ribs)
  • rt_seam_patch      — 7‑tile patch (center + 6 neighbors)
  • hopf_weave         — multi‑strand Hopf/torus‑knot braid
  • wire_weave         — orthogonal sinusoid lattice (wire gyroid weave)
  • ablative_skin      — redshift hull: graded skin over hex face (thickness map)

Usage examples:
  python adaptivecad_shapes_builder.py --shape rt_seam_hex --out seam_hex.stl
  python adaptivecad_shapes_builder.py --shape rt_seam_patch --R 25 --out patch.stl
  python adaptivecad_shapes_builder.py --shape hopf_weave --n_strands 6 --out hopf.stl
  python adaptivecad_shapes_builder.py --shape wire_weave --wires 3 --r_wire 0.6 --out wire.stl
  python adaptivecad_shapes_builder.py --shape ablative_skin --t0 2.0 --alpha 0.6 --out skin.stl

Requires: Python 3.9+ and numpy (pip install numpy)
No other dependencies.
"""
import argparse
import math
from pathlib import Path

import numpy as np


# ---------------- STL writer (binary) ----------------
def write_stl_binary(path, verts, faces):
    path = Path(path)
    with open(path, "wb") as f:
        header = b"AdaptiveCAD offline" + b" " * (80 - len("AdaptiveCAD offline"))
        f.write(header)
        f.write(np.array([faces.shape[0]], dtype="<u4").tobytes())
        V = verts.astype(np.float32)
        for tri in faces:
            a, b, c = V[tri[0]], V[tri[1]], V[tri[2]]
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn > 0:
                n /= nn
            else:
                n[:] = 0.0
            f.write(np.asarray(n, dtype="<f4").tobytes())
            f.write(np.asarray(a, dtype="<f4").tobytes())
            f.write(np.asarray(b, dtype="<f4").tobytes())
            f.write(np.asarray(c, dtype="<f4").tobytes())
            f.write(np.array([0], dtype="<u2").tobytes())
    return str(path)


# ---------------- Primitive: cylinder between points ----------------
def cylinder_between(a, b, r=0.8, segments=14):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ab = b - a
    L = np.linalg.norm(ab)
    if L < 1e-9:
        t = np.linspace(0, 2 * math.pi, segments, endpoint=False)
        circle = np.stack([r * np.cos(t), r * np.sin(t), np.zeros_like(t)], axis=1)
        V = np.vstack([a + circle, a + 0 * circle])
        F = []
        m = segments
        for i in range(m):
            F.append([i, (i + 1) % m, m + i])
            F.append([m + i, (i + 1) % m, m + ((i + 1) % m)])
        return V, np.array(F, int)
    z = ab / L
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, z)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    circle = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
    ring0 = a + circle[:, 0, None] * x[None, :] + circle[:, 1, None] * y[None, :]
    ring1 = b + circle[:, 0, None] * x[None, :] + circle[:, 1, None] * y[None, :]
    V = np.vstack([ring0, ring1])
    F = []
    m = segments
    for i in range(m):
        a0 = i
        a1 = (i + 1) % m
        b0 = i + m
        b1 = ((i + 1) % m) + m
        F.append([a0, a1, b1])
        F.append([a0, b1, b0])
    # caps
    center0 = ring0.mean(axis=0)
    center1 = ring1.mean(axis=0)
    V = np.vstack([V, center0, center1])
    c0 = V.shape[0] - 2
    c1 = V.shape[0] - 1
    for i in range(m):
        F.append([c0, (i + 1) % m, i])
        F.append([c1, i + m, (i + 1) % m + m])
    return V, np.array(F, int)


# ---------------- Shape A: RT seam frames ----------------
def hex_vertices(R=25.0, z=6.0):
    return np.array(
        [[R * math.cos(k * math.pi / 3), R * math.sin(k * math.pi / 3), z] for k in range(6)], float
    )


def build_rt_seam_frame(R=25.0, z=6.0, r_edge=0.8, r_rib=0.65):
    V_parts = []
    F_parts = []
    base = 0
    Vhex = hex_vertices(R, z)
    # ring
    for k in range(6):
        a = Vhex[k]
        b = Vhex[(k + 1) % 6]
        Vc, Fc = cylinder_between(a, b, r=r_edge, segments=14)
        F_parts.append(Fc + base)
        V_parts.append(Vc)
        base += Vc.shape[0]
    # ribs to center
    center = np.array([0.0, 0.0, z])
    for k in range(6):
        Vc, Fc = cylinder_between(Vhex[k], center, r=r_rib, segments=12)
        F_parts.append(Fc + base)
        V_parts.append(Vc)
        base += Vc.shape[0]
    V = np.vstack(V_parts)
    F = np.vstack(F_parts)
    return V, F


def hex_centers(R):
    a = math.sqrt(3) * R
    b = 1.5 * R
    return [
        (0.0, 0.0),
        (a, 0.0),
        (0.5 * a, b),
        (-0.5 * a, b),
        (-a, 0.0),
        (-0.5 * a, -b),
        (0.5 * a, -b),
    ]


def build_rt_seam_patch(R=25.0, z=6.0, r_edge=0.75, r_rib=0.6):
    V1, F1 = build_rt_seam_frame(R, z, r_edge, r_rib)
    centers = hex_centers(R)
    Vall = []
    Fall = []
    base = 0
    for cx, cy in centers:
        Vsh = V1.copy()
        Vsh[:, 0] += cx
        Vsh[:, 1] += cy
        Vall.append(Vsh)
        Fall.append(F1 + base)
        base += Vsh.shape[0]
    return np.vstack(Vall), np.vstack(Fall)


# ---------------- Shape B: Hopf braid weave ----------------
def torus_knot_points(p=3, q=2, R=20.0, r=6.0, N=600):
    t = np.linspace(0, 2 * math.pi, N, endpoint=False)
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)
    return np.stack([x, y, z], axis=1)


def tube_polyline(points, r=0.6, segments=10):
    V_all = []
    F_all = []
    base = 0
    for i in range(len(points) - 1):
        Va, Fa = cylinder_between(points[i], points[i + 1], r=r, segments=segments)
        F_all.append(Fa + base)
        V_all.append(Va)
        base += Va.shape[0]
    if not V_all:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.vstack(V_all), np.vstack(F_all)


def hopf_braid_weave(n_strands=6, p=3, q=2, R=20.0, r=6.0, r_strand=0.7, phase_jitter=0.2, seed=2):
    rng = np.random.default_rng(seed)
    V_parts = []
    F_parts = []
    base = 0
    N = 600
    for s in range(n_strands):
        phase = (2 * math.pi * s / n_strands) + rng.uniform(-phase_jitter, phase_jitter)
        P = torus_knot_points(p, q, R, r, N=N)
        shift = int((phase / (2 * math.pi)) * P.shape[0]) % P.shape[0]
        P = np.roll(P, shift, axis=0)
        Vt, Ft = tube_polyline(P, r=r_strand, segments=9)
        F_parts.append(Ft + base)
        V_parts.append(Vt)
        base += Vt.shape[0]
    return np.vstack(V_parts), np.vstack(F_parts)


# ---------------- Shape C: Wire gyroid weave ----------------
def wire_gyroid_paths(L=40.0, n_cells=3, wires_per_axis=3, amp=4.0, N=120):
    xmin = ymin = zmin = -L / 2
    xmax = ymax = zmax = L / 2
    xs = np.linspace(xmin, xmax, N)
    ys = np.linspace(ymin, ymax, N)
    zs = np.linspace(zmin, zmax, N)
    k = 2 * math.pi * n_cells / L
    paths = []
    # X-wires
    for i in range(wires_per_axis):
        for j in range(wires_per_axis):
            y0 = ymin + (i + 0.5) * (L / wires_per_axis)
            z0 = zmin + (j + 0.5) * (L / wires_per_axis)
            P = np.stack(
                [xs, y0 + amp * np.sin(k * xs + 0.6 * j), z0 + amp * np.cos(k * xs + 0.4 * i)],
                axis=1,
            )
            paths.append(P)
    # Y-wires
    for i in range(wires_per_axis):
        for j in range(wires_per_axis):
            x0 = xmin + (i + 0.5) * (L / wires_per_axis)
            z0 = zmin + (j + 0.5) * (L / wires_per_axis)
            P = np.stack(
                [x0 + amp * np.sin(k * ys + 0.5 * i), ys, z0 + amp * np.cos(k * ys + 0.7 * j)],
                axis=1,
            )
            paths.append(P)
    # Z-wires
    for i in range(wires_per_axis):
        for j in range(wires_per_axis):
            x0 = xmin + (i + 0.5) * (L / wires_per_axis)
            y0 = ymin + (j + 0.5) * (L / wires_per_axis)
            P = np.stack(
                [x0 + amp * np.sin(k * zs + 0.4 * j), y0 + amp * np.cos(k * zs + 0.6 * i), zs],
                axis=1,
            )
            paths.append(P)
    return paths


def wire_gyroid_weave(L=40.0, n_cells=3, wires_per_axis=3, amp=4.0, r_wire=0.6, N=120):
    paths = wire_gyroid_paths(L, n_cells, wires_per_axis, amp, N)
    V_all = []
    F_all = []
    base = 0
    for P in paths:
        Vt, Ft = tube_polyline(P, r=r_wire, segments=8)
        F_all.append(Ft + base)
        V_all.append(Vt)
        base += Vt.shape[0]
    return np.vstack(V_all), np.vstack(F_all)


# ---------------- Shape D: Redshift hull (ablative skin) ----------------
def inside_hex_xy(x, y, R):
    return (np.abs(x) <= R) & (np.abs(x) + np.sqrt(3.0) * np.abs(y) <= 2 * R)


def synth_flux(X, Y, R):
    q_hot = np.exp(-(((X - 0.25 * R) ** 2 + (Y + 0.1 * R) ** 2) / (2 * (0.22 * R) ** 2)))
    q_str = 0.6 * np.exp(-((X + 0.05 * R) ** 2) / (2 * (0.35 * R) ** 2))
    q_rim = 0.25 * (1.0 - ((X**2 + Y**2) / (R**2)))
    q = q_hot + q_str + q_rim
    m = inside_hex_xy(X, Y, R)
    q *= m
    if np.any(m):
        q = (q - q[m].min()) / (q[m].ptp() + 1e-12)
    return q


def triangulate_skin(xs, ys, mask, tmap, base_z):
    nx, ny = len(xs), len(ys)

    def vid(i, j, top):
        return (i * ny + j) * 2 + (1 if top else 0)

    verts = np.zeros((nx * ny * 2, 3), float)
    for i in range(nx):
        for j in range(ny):
            z_top = base_z + (tmap[i, j] if mask[i, j] else 0.0)
            verts[vid(i, j, False)] = [xs[i], ys[j], base_z]
            verts[vid(i, j, True)] = [xs[i], ys[j], z_top]
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            cell_inside = mask[i, j] & mask[i + 1, j] & mask[i, j + 1] & mask[i + 1, j + 1]
            if cell_inside:
                a = vid(i, j, True)
                b = vid(i + 1, j, True)
                c = vid(i + 1, j + 1, True)
                d = vid(i, j + 1, True)
                faces += [[a, b, c], [a, c, d]]
                a = vid(i, j, False)
                b = vid(i + 1, j, False)
                c = vid(i + 1, j + 1, False)
                d = vid(i, j + 1, False)
                faces += [[c, b, a], [d, c, a]]
    for i in range(nx - 1):
        for j in range(ny):
            a_inside = mask[i, j]
            b_inside = mask[i + 1, j]
            if a_inside != b_inside:
                if a_inside:
                    left = (i, j)
                    right = (i + 1, j)
                else:
                    left = (i + 1, j)
                    right = (i, j)
                a0 = vid(left[0], left[1], False)
                a1 = vid(left[0], left[1], True)
                b0 = vid(right[0], right[1], False)
                b1 = vid(right[0], right[1], True)
                faces += [[a0, b0, b1], [a0, b1, a1]]
    for i in range(nx):
        for j in range(ny - 1):
            a_inside = mask[i, j]
            b_inside = mask[i, j + 1]
            if a_inside != b_inside:
                if a_inside:
                    low = (i, j)
                    high = (i, j + 1)
                else:
                    low = (i, j + 1)
                    high = (i, j)
                a0 = vid(low[0], low[1], False)
                a1 = vid(low[0], low[1], True)
                b0 = vid(high[0], high[1], False)
                b1 = vid(high[0], high[1], True)
                faces += [[a0, b0, b1], [a0, b1, a1]]
    return np.array(verts, float), np.array(faces, int)


# ---------------- Main CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="AdaptiveCAD Shape Builder (offline)")
    ap.add_argument(
        "--shape",
        required=True,
        choices=["rt_seam_hex", "rt_seam_patch", "hopf_weave", "wire_weave", "ablative_skin"],
    )
    ap.add_argument("--out", required=True, help="Output STL path")
    # common params
    ap.add_argument("--R", type=float, default=25.0, help="hex radius / scale (mm)")
    ap.add_argument("--z", type=float, default=6.0, help="z height for seam frames (mm)")
    # hopf
    ap.add_argument("--n_strands", type=int, default=6)
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--r_strand", type=float, default=0.7)
    # wire weave
    ap.add_argument("--wires", type=int, default=3, help="wires per axis")
    ap.add_argument("--r_wire", type=float, default=0.6)
    ap.add_argument("--L", type=float, default=40.0, help="lattice cube size (mm)")
    # ablative skin
    ap.add_argument("--t0", type=float, default=2.0, help="base thickness (mm)")
    ap.add_argument("--alpha", type=float, default=0.6, help="thickness gain vs flux")
    ap.add_argument("--tile_thick", type=float, default=12.0, help="tile base z (mm)")
    ap.add_argument("--grid", type=int, default=180, help="grid for skin triangulation")

    args = ap.parse_args()
    shape = args.shape

    if shape == "rt_seam_hex":
        V, F = build_rt_seam_frame(R=args.R, z=args.z)
    elif shape == "rt_seam_patch":
        V, F = build_rt_seam_patch(R=args.R, z=args.z)
    elif shape == "hopf_weave":
        V, F = hopf_braid_weave(
            n_strands=args.n_strands, p=args.p, q=args.q, r_strand=args.r_strand
        )
    elif shape == "wire_weave":
        V, F = wire_gyroid_weave(L=args.L, wires_per_axis=args.wires, r_wire=args.r_wire)
    elif shape == "ablative_skin":
        nx = ny = int(args.grid)
        xs = np.linspace(-args.R, args.R, nx)
        ys = np.linspace(-args.R, args.R, ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        mask = inside_hex_xy(X, Y, args.R)
        qhat = synth_flux(X, Y, args.R)
        tmap = args.t0 * (1.0 + args.alpha * qhat) * mask
        V, F = triangulate_skin(xs, ys, mask, tmap, base_z=args.tile_thick)
    else:
        raise SystemExit("Unknown shape")

    out = write_stl_binary(args.out, V, F)
    print(f"Wrote: {out}  (triangles={len(F)})")


if __name__ == "__main__":
    main()
