import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def tri_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def face_angles(a, b, c):
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ca = np.linalg.norm(a - c)
    cosA = (ab**2 + ca**2 - bc**2) / (2 * ab * ca + 1e-18)
    cosB = (ab**2 + bc**2 - ca**2) / (2 * ab * bc + 1e-18)
    cosC = (bc**2 + ca**2 - ab**2) / (2 * bc * ca + 1e-18)
    cosA = np.clip(cosA, -1.0, 1.0)
    cosB = np.clip(cosB, -1.0, 1.0)
    cosC = np.clip(cosC, -1.0, 1.0)
    return np.arccos(cosA), np.arccos(cosB), np.arccos(cosC)


def vertex_curvature_angle_deficit(verts, faces):
    n = len(verts)
    angle_sum = np.zeros(n)
    area_sum = np.zeros(n)
    for f in faces:
        i, j, k = f
        a, b, c = verts[i], verts[j], verts[k]
        A, B, C = face_angles(a, b, c)
        area = tri_area(a, b, c)
        angle_sum[i] += A
        angle_sum[j] += B
        angle_sum[k] += C
        area_sum[i] += area / 3.0
        area_sum[j] += area / 3.0
        area_sum[k] += area / 3.0
    K = (2 * np.pi - angle_sum) / (area_sum + 1e-18)
    return K, area_sum


def write_ply_with_colors(path, verts, faces, colors):
    cols = np.clip(colors, 0.0, 1.0)
    cols = (cols * 255).astype(np.uint8)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(verts, cols):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def make_icosahedron(R=1.0):
    t = (1.0 + 5.0**0.5) / 2.0
    verts = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=float,
    )
    verts /= np.linalg.norm(verts, axis=1)[:, None]
    verts *= R
    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=int,
    )
    return verts, faces


def subdivide(verts, faces, R):
    verts = verts.tolist()
    faces = faces.tolist()
    edge_mid_cache = {}

    def mid(a, b):
        key = tuple(sorted((a, b)))
        if key in edge_mid_cache:
            return edge_mid_cache[key]
        p = (np.array(verts[a]) + np.array(verts[b])) / 2.0
        p = p / np.linalg.norm(p) * R
        idx = len(verts)
        verts.append(p.tolist())
        edge_mid_cache[key] = idx
        return idx

    new_faces = []
    for tri in faces:
        i, j, k = tri
        a = mid(i, j)
        b = mid(j, k)
        c = mid(k, i)
        new_faces += [[i, a, c], [a, j, b], [c, b, k], [a, b, c]]
    return np.array(verts, float), np.array(new_faces, int)


def make_icosphere(R=1.0, depth=2):
    v, f = make_icosahedron(R)
    for _ in range(depth):
        v, f = subdivide(v, f, R)
    return v, f


def sphere_example(outdir, R=1.0, depth=3, r_frac=0.1):
    os.makedirs(outdir, exist_ok=True)
    verts, faces = make_icosphere(R, depth)
    K, areas = vertex_curvature_angle_deficit(verts, faces)
    r = r_frac * R
    pi_eff = math.pi * (1.0 - (r * r / 6.0) * K)
    rel = (pi_eff - math.pi) / math.pi
    rel_clip = np.clip((rel + 0.02) / 0.04, 0.0, 1.0)
    colors = np.stack([rel_clip, np.zeros_like(rel_clip), 1.0 - rel_clip], axis=1)
    write_ply_with_colors(f"{outdir}/icosphere_pia_heatmap.ply", verts, faces, colors)
    pd.DataFrame(
        {
            "x": verts[:, 0],
            "y": verts[:, 1],
            "z": verts[:, 2],
            "K": K,
            "area": areas,
            "pi_eff": pi_eff,
            "rel_pi": rel,
        }
    ).to_csv(f"{outdir}/vertices_curvature.csv", index=False)
    plt.figure(figsize=(5, 3))
    plt.hist(K, bins=50)
    plt.title("Sphere: Gaussian curvature")
    plt.xlabel("K")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f"{outdir}/sphere_K_hist.png")
    plt.close()
    c = np.array([0, 0, R], float)
    u = np.array([1, 0, 0], float)
    v = np.array([0, 1, 0], float)
    theta = r / R
    N = 360
    circle_pts = []
    for t in np.linspace(0, 2 * np.pi, N, endpoint=False):
        w = math.cos(t) * u + math.sin(t) * v
        p = math.cos(theta) * c + math.sin(theta) * R * w
        p = p / np.linalg.norm(p) * R
        circle_pts.append(p)
    circle_pts = np.array(circle_pts)
    diffs = np.diff(np.vstack([circle_pts, circle_pts[0]]), axis=0)
    C_meas = np.sum(np.linalg.norm(diffs, axis=1))
    C_theory = 2 * math.pi * R * math.sin(theta)
    C_asym = 2 * math.pi * r * (1 - (r * r) / (6 * R * R))
    with open(f"{outdir}/geodesic_circle_report.txt", "w") as f:
        f.write(f"Sphere R={R}, r={r} (θ={theta} rad)\n")
        f.write(f"Measured circumference (polyline N={N}): {C_meas:.6f}\n")
        f.write(f"Theory 2π R sinθ:                         {C_theory:.6f}\n")
        f.write(f"Asymptotic 2π r (1 - r^2/(6R^2)):          {C_asym:.6f}\n")
        f.write(f"π_eff_meas = C_meas/(2r) = {C_meas/(2*r):.9f}\n")
        f.write(f"π_eff_asym = π*(1 - K r^2/6) with K=1/R^2 = {math.pi*(1-(r*r/(6*R*R))):.9f}\n")
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c=rel_clip, cmap="coolwarm", s=5)
    ax.set_title("Sphere: π_eff deviation (blue<0, red>0) for r/R=0.1")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"{outdir}/sphere_pi_eff_heatmap.png")
    plt.close()


def make_saddle(R=1.0, a=1.0, n=120):
    xs = np.linspace(-a, a, n)
    ys = np.linspace(-a, a, n)
    X, Y = np.meshgrid(xs, ys)
    Z = (X * X - Y * Y) / (2.0 * R)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            idx = i * n + j
            faces.append([idx, idx + 1, idx + n])
            faces.append([idx + 1, idx + n + 1, idx + n])
    return verts, np.array(faces, int)


def saddle_example(outdir, R=1.0, a=1.2, n=120, r_scale=0.2):
    os.makedirs(outdir, exist_ok=True)
    verts, faces = make_saddle(R, a, n)
    K, areas = vertex_curvature_angle_deficit(verts, faces)
    r0 = r_scale * R
    pi_eff = math.pi * (1.0 - (r0 * r0 / 6.0) * K)
    rel = (pi_eff - math.pi) / math.pi
    rel_clip = np.clip((rel + 0.05) / 0.1, 0.0, 1.0)
    colors = np.stack([rel_clip, np.zeros_like(rel_clip), 1.0 - rel_clip], axis=1)
    write_ply_with_colors(f"{outdir}/saddle_pia_heatmap.ply", verts, faces, colors)
    pd.DataFrame(
        {
            "x": verts[:, 0],
            "y": verts[:, 1],
            "z": verts[:, 2],
            "K": K,
            "pi_eff": pi_eff,
            "rel_pi": rel,
        }
    ).to_csv(f"{outdir}/vertices_curvature.csv", index=False)
    plt.figure(figsize=(6, 4))
    plt.hist(K, bins=80)
    plt.title("Saddle: Gaussian curvature")
    plt.xlabel("K")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f"{outdir}/saddle_K_hist.png")
    plt.close()
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c=rel_clip, cmap="coolwarm", s=2)
    ax.set_title("Saddle: π_eff deviation (blue<0, red>0) for r=0.2R")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"{outdir}/saddle_pi_eff_heatmap.png")
    plt.close()


if __name__ == "__main__":
    sphere_example("examples/gu_manifolds/sphere", R=1.0, depth=3, r_frac=0.1)
    saddle_example("examples/gu_manifolds/saddle", R=1.0, a=1.2, n=120, r_scale=0.2)
