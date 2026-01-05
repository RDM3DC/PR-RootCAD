# Minimal mesh helpers: cotangent Laplacian, Taubin smoothing, and a few meshes
from __future__ import annotations

import math

import numpy as np


# ---------- Laplacian (cotangent, symmetric weights) ----------
def _cotangent(a, b, c):
    u = b - a
    v = c - a
    dot = float(u @ v)
    cross = np.linalg.norm(np.cross(u, v))
    return dot / (cross + 1e-18)


def build_cotan_weights(verts: np.ndarray, faces: np.ndarray, sample_stride: int = 1):
    """Return neighbor list and weights using cotangent formula.
    sample_stride>1 subsamples faces for speed."""
    N = len(verts)
    W = {}
    F = faces[:: max(1, sample_stride)]
    for i, j, k in F:
        a, b, c = verts[i], verts[j], verts[k]
        cotA = _cotangent(a, b, c)
        cotB = _cotangent(b, c, a)
        cotC = _cotangent(c, a, b)
        for p, q, w in ((i, j, cotC), (j, k, cotA), (k, i, cotB)):
            if p > q:
                p, q = q, p
            W[(p, q)] = W.get((p, q), 0.0) + max(w, 0.0) / 2.0  # clamp small negatives
    nbrs = [[] for _ in range(N)]
    weights = [[] for _ in range(N)]
    for (p, q), w in W.items():
        nbrs[p].append(q)
        weights[p].append(w)
        nbrs[q].append(p)
        weights[q].append(w)
    return nbrs, weights


def laplacian_apply(V: np.ndarray, nbrs, weights):
    out = np.zeros_like(V)
    for i in range(len(V)):
        Vi = V[i]
        s = np.zeros(3)
        for j, w in zip(nbrs[i], weights[i]):
            s += w * (Vi - V[j])
        out[i] = s
    return out


def taubin_step(V, nbrs, weights, lam=0.02, mu=-0.01):
    V1 = V - lam * laplacian_apply(V, nbrs, weights)
    V2 = V1 - mu * laplacian_apply(V1, nbrs, weights)
    return V2


def curvature_proxy(V, nbrs, weights):
    LV = laplacian_apply(V, nbrs, weights)
    k = np.linalg.norm(LV, axis=1)
    k = (k - k.min()) / (k.max() - k.min() + 1e-12)
    return k


# ---------- Sphere (icosphere) ----------
def _make_icosahedron(R=1.0):
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
        float,
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
        int,
    )
    return verts, faces


def _subdivide(verts, faces, R):
    verts = verts.tolist()
    faces = faces.tolist()
    edge = {}

    def mid(a, b):
        key = tuple(sorted((a, b)))
        if key in edge:
            return edge[key]
        p = (np.array(verts[a]) + np.array(verts[b])) / 2.0
        p = p / np.linalg.norm(p) * R
        idx = len(verts)
        verts.append(p.tolist())
        edge[key] = idx
        return idx

    newF = []
    for i, j, k in faces:
        a = mid(i, j)
        b = mid(j, k)
        c = mid(k, i)
        newF += [(i, a, c), (a, j, b), (c, b, k), (a, b, c)]
    return np.array(verts, float), np.array(newF, int)


def make_icosphere(R=1.0, depth=2):
    v, f = _make_icosahedron(R)
    for _ in range(depth):
        v, f = _subdivide(v, f, R)
    return v, f


# ---------- Saddle z=(x^2 - y^2)/(2R) on [-a,a]^2 ----------
def make_saddle(R=1.0, a=1.2, n=80):
    xs = np.linspace(-a, a, n)
    ys = np.linspace(-a, a, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = (X * X - Y * Y) / (2.0 * R)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    faces = []

    def idx(i, j):
        return (i % n) * n + (j % n)

    for i in range(n - 1):
        for j in range(n - 1):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    return verts, np.array(faces, int)


# ---------- Klein bottle (figure-8 immersion) ----------
def klein_param(U, V, a=2.0):
    x = (a + np.cos(U / 2) * np.sin(V) - np.sin(U / 2) * np.sin(2 * V)) * np.cos(U)
    y = (a + np.cos(U / 2) * np.sin(V) - np.sin(U / 2) * np.sin(2 * V)) * np.sin(U)
    z = np.sin(U / 2) * np.sin(V) + np.cos(U / 2) * np.sin(2 * V)
    return x, y, z


def make_klein(nu=50, nv=50, a=2.0):
    us = np.linspace(0, 2 * math.pi, nu, endpoint=False)
    vs = np.linspace(0, 2 * math.pi, nv, endpoint=False)
    U, V = np.meshgrid(us, vs, indexing="ij")
    X, Y, Z = klein_param(U, V, a)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    faces = []

    def idx(i, j):
        return (i % nu) * nv + (j % nv)

    for i in range(nu):
        for j in range(nv):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            faces.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    return verts, np.array(faces, int)
