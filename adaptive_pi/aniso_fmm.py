"""
adaptive_pi.aniso_fmm
---------------------
A compact anisotropic geodesic distance solver ("FMM-lite") and backtracer.

Solves: sqrt(∇Tᵀ G⁻¹(x,y) ∇T) = 1 with T(source)=0 over a 2D grid.

Exports
-------
- anisotropic_fmm(Gfield, source, hx=1.0, hy=1.0, use_diagonals=True) -> T
- trace_geodesic(T, Gfield, start, src, hx=1.0, hy=1.0, step=0.8, ... ) -> [(x,y)]
- metric_const_aniso(nx, ny, a=1.3, b=1.0) -> (nx,ny,2,2) SPD field
- metric_conformal(nx, ny, s) -> SPD field for G = s(x,y) I, where s is scalar or callable
- metric_from_unit_vector(nx, ny, ufield, s_para=1.3, s_perp=1.0) -> SPD field aligned to u(x,y)

Notes
-----
- This is a practical, stable baseline (ordered method with a two-neighbor PDE
  update + one-sided fallback). It soft-lands to isotropic FMM when G = s I.
- Keep anisotropy ratio moderate (κ ≲ 10–20) for best behavior.
"""

from __future__ import annotations

import heapq
import math
from typing import List, Tuple

import numpy as np

# --------------------------- Metric helpers -----------------------------


def ensure_spd(G: np.ndarray) -> np.ndarray:
    S = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-12)
    return (V * w) @ V.T


def inv2(G: np.ndarray) -> np.ndarray:
    a, b, c, d = G[0, 0], G[0, 1], G[1, 0], G[1, 1]
    det = a * d - b * c
    if det <= 0:
        raise ValueError("Metric not SPD (det<=0).")
    return (1.0 / det) * np.array([[d, -b], [-c, a]], dtype=float)


def riemann_step_cost(G: np.ndarray, dx: np.ndarray) -> float:
    return float(np.sqrt(dx @ G @ dx))


# --------------------------- Neighborhoods ------------------------------


def neighbors8(i, j, nx, ny):
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        ii, jj = i + di, j + dj
        if 0 <= ii < nx and 0 <= jj < ny:
            yield ii, jj


def axial_neighbors(i, j, nx, ny):
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ii, jj = i + di, j + dj
        if 0 <= ii < nx and 0 <= jj < ny:
            yield ii, jj


# --------------------------- Core solver --------------------------------


def anisotropic_fmm(
    Gfield: np.ndarray,
    source: Tuple[int, int],
    hx: float = 1.0,
    hy: float = 1.0,
    use_diagonals: bool = True,
) -> np.ndarray:
    nx, ny = Gfield.shape[:2]
    T = np.full((nx, ny), np.inf, dtype=float)
    state = np.zeros((nx, ny), dtype=np.uint8)  # 0=FAR,1=TRIAL,2=ACCEPTED
    heap: List[Tuple[float, int, int]] = []

    i0, j0 = source
    T[i0, j0] = 0.0
    state[i0, j0] = 2  # ACCEPTED

    # Initialize neighbors
    for ii, jj in (neighbors8 if use_diagonals else axial_neighbors)(i0, j0, nx, ny):
        cand = one_sided_candidates((ii, jj), T, Gfield, hx, hy)
        if cand < T[ii, jj]:
            T[ii, jj] = cand
            state[ii, jj] = 1
            heapq.heappush(heap, (T[ii, jj], ii, jj))

    while heap:
        val, i, j = heapq.heappop(heap)
        if val != T[i, j]:
            continue  # stale
        state[i, j] = 2  # ACCEPTED
        for ii, jj in (neighbors8 if use_diagonals else axial_neighbors)(i, j, nx, ny):
            if state[ii, jj] == 2:  # already ACCEPTED
                continue
            cand_pde = two_neighbor_update((ii, jj), T, Gfield, hx, hy)
            cand_edge = one_sided_candidates((ii, jj), T, Gfield, hx, hy)
            cand = min(c for c in (cand_pde, cand_edge) if math.isfinite(c))
            if cand < T[ii, jj]:
                T[ii, jj] = cand
                state[ii, jj] = 1
                heapq.heappush(heap, (T[ii, jj], ii, jj))
    return T


def one_sided_candidates(node, T, Gfield, hx, hy):
    i, j = node
    nx, ny = T.shape
    best = np.inf
    G = ensure_spd(Gfield[i, j])
    for ii, jj in neighbors8(i, j, nx, ny):
        if not np.isfinite(T[ii, jj]):
            continue
        dx = np.array([(i - ii) * hx, (j - jj) * hy], dtype=float)
        w = riemann_step_cost(G, dx)
        best = min(best, T[ii, jj] + w)
    return best


def two_neighbor_update(node, T, Gfield, hx, hy):
    i, j = node
    nx, ny = T.shape
    Tx_candidates = []
    if i - 1 >= 0 and np.isfinite(T[i - 1, j]):
        Tx_candidates.append(T[i - 1, j])
    if i + 1 < nx and np.isfinite(T[i + 1, j]):
        Tx_candidates.append(T[i + 1, j])
    Ty_candidates = []
    if j - 1 >= 0 and np.isfinite(T[i, j - 1]):
        Ty_candidates.append(T[i, j - 1])
    if j + 1 < ny and np.isfinite(T[i, j + 1]):
        Ty_candidates.append(T[i, j + 1])
    if len(Tx_candidates) == 0 or len(Ty_candidates) == 0:
        return np.inf

    Tp = min(Tx_candidates)  # upwind x
    Tq = min(Ty_candidates)  # upwind y

    G = ensure_spd(Gfield[i, j])
    A = inv2(G)
    a = A[0, 0] / (hx * hx)
    b = A[0, 1] / (hx * hy)
    c = A[1, 1] / (hy * hy)

    S = a + 2.0 * b + c
    P = a * Tp + b * (Tp + Tq) + c * Tq
    Q = a * Tp * Tp + 2.0 * b * Tp * Tq + c * Tq * Tq - 1.0

    disc = P * P - S * Q
    if disc < 0:
        return np.inf
    X = (P + math.sqrt(max(0.0, disc))) / S  # larger root (upwind)
    if X < max(Tp, Tq) - 1e-12:
        return np.inf
    return X


# --------------------------- Backtracer ---------------------------------


def trace_geodesic(
    T: np.ndarray,
    Gfield: np.ndarray,
    start: Tuple[int, int],
    src: Tuple[int, int],
    hx: float = 1.0,
    hy: float = 1.0,
    step: float = 0.8,
    max_steps: int = 10000,
    tol: float = 1e-3,
):
    nx, ny = T.shape
    x, y = float(start[0]), float(start[1])
    xs, ys = float(src[0]), float(src[1])

    def sample_T(xf, yf):
        i0 = int(np.clip(np.floor(xf), 0, nx - 2))
        j0 = int(np.clip(np.floor(yf), 0, ny - 2))
        tx = xf - i0
        ty = yf - j0
        T00 = T[i0, j0]
        T10 = T[i0 + 1, j0]
        T01 = T[i0, j0 + 1]
        T11 = T[i0 + 1, j0 + 1]
        return (1 - tx) * (1 - ty) * T00 + tx * (1 - ty) * T10 + (1 - tx) * ty * T01 + tx * ty * T11

    def sample_gradT(xf, yf):
        eps = 0.5
        Tx1 = sample_T(np.clip(xf + eps, 0, nx - 1), yf)
        Tx0 = sample_T(np.clip(xf - eps, 0, nx - 1), yf)
        Ty1 = sample_T(xf, np.clip(yf + eps, 0, ny - 1))
        Ty0 = sample_T(xf, np.clip(yf - eps, 0, ny - 1))
        return np.array([(Tx1 - Tx0) / (2 * eps * hx), (Ty1 - Ty0) / (2 * eps * hy)], dtype=float)

    def sample_A(xf, yf):
        i = int(np.clip(round(xf), 0, nx - 1))
        j = int(np.clip(round(yf), 0, ny - 1))
        G = ensure_spd(Gfield[i, j])
        return inv2(G)

    path = [(x, y)]
    for _ in range(max_steps):
        if (x - xs) ** 2 + (y - ys) ** 2 <= tol * tol:
            break
        g = sample_gradT(x, y)
        A = sample_A(x, y)
        v = -A @ g
        nv = float(np.linalg.norm(v)) + 1e-12
        x += (step * v[0]) / nv
        y += (step * v[1]) / nv
        x = float(np.clip(x, 0, nx - 1))
        y = float(np.clip(y, 0, ny - 1))
        path.append((x, y))
    return path


# --------------------------- Metric builders ----------------------------


def metric_const_aniso(nx: int, ny: int, a: float = 1.3, b: float = 1.0) -> np.ndarray:
    G = np.zeros((nx, ny, 2, 2), dtype=float)
    G[..., 0, 0] = a
    G[..., 1, 1] = b
    return G


def metric_conformal(nx: int, ny: int, s) -> np.ndarray:
    """
    Build G = s(x,y) I, where s is:
      - float: constant scale
      - np.ndarray (nx,ny): scalar field
      - callable(x_idx, y_idx) -> float
    """
    G = np.zeros((nx, ny, 2, 2), dtype=float)
    if isinstance(s, (int, float)):
        G[..., 0, 0] = s
        G[..., 1, 1] = s
        return G
    if callable(s):
        for i in range(nx):
            for j in range(ny):
                val = float(s(i, j))
                G[i, j, 0, 0] = val
                G[i, j, 1, 1] = val
        return G
    # assume array-like
    s_arr = np.asarray(s, dtype=float)
    assert s_arr.shape == (nx, ny)
    G[..., 0, 0] = s_arr
    G[..., 1, 1] = s_arr
    return G


def metric_from_unit_vector(
    nx: int, ny: int, ufield, s_para: float = 1.3, s_perp: float = 1.0
) -> np.ndarray:
    """
    Build G aligned to unit vector field u(x,y) ∈ R²:
      G = s_perp I + (s_para - s_perp) u u^T.
    ufield can be:
      - callable(i,j) -> (ux, uy) assumed unit
      - ndarray (nx,ny,2) already unit
    """
    G = np.zeros((nx, ny, 2, 2), dtype=float)
    if callable(ufield):
        for i in range(nx):
            for j in range(ny):
                ux, uy = ufield(i, j)
                u = np.array([ux, uy], dtype=float)
                u /= max(1e-12, np.linalg.norm(u))
                M = s_perp * np.eye(2) + (s_para - s_perp) * np.outer(u, u)
                G[i, j] = M
    else:
        U = np.asarray(ufield, dtype=float)
        assert U.shape == (nx, ny, 2)
        for i in range(nx):
            for j in range(ny):
                u = U[i, j]
                n = max(1e-12, np.linalg.norm(u))
                u = u / n
                M = s_perp * np.eye(2) + (s_para - s_perp) * np.outer(u, u)
                G[i, j] = M
    return G
