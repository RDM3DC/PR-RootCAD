"""
- Builds a 4D tesseract (16 vertices, 32 edges)
- Projects to 3D (perspective), producing nested cubes
- Solves a TSP over the 16 vertices (NN + 2-opt)
- Exports: vertices.csv, edges.csv, tsp_order.csv, and two PNG plots
- If AdaptiveCAD's πₐ kernel is installed, uses it to modulate the projection.
"""
import os, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from adaptivecad.pi.kernel import pi_a, PiAParams
    HAVE_PIA = True
    PIA_PARAMS = PiAParams(beta=0.2, s0=1.0, clamp=0.3)
except Exception:
    HAVE_PIA = False
    def pi_a(kappa, scale, params=None):
        return math.pi
    PIA_PARAMS = None

def tesseract_vertices(scale=1.0):
    verts4 = []
    for x in (-1,1):
        for y in (-1,1):
            for z in (-1,1):
                for w in (-1,1):
                    verts4.append([scale*x, scale*y, scale*z, scale*w])
    return np.array(verts4, dtype=float)

def tesseract_edges(verts4):
    n = len(verts4); edges = []
    for i in range(n):
        for j in range(i+1, n):
            if np.sum(verts4[i]!=verts4[j]) == 1:
                edges.append((i,j))
    return edges

def project4_to_3(verts4, d=3.0, pia_params=PIA_PARAMS):
    out = np.zeros((len(verts4), 3), float)
    for idx,(x,y,z,w) in enumerate(verts4):
        kappa = 0.0
        pa = pi_a(kappa, scale=abs(w)+1e-9, params=pia_params) if HAVE_PIA else math.pi
        d_eff = d * (pa / math.pi)
        denom = max(d_eff - w, 1e-9)
        f = d_eff / denom
        out[idx] = [x*f, y*f, z*f]
    return out

def distance_matrix(P):
    n = len(P); D = np.zeros((n,n), float)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(P[i]-P[j])
            D[i,j]=D[j,i]=d
    return D

def nn_tour(D):
    n = D.shape[0]; unv = set(range(n)); tour = []; cur = 0
    tour.append(cur); unv.remove(cur)
    while unv:
        nxt = min(unv, key=lambda j: D[cur,j])
        tour.append(nxt); unv.remove(nxt); cur = nxt
    return tour

def two_opt(tour, D, max_iter=10000):
    n = len(tour); best = tour[:]; it=0
    while it<max_iter:
        improved=False
        for i in range(1, n-2):
            for k in range(i+1, n-1):
                a,b = best[i-1], best[i]
                c,d = best[k], best[(k+1)%n]
                gain = (D[a,b]+D[c,d]) - (D[a,c]+D[b,d])
                if gain > 1e-12:
                    best[i:k+1] = reversed(best[i:k+1])
                    improved=True
        if not improved: break
        it+=1
    return best

def plot_edges(P, edges, path_png):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:,0], P[:,1], P[:,2], s=40, depthshade=True)
    for i,j in edges:
        xs = [P[i,0], P[j,0]]; ys=[P[i,1], P[j,1]]; zs=[P[i,2], P[j,2]]
        ax.plot(xs, ys, zs, linewidth=1.0, alpha=0.8)
    ax.set_title("4D Tesseract → 3D projection (nested cubes)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout(); plt.savefig(path_png, dpi=180); plt.close(fig)

def plot_tsp(P, order, path_png):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:,0], P[:,1], P[:,2], s=40, depthshade=True, c="k")
    for idx, pidx in enumerate(order):
        ax.text(P[pidx,0], P[pidx,1], P[pidx,2], str(idx), fontsize=8)
    for i in range(len(order)-1):
        a,b = order[i], order[i+1]
        ax.plot([P[a,0],P[b,0]],[P[a,1],P[b,1]],[P[a,2],P[b,2]], linewidth=1.5)
    a,b = order[-1], order[0]
    ax.plot([P[a,0],P[b,0]],[P[a,1],P[b,1]],[P[a,2],P[b,2]], linewidth=1.0, linestyle=":")
    ax.set_title("TSP path over tesseract vertices (NN+2opt)")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    plt.tight_layout(); plt.savefig(path_png, dpi=200); plt.close(fig)

def main(out_dir="out"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    V4 = tesseract_vertices(scale=1.0)
    E = tesseract_edges(V4)
    P3 = project4_to_3(V4, d=3.0)
    pd.DataFrame(P3, columns=["x","y","z"]).to_csv(Path(out_dir)/"vertices.csv", index=False)
    pd.DataFrame(E, columns=["i","j"]).to_csv(Path(out_dir)/"edges.csv", index=False)
    plot_edges(P3, E, Path(out_dir)/"tesseract_projection.png")
    D = distance_matrix(P3)
    tour = two_opt(nn_tour(D), D, max_iter=10000)
    pd.DataFrame({"order": list(range(len(tour))), "vertex": tour}).to_csv(Path(out_dir)/"tsp_order.csv", index=False)
    plot_tsp(P3, tour, Path(out_dir)/"tsp_path.png")
    print("[ok] wrote:", Path(out_dir)/"vertices.csv")

if __name__=="__main__":
    main()
