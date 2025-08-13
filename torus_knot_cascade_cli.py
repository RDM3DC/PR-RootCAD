#!/usr/bin/env python3
"""
Torus-knot π_a cascade CLI
- Builds a Möbius ribbon around a (p,q) torus knot
- π_a = π * (1 + λ * trC(t)) with multi-hotspot constraint field trC
- Exports: render PNG, turntable GIF, and a metrics CSV (throat count, min width)
- Optional λ-sweep to visualize throat "persistence" vs λ
"""
import argparse, math, numpy as np, csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

def centerline(p, q, R, r, N):
    t = np.linspace(0, 2*math.pi, N, endpoint=False)
    x = (R + r*np.cos(q*t)) * np.cos(p*t)
    y = (R + r*np.cos(q*t)) * np.sin(p*t)
    z = r*np.sin(q*t)
    C = np.stack([x,y,z], axis=1)
    return t, C

def tangent(C):
    Cp = np.roll(C,-1,axis=0) - np.roll(C,1,axis=0)
    T = Cp / (np.linalg.norm(Cp,axis=1,keepdims=True)+1e-12)
    return T

def parallel_transport_frame(C):
    T = tangent(C)
    z = np.array([0.0,0.0,1.0])
    n0 = z - (z@T[0])*T[0]
    if np.linalg.norm(n0) < 1e-6:
        z = np.array([0.0,1.0,0.0]); n0 = z - (z@T[0])*T[0]
    n0 = n0/np.linalg.norm(n0)
    Nvec = np.zeros_like(C); Bvec = np.zeros_like(C)
    Nvec[0] = n0; Bvec[0] = np.cross(T[0], Nvec[0])
    for i in range(1, len(C)):
        v = Nvec[i-1] - (Nvec[i-1] @ T[i]) * T[i]
        if np.linalg.norm(v) < 1e-9:
            v = Bvec[i-1] - (Bvec[i-1] @ T[i]) * T[i]
        v = v/(np.linalg.norm(v)+1e-12)
        Nvec[i] = v; Bvec[i] = np.cross(T[i], v)
    return T, Nvec, Bvec

def make_hotspots(t, k, neg_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.choice(t, size=k, replace=False)
    widths  = rng.uniform(0.15, 0.35, size=k)
    signs   = np.ones(k)
    if neg_frac>0:
        neg_idx = rng.choice(np.arange(k), size=max(1, int(neg_frac*k)), replace=False)
        signs[neg_idx] = -1.0
    trC = np.zeros_like(t)
    for c, w, s in zip(centers, widths, signs):
        # periodic distance on a circle
        dt = np.minimum(np.abs(t-c), 2*math.pi - np.abs(t-c))
        trC += s * np.exp(-(dt**2)/(2*w**2))
    trC += 0.25*np.cos(3*t)
    # normalize to [-1,1]
    trC = (trC - trC.mean())
    trC = trC / (np.max(np.abs(trC))+1e-12)
    return trC

def build_ribbon(C, Nvec, Bvec, lam, trC, w=0.35, M=64, twist=0.5):
    pi = math.pi
    N = len(C)
    f = pi / (pi*(1 + lam*trC))  # π/π_a
    s = np.linspace(-w, w, M)
    theta = twist * np.linspace(0, 2*math.pi, N, endpoint=False)  # Möbius: 0.5
    ct, st = np.cos(theta), np.sin(theta)
    Uhat = (ct[:,None]*Nvec + st[:,None]*Bvec)
    V = np.zeros((N,M,3), float)
    for i in range(N):
        V[i,:,:] = C[i][None,:] + (s[:,None]*f[i]) * Uhat[i][None,:]
    # faces
    faces = []
    def vid(i,j): return (i%N)*M + (j%M)
    for i in range(N):
        for j in range(M):
            a=vid(i,j); b=vid(i+1,j); c=vid(i+1,j+1); d=vid(i,j+1)
            faces.append([a,b,c]); faces.append([a,c,d])
    return V.reshape(-1,3), np.array(faces,int), f

def throat_metrics(f, w):
    widths = 2*w*f
    # local minima detection (1D periodic)
    N = len(widths)
    mins = []
    for i in range(N):
        im1 = (i-1)%N; ip1=(i+1)%N
        if widths[i] < widths[im1] and widths[i] < widths[ip1]:
            mins.append((i, widths[i]))
    if not mins:
        return 0, float(widths.min()), float(widths.mean())
    idx, vals = zip(*mins)
    # de-duplicate minima too close (within 2 steps)
    sel = []
    for i,v in mins:
        if not sel or (i - sel[-1][0])%N > 2:
            sel.append((i,v))
    count = len(sel)
    minw  = float(np.min([v for _,v in sel]))
    meanw = float(np.mean([v for _,v in sel]))
    return count, minw, meanw

def render_png(verts, faces, color_vals, out_png, elev=25, az=35):
    tris = faces[::1]
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=tris, linewidth=0.05, antialiased=False, alpha=1.0)
    surf.set_array(color_vals[tris].mean(axis=1))
    ax.set_axis_off(); ax.view_init(elev=elev, azim=az)
    plt.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)

def gif_turntable(verts, faces, color_vals, out_gif, frames=32):
    tris = faces[::1]
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=tris, linewidth=0.05, antialiased=False, alpha=1.0)
    surf.set_array(color_vals[tris].mean(axis=1))
    ax.set_axis_off(); ax.view_init(elev=25, azim=0)
    plt.tight_layout()
    def animate(k):
        ax.view_init(elev=25, azim= (360.0/frames)*k )
        return (surf,)
    anim = FuncAnimation(fig, animate, frames=frames, interval=80, blit=False)
    anim.save(out_gif, writer=PillowWriter(fps=12))
    plt.close(fig)

def run_once(args):
    t, C = centerline(args.p, args.q, args.R, args.r, args.N)
    T, Nvec, Bvec = parallel_transport_frame(C)
    trC = make_hotspots(t, args.hotspots, neg_frac=(args.neg_frac if args.neg_energy else 0.0), seed=args.seed)
    verts, faces, f = build_ribbon(C, Nvec, Bvec, args.lam, trC, w=args.width, M=args.M, twist=args.twist)
    # color by π_a
    pi = math.pi; pi_a = pi*(1 + args.lam*trC); col = np.repeat(pi_a, args.M); col=(col-col.min())/(col.max()-col.min()+1e-12)
    # metrics
    n_throat, minw, meanw = throat_metrics(f, args.width)
    # outputs
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    render_png(verts, faces, col, out/"cascade_render.png", elev=25, az=35)
    gif_turntable(verts, faces, col, out/"cascade_turntable.gif", frames=args.frames)
    # CSV
    with open(out/"metrics.csv","w",newline="") as fcsv:
        w=csv.writer(fcsv); w.writerow(["n_throat","min_width","mean_width","lambda","hotspots","neg_energy"]); w.writerow([n_throat,minw,meanw,args.lam,args.hotspots,int(args.neg_energy)])
    return n_throat, minw, meanw

def sweep_lambda(args):
    lams = np.linspace(args.lmin, args.lmax, args.lsteps)
    counts=[]; mins=[]
    for lam in lams:
        a = argparse.Namespace(**vars(args)); a.lam=float(lam)
        n,minw,_ = run_once(a)
        counts.append(n); mins.append(minw)
    # plot
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    ax.plot(lams, counts, label="throat count")
    ax.set_xlabel("λ"); ax.set_ylabel("count")
    ax2 = ax.twinx()
    ax2.plot(lams, mins, label="min width")
    ax2.set_ylabel("min width")
    fig.tight_layout(); fig.savefig(out/"lambda_persistence.png", dpi=170); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pq", type=str, default="3,2")
    ap.add_argument("--R", type=float, default=2.4)
    ap.add_argument("--r", type=float, default=0.9)
    ap.add_argument("--N", type=int, default=800)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--width", type=float, default=0.35)
    ap.add_argument("--twist", type=float, default=0.5, help="Möbius twist factor (0.5=half twist)")
    ap.add_argument("--lam", type=float, default=0.12)
    ap.add_argument("--hotspots", type=int, default=4)
    ap.add_argument("--neg_energy", action="store_true", help="enable negative trC lobes")
    ap.add_argument("--neg_frac", type=float, default=0.25, help="fraction of hotspots that are negative when --neg_energy on")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--out", type=str, default=str(Path(__file__).resolve().parent / "out"))
    ap.add_argument("--sweep", action="store_true", help="sweep λ and plot persistence")
    ap.add_argument("--lmin", type=float, default=0.02)
    ap.add_argument("--lmax", type=float, default=0.20)
    ap.add_argument("--lsteps", type=int, default=12)
    args = ap.parse_args()
    p,q = map(int, args.pq.split(",")); args.p, args.q = p, q
    run_once(args)
    if args.sweep: sweep_lambda(args)

if __name__ == "__main__":
    main()
