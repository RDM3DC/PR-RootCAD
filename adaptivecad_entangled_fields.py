#!/usr/bin/env python3
"""
AdaptiveCAD Entangled Fields — λ–α (copy‑paste, numpy‑only)

Goal
----
Model "quantum‑logic infusions" as *entangled parameter fields*: two correlated random
fields over a hex tile — λ(x,y) (weave density scaler) and α(x,y) (hull thickness scaler).
Compute practical proxies (mass/density, A_eff, tortuosity, ΔT proxy), sweep correlation ρ,
and export a CSV + PNGs. No external deps beyond numpy/matplotlib.

Why this framing
---------------
We keep it physically grounded: instead of qubits, we treat λ and α as *correlated controls*.
ρ≈0 → independent tuning; ρ→1 → tightly coupled. This lets us answer “does entangling
parameters help?” with concrete metrics you can plot/post.

Usage
-----
python adaptivecad_entangled_fields.py --rho 0.0 0.25 0.5 0.75 0.9 \
    --seeds 0 1 2 --R 25 --t0 2.0 --alpha0 0.6 --r_wire0 0.6 --gam_w 0.3 --gam_a 0.6 \
    --out metrics.csv --figdir figs_entangled

Outputs
-------
• metrics.csv — rows per (ρ, seed) with proxies
• figs_entangled/lam_field.png, alpha_field.png, lam_vs_alpha_scatter.png
  and lam_alpha_panels.png (side‑by‑side fields within hex mask)

Notes
-----
• We map ⟨λ⟩ to wire radius: r_wire = r_wire0 * (1 + γ_w * ⟨λ⟩)
• We map ⟨α⟩ to skin thickness: t = t0 * (1 + γ_a * ⟨α⟩)
• Proxies:
  - wire volume ≈ π r_wire^2 L_tot  (L_tot from wire gyroid path family)
  - density ρ ≈ vol / bbox
  - A_eff ≈ vol / L
  - tortuosity τ ≈ mean(path length / straight span) across groups
  - ΔT_proxy ≈ 1 / (A_eff + κ * t)  (lower is better); you can tune κ

"""
import argparse, math
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# ------------- Hex mask -------------
def inside_hex_xy(X, Y, R):
    return (np.abs(X) <= R) & (np.abs(X) + np.sqrt(3.0)*np.abs(Y) <= 2*R)

# ------------- Correlated random fields (Gaussian blur via FFT) -------------
def gaussian_kernel_fft(nx, ny, sigma):
    # construct frequency response of Gaussian for separable blur
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    # Gaussian in freq: exp(-2π^2 σ^2 (fx^2 + fy^2))
    H = np.exp(-2*(np.pi**2)*(sigma**2)*(KX**2 + KY**2))
    return H

def correlated_fields(nx, ny, corr=0.5, sigma=6.0, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((nx,ny))
    w = rng.standard_normal((nx,ny))
    v = corr*u + np.sqrt(max(0.0, 1.0 - corr**2))*w
    H = gaussian_kernel_fft(nx, ny, sigma)
    def blur(a):
        Af = np.fft.fftn(a)
        return np.fft.ifftn(Af*H).real
    u_b = blur(u); v_b = blur(v)
    # normalize to [0,1]
    def norm01(a):
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-12)
    return norm01(u_b), norm01(v_b)

# ------------- Wire gyroid paths + proxies (polyline; numpy‑only) -------------
def wire_gyroid_paths(L=40.0, n_cells=3, wires_per_axis=3, amp=4.0, N=120):
    xmin=ymin=zmin = -L/2; xmax=ymax=zmax = L/2
    xs = np.linspace(xmin, xmax, N)
    ys = np.linspace(ymin, ymax, N)
    zs = np.linspace(zmin, zmax, N)
    k = 2*math.pi*n_cells/L
    paths = []
    for i in range(wires_per_axis):
        for j in range(wires_per_axis):
            y0 = ymin + (i+0.5)*(L/wires_per_axis)
            z0 = zmin + (j+0.5)*(L/wires_per_axis)
            P = np.stack([xs,
                          y0 + amp*np.sin(k*xs + 0.6*j),
                          z0 + amp*np.cos(k*xs + 0.4*i)], axis=1)
            paths.append(P)
    for i in range(wires_per_axis):
        for j in range(wires_per_axis):
            x0 = xmin + (i+0.5)*(L/wires_per_axis)
            z0 = zmin + (j+0.5)*(L/wires_per_axis)
            P = np.stack([x0 + amp*np.sin(k*ys + 0.5*i), ys, z0 + amp*np.cos(k*ys + 0.7*j)], axis=1)
            paths.append(P)
    for i in range(wires_per_axis):
        for j in range(wires_per_axis):
            x0 = xmin + (i+0.5)*(L/wires_per_axis)
            y0 = ymin + (j+0.5)*(L/wires_per_axis)
            P = np.stack([x0 + amp*np.sin(k*zs + 0.4*j), y0 + amp*np.cos(k*zs + 0.6*i), zs], axis=1)
            paths.append(P)
    return paths

def wire_metrics(L=40.0, wires_per_axis=3, r_wire=0.6, n_cells=3, amp=4.0, N=120):
    paths = wire_gyroid_paths(L,n_cells,wires_per_axis,amp,N)
    total_len = 0.0
    per_axis = wires_per_axis*wires_per_axis
    taus = []
    for idx,P in enumerate(paths):
        seg = np.linalg.norm(np.roll(P,-1,0) - P, axis=1).sum()
        total_len += seg
        span = L  # dominant axis span
        taus.append(seg/span)
    tau_mean = float(np.mean(taus))
    vol = math.pi * (r_wire**2) * total_len
    bbox_vol = L**3
    Aeff = vol / L
    density = vol / bbox_vol
    return {
        "total_len": total_len,
        "tau": tau_mean,
        "A_eff": Aeff,
        "density": density,
        "volume": vol,
        "bbox": bbox_vol,
    }

# ------------- Main sweep -------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rho', type=float, nargs='+', default=[0.0,0.25,0.5,0.75,0.9])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--R', type=float, default=25.0, help='Hex radius')
    ap.add_argument('--t0', type=float, default=2.0)
    ap.add_argument('--alpha0', type=float, default=0.6)
    ap.add_argument('--r_wire0', type=float, default=0.6)
    ap.add_argument('--gam_w', type=float, default=0.3)
    ap.add_argument('--gam_a', type=float, default=0.6)
    ap.add_argument('--kappa', type=float, default=0.1, help='ΔT proxy coefficient')
    ap.add_argument('--nx', type=int, default=128)
    ap.add_argument('--ny', type=int, default=128)
    ap.add_argument('--sigma', type=float, default=6.0)
    ap.add_argument('--L', type=float, default=40.0)
    ap.add_argument('--n_cells', type=int, default=3)
    ap.add_argument('--wires_per_axis', type=int, default=3)
    ap.add_argument('--amp', type=float, default=4.0)
    ap.add_argument('--N', type=int, default=120)
    ap.add_argument('--out', type=Path, default=Path('metrics.csv'))
    ap.add_argument('--figdir', type=Path, default=Path('figs_entangled'))
    args = ap.parse_args()

    args.figdir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []

    # coordinate grid for mask
    x = np.linspace(-args.R, args.R, args.nx)
    y = np.linspace(-args.R, args.R, args.ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    mask = inside_hex_xy(X, Y, args.R)

    for rho in args.rho:
        for seed in args.seeds:
            lam_field, alpha_field = correlated_fields(
                args.nx, args.ny, corr=rho, sigma=args.sigma, seed=seed
            )
            lam_vals = lam_field[mask]
            alpha_vals = alpha_field[mask]
            lam_mean = float(lam_vals.mean())
            alpha_mean = float(alpha_vals.mean())
            r_wire = args.r_wire0 * (1.0 + args.gam_w * lam_mean)
            t = args.t0 * (1.0 + args.gam_a * alpha_mean)
            wm = wire_metrics(
                L=args.L,
                wires_per_axis=args.wires_per_axis,
                r_wire=r_wire,
                n_cells=args.n_cells,
                amp=args.amp,
                N=args.N,
            )
            deltaT = 1.0 / (wm['A_eff'] + args.kappa * t)
            row = {
                'rho': rho,
                'seed': seed,
                'lam_mean': lam_mean,
                'alpha_mean': alpha_mean,
                'r_wire': r_wire,
                't': t,
                'density': wm['density'],
                'A_eff': wm['A_eff'],
                'tau': wm['tau'],
                'deltaT_proxy': deltaT,
            }
            metrics_rows.append(row)

            # plotting
            lam_plot = np.where(mask, lam_field, np.nan)
            alpha_plot = np.where(mask, alpha_field, np.nan)

            plt.figure(figsize=(4,4))
            plt.imshow(lam_plot, origin='lower', cmap='viridis', extent=(-args.R,args.R,-args.R,args.R))
            plt.colorbar(label='λ')
            plt.title(f'λ field ρ={rho} seed={seed}')
            plt.tight_layout()
            plt.savefig(args.figdir / f'lam_field_rho{rho}_seed{seed}.png')
            plt.close()

            plt.figure(figsize=(4,4))
            plt.imshow(alpha_plot, origin='lower', cmap='plasma', extent=(-args.R,args.R,-args.R,args.R))
            plt.colorbar(label='α')
            plt.title(f'α field ρ={rho} seed={seed}')
            plt.tight_layout()
            plt.savefig(args.figdir / f'alpha_field_rho{rho}_seed{seed}.png')
            plt.close()

            plt.figure(figsize=(4,4))
            plt.scatter(lam_vals, alpha_vals, s=5, alpha=0.4)
            plt.xlabel('λ')
            plt.ylabel('α')
            plt.title(f'λ vs α ρ={rho} seed={seed}')
            plt.tight_layout()
            plt.savefig(args.figdir / f'lam_vs_alpha_scatter_rho{rho}_seed{seed}.png')
            plt.close()

            fig, axs = plt.subplots(1,2,figsize=(8,4))
            im0 = axs[0].imshow(lam_plot, origin='lower', cmap='viridis', extent=(-args.R,args.R,-args.R,args.R))
            axs[0].set_title('λ')
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
            im1 = axs[1].imshow(alpha_plot, origin='lower', cmap='plasma', extent=(-args.R,args.R,-args.R,args.R))
            axs[1].set_title('α')
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
            fig.suptitle(f'ρ={rho} seed={seed}')
            plt.tight_layout()
            plt.savefig(args.figdir / f'lam_alpha_panels_rho{rho}_seed{seed}.png')
            plt.close()

    # write metrics csv
    fieldnames = [
        'rho','seed','lam_mean','alpha_mean','r_wire','t','density','A_eff','tau','deltaT_proxy'
    ]
    with args.out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

if __name__ == '__main__':
    main()
