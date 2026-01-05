#!/usr/bin/env python3
"""
PR Volume Demo — 3D phase field with diffusion and isosurface extraction.

Examples:
    # Basic 32³ volume, smooth noise geometry
    python pr_volume_demo.py --size 32 --steps 100 --out volume.ama

    # Higher resolution with spectral band-pass
    python pr_volume_demo.py --size 48 --geom-mode spectral_band --freq-low 0.03 --freq-high 0.12 --out volume_spectral.ama

    # Extract isosurface at specific level
    python pr_volume_demo.py --size 32 --iso 0.2 --out volume_iso.ama --plot
"""
from __future__ import annotations

import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="PR Volume 3D demo")
    ap.add_argument("--size", type=int, default=32, help="Grid size N (NxNxN)")
    ap.add_argument("--steps", type=int, default=100, help="Relaxation steps")
    ap.add_argument("--dt", type=float, default=0.15, help="Time step")
    ap.add_argument("--diffusion", type=float, default=0.35, help="Diffusion rate")
    ap.add_argument("--coupling", type=float, default=0.20, help="Geometry coupling")
    ap.add_argument("--geom-mode", choices=["smooth_noise", "spectral_band"], default="smooth_noise")
    ap.add_argument("--geom-smooth-iters", type=int, default=4)
    ap.add_argument("--freq-low", type=float, default=0.02)
    ap.add_argument("--freq-high", type=float, default=0.15)
    ap.add_argument("--freq-power", type=float, default=1.0)
    ap.add_argument("--phase-space", choices=["unwrapped", "wrapped"], default="unwrapped")
    ap.add_argument("--iso", type=float, default=0.0, help="Isosurface level")
    ap.add_argument("--scale", type=float, default=10.0, help="Output scale (mm)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="pr_volume.ama", help="Output .ama path")
    ap.add_argument("--plot", action="store_true", help="Show 3D visualization")
    args = ap.parse_args()

    from adaptivecad.pr.volume import PRVolumeConfig, relax_phase_volume, save_volume_ama, extract_isosurface

    cfg = PRVolumeConfig(
        size=args.size,
        steps=args.steps,
        dt=args.dt,
        diffusion=args.diffusion,
        coupling=args.coupling,
        coupling_mode="geom_target",
        geom_mode=args.geom_mode,
        geom_smooth_iters=args.geom_smooth_iters,
        geom_freq_low=args.freq_low,
        geom_freq_high=args.freq_high,
        geom_freq_power=args.freq_power,
        phase_space=args.phase_space,
        iso_level=args.iso,
        seed=args.seed,
    )

    print(f"PR Volume Demo")
    print(f"  Grid: {cfg.size}³ = {cfg.size**3:,} voxels")
    print(f"  Steps: {cfg.steps}, dt={cfg.dt}, diffusion={cfg.diffusion}, coupling={cfg.coupling}")
    print(f"  Geom mode: {cfg.geom_mode}")
    print(f"  Iso level: {cfg.iso_level}")
    print()

    print("Running 3D phase relaxation...")
    metrics, state = relax_phase_volume(cfg)

    print(f"Done!")
    print(f"  φ range: [{metrics['phi_min']:.4f}, {metrics['phi_max']:.4f}]")
    print(f"  φ mean: {metrics['phi_mean']:.4f}")
    print(f"  Residual mean: {metrics['residual_mean']:.4f}, max: {metrics['residual_max']:.4f}")
    print(f"  Final delta: {metrics['final_delta']:.6f}")
    print()

    # Extract isosurface for stats
    phi = state.phi if state.phi.ndim == 3 else state.phi[..., 0]
    verts, faces = extract_isosurface(phi, level=args.iso, scale=args.scale)
    print(f"Isosurface at φ={args.iso}:")
    print(f"  Vertices: {len(verts):,}")
    print(f"  Faces: {len(faces):,}")

    # Save AMA
    path = save_volume_ama(state, cfg, args.out, iso_level=args.iso, scale=args.scale, units="mm")
    print(f"\nSaved: {path}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            fig = plt.figure(figsize=(14, 5))

            # 1) Slice through volume (XY at Z=N/2)
            ax1 = fig.add_subplot(131)
            mid_z = cfg.size // 2
            im = ax1.imshow(phi[:, :, mid_z].T, origin="lower", cmap="plasma")
            ax1.set_title(f"φ slice at z={mid_z}")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            plt.colorbar(im, ax=ax1, fraction=0.046)

            # 2) Geometry proxy slice
            ax2 = fig.add_subplot(132)
            im2 = ax2.imshow(state.geom[:, :, mid_z].T, origin="lower", cmap="viridis")
            ax2.set_title(f"Geometry proxy at z={mid_z}")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            plt.colorbar(im2, ax=ax2, fraction=0.046)

            # 3) Isosurface 3D view
            ax3 = fig.add_subplot(133, projection="3d")
            if len(verts) > 0 and len(faces) > 0:
                # Subsample faces for plotting (too many is slow)
                max_faces = 5000
                if len(faces) > max_faces:
                    idx = np.random.choice(len(faces), max_faces, replace=False)
                    faces_plot = faces[idx]
                else:
                    faces_plot = faces

                tri_verts = verts[faces_plot]
                
                # Color by z-coordinate
                z_vals = tri_verts[:, :, 2].mean(axis=1)
                z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-9)
                colors = plt.cm.plasma(z_norm)
                colors[:, 3] = 0.7

                collection = Poly3DCollection(tri_verts, facecolors=colors, edgecolors="k", linewidths=0.1)
                ax3.add_collection3d(collection)

                ax3.set_xlim(verts[:, 0].min(), verts[:, 0].max())
                ax3.set_ylim(verts[:, 1].min(), verts[:, 1].max())
                ax3.set_zlim(verts[:, 2].min(), verts[:, 2].max())
            ax3.set_xlabel("X")
            ax3.set_ylabel("Y")
            ax3.set_zlabel("Z")
            ax3.set_title(f"Isosurface φ={args.iso}")

            plt.tight_layout()
            plt.savefig("pr_volume_demo.png", dpi=150)
            print("Saved: pr_volume_demo.png")
            plt.show()

        except ImportError as e:
            print(f"Plotting unavailable: {e}")


if __name__ == "__main__":
    main()
