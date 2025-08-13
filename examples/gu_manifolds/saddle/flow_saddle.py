import numpy as np, matplotlib.pyplot as plt
from matplotlib import animation
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from mesh_utils import make_saddle, build_cotan_weights, taubin_step, curvature_proxy

OUT_GIF = Path(__file__).with_name("saddle_curvature_flow.gif")
OUT_PNG = Path(__file__).with_name("saddle_curvature_flow_preview.png")

def main(R=1.0, a=1.2, n=60, steps=24, lam=0.02, mu=-0.01, sample_stride=2):
    V, F = make_saddle(R=R, a=a, n=n)
    nbrs, wts = build_cotan_weights(V, F, sample_stride=sample_stride)
    tris = F[::15]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(V[:,0], V[:,1], V[:,2], triangles=tris, linewidth=0.1, antialiased=False, alpha=1.0)
    ax.set_axis_off(); ax.set_title("Saddle — curvature flow")

    def animate(f):
        nonlocal V, surf
        V = taubin_step(V, nbrs, wts, lam=lam, mu=mu)
        colors = curvature_proxy(V, nbrs, wts)
        surf.remove()
        surf = ax.plot_trisurf(V[:,0], V[:,1], V[:,2], triangles=tris, linewidth=0.1, antialiased=False, alpha=1.0)
        surf.set_array(colors[tris].mean(axis=1))
        return (surf,)

    anim = animation.FuncAnimation(fig, animate, frames=steps, interval=80, blit=False)
    from matplotlib.animation import PillowWriter
    anim.save(OUT_GIF, writer=PillowWriter(fps=12))
    fig.savefig(OUT_PNG, dpi=160); plt.close(fig)
    print(f"[ok] wrote {OUT_GIF} and {OUT_PNG}")

if __name__ == "__main__":
    main()
