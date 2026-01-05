import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from mesh_utils import build_cotan_weights, curvature_proxy, make_klein, taubin_step

OUT_GIF = Path(__file__).with_name("klein_curvature_flow.gif")
OUT_PNG = Path(__file__).with_name("klein_curvature_flow_preview.png")


def main(nu=40, nv=40, steps=24, lam=0.02, mu=-0.01, sample_stride=2):
    V, F = make_klein(nu=nu, nv=nv, a=2.0)
    nbrs, wts = build_cotan_weights(V, F, sample_stride=sample_stride)
    tris = F[::10]  # draw subset for speed

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(
        V[:, 0], V[:, 1], V[:, 2], triangles=tris, linewidth=0.1, antialiased=False, alpha=1.0
    )
    ax.set_axis_off()
    ax.set_title("Klein bottle — curvature flow")

    def animate(f):
        nonlocal V, surf
        V = taubin_step(V, nbrs, wts, lam=lam, mu=mu)
        colors = curvature_proxy(V, nbrs, wts)
        surf.remove()
        surf = ax.plot_trisurf(
            V[:, 0], V[:, 1], V[:, 2], triangles=tris, linewidth=0.1, antialiased=False, alpha=1.0
        )
        surf.set_array(colors[tris].mean(axis=1))
        return (surf,)

    anim = animation.FuncAnimation(fig, animate, frames=steps, interval=80, blit=False)
    from matplotlib.animation import PillowWriter

    anim.save(OUT_GIF, writer=PillowWriter(fps=12))
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)
    print(f"[ok] wrote {OUT_GIF} and {OUT_PNG}")


if __name__ == "__main__":
    main()
