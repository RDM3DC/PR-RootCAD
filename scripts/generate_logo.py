"""Generate a simple logo inspired by the attached analytic viewport image.

This script uses matplotlib to draw overlapping parametric shapes (superellipse-like
lobes and a central ring) and saves both PNG and SVG outputs.

Run:
    python scripts/generate_logo.py

Outputs:
    logo.png, logo.svg in the repository root.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path


def superellipse(a, b, n, t):
    # parametric superellipse-like curve
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    x = a * np.sign(cos_t) * (np.abs(cos_t) ** (2.0 / n))
    y = b * np.sign(sin_t) * (np.abs(sin_t) ** (2.0 / n))
    return x, y


def lobe(a, b, n, rot=0.0, offset=(0, 0), scale=1.0):
    t = np.linspace(0, 2 * math.pi, 512)
    x, y = superellipse(a, b, n, t)
    coords = np.vstack([x, y]).T * scale
    # rotate
    c = math.cos(rot)
    s = math.sin(rot)
    R = np.array([[c, -s], [s, c]])
    coords = coords.dot(R.T)
    coords += np.array(offset)
    return coords


def make_logo(filename_png="logo.png", filename_svg="logo.svg"):
    fig = plt.figure(figsize=(8, 8), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")

    # background circle (dark)
    bg = Circle((0, 0), 1.5, color="#0f1b22")
    ax.add_patch(bg)

    # lobes
    coords1 = lobe(1.0, 0.6, 2.2, rot=0.2, offset=(-0.15, 0.0), scale=1.1)
    coords2 = lobe(1.0, 0.6, 2.2, rot=-0.9, offset=(0.15, 0.0), scale=1.1)
    coords3 = lobe(0.9, 0.5, 1.6, rot=1.6, offset=(0.0, 0.0), scale=1.0)

    # create patches with smooth edges
    codes = [Path.MOVETO] + [Path.CURVE4] * (len(coords1) - 1)
    path1 = Path(coords1, codes)
    patch1 = PathPatch(path1, facecolor="#7fe27f", edgecolor="none", alpha=0.95)
    ax.add_patch(patch1)

    codes = [Path.MOVETO] + [Path.CURVE4] * (len(coords2) - 1)
    path2 = Path(coords2, codes)
    patch2 = PathPatch(path2, facecolor="#3fa3d0", edgecolor="none", alpha=0.95)
    ax.add_patch(patch2)

    codes = [Path.MOVETO] + [Path.CURVE4] * (len(coords3) - 1)
    path3 = Path(coords3, codes)
    patch3 = PathPatch(path3, facecolor="#7fe27f", edgecolor="none", alpha=0.9)
    ax.add_patch(patch3)

    # central ring (donut)
    outer = Circle((0, 0), 0.35, facecolor="#c14d3a", edgecolor="none")
    inner = Circle((0, 0), 0.17, facecolor="#0f1b22", edgecolor="none")
    ax.add_patch(outer)
    ax.add_patch(inner)

    # highlight smaller ring offset
    small = Circle((0.32, 0.0), 0.09, facecolor="#7fe27f", edgecolor="none")
    ax.add_patch(small)

    # subtle vignette (radial gradient via overlay)
    xx = np.linspace(-1.6, 1.6, 800)
    yy = np.linspace(-1.6, 1.6, 800)
    X, Y = np.meshgrid(xx, yy)
    R = np.sqrt(X**2 + Y**2)
    vignette = np.clip((1.6 - R) / 1.6, 0, 1)
    ax.imshow(np.dstack([vignette * 0.05] * 3), extent=[-1.6, 1.6, -1.6, 1.6], origin="lower")

    # frame and export
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(filename_png, dpi=200, transparent=False)
    fig.savefig(filename_svg)
    plt.close(fig)
    print(f"Saved {filename_png} and {filename_svg}")


if __name__ == "__main__":
    make_logo()
