"""
plugins/aniso_distance_plugin.py

Anisotropic Distance (FMM-lite) plugin/CLI for AdaptiveCAD.
- CLI: compute distance map for a small grid and print samples
- register_plugin(app): stub for Playground integration
"""

from __future__ import annotations

import argparse
import json
import math

from adaptive_pi.aniso_fmm import (
    anisotropic_fmm,
    metric_const_aniso,
    trace_geodesic,
)

PLUGIN_META = {
    "name": "Anisotropic Distance (FMM-lite)",
    "slug": "aniso_distance",
    "version": "0.1.0",
    "entrypoints": ["compute_demo", "trace_demo"],
}


def compute_demo(nx=129, ny=129, a=1.3, b=1.0):
    G = metric_const_aniso(nx, ny, a=a, b=b)
    src = (nx // 2, ny // 2)
    T = anisotropic_fmm(G, src, use_diagonals=True)
    # print a few samples along axes
    mid = src[1]
    xs = [0, 5, 10, 20, 30, 40]
    ys = [0, 5, 10, 20, 30, 40]
    row_x = [round(float(T[src[0] + k, mid]), 4) for k in xs]
    row_y = [round(float(T[src[0], mid + k]), 4) for k in ys]
    print("Sample T along +x:", row_x)
    print("Sample T along +y:", row_y)
    print(f"Expected unit costs: sqrt(a)={math.sqrt(a):.4f}, sqrt(b)={math.sqrt(b):.4f}")


def trace_demo(nx=129, ny=129, a=1.3, b=1.0):
    G = metric_const_aniso(nx, ny, a=a, b=b)
    src = (nx // 2, ny // 2)
    T = anisotropic_fmm(G, src, use_diagonals=True)
    start = (nx - 5, ny // 2)
    path = trace_geodesic(T, G, start, src, step=0.8)
    print(json.dumps({"path_len": len(path), "start": start, "end": path[-1]}, indent=2))


def build_parser():
    p = argparse.ArgumentParser(description="Anisotropic Distance (FMM-lite)")
    p.add_argument("--mode", choices=["demo", "trace"], default="demo")
    p.add_argument("--nx", type=int, default=129)
    p.add_argument("--ny", type=int, default=129)
    p.add_argument("--a", type=float, default=1.3, help="G=diag(a,b)")
    p.add_argument("--b", type=float, default=1.0, help="G=diag(a,b)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "demo":
        compute_demo(args.nx, args.ny, args.a, args.b)
    else:
        trace_demo(args.nx, args.ny, args.a, args.b)


def register_plugin(app=None):
    # Playground can import this and wire buttons to `compute_demo`/`trace_demo`
    return PLUGIN_META


if __name__ == "__main__":
    main()
