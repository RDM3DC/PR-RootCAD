#!/usr/bin/env python3
"""
CLI: apply Adaptive-π compensation to G-code arcs (G2/G3 with R mode).
"""
import argparse

from adaptivecad.cam.pia_compensation import PiAParams, compensate_gcode_lines

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input G-code file")
    ap.add_argument("-o", "--output", default="out.gcode")
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--s0", type=float, default=1.0)
    ap.add_argument("--clamp", type=float, default=0.3)
    ap.add_argument("--min_arc_mm", type=float, default=0.5)
    ap.add_argument("--max_arc_mm", type=float, default=1000.0)
    args = ap.parse_args()

    params = PiAParams(beta=args.beta, s0=args.s0, clamp=args.clamp)
    with open(args.input, "r") as f:
        lines = f.readlines()
    out = compensate_gcode_lines(lines, params, min_arc=args.min_arc_mm, max_arc=args.max_arc_mm)
    with open(args.output, "w") as f:
        f.writelines(out)
    print(f"[ok] wrote {args.output}")
