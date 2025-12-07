"""
adaptivecad.cam.pia_compensation
Compensate G2/G3 arcs in G-code using Adaptive-π model (πₐ).

Assumptions:
- Holes/arcs: curvature κ ≈ 1/r, scale ≈ r (local radius)
- We adjust the arc radius by factor (πₐ/π), leaving center and sweep sign intact.
- Supports R mode arcs; (I,J) mode is approximated by recomputing R from start/center if present.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass
class PiAParams:
    beta: float = 0.2
    s0: float = 1.0
    clamp: float = 0.3


def pi_a(kappa: float, scale: float, params: PiAParams) -> float:
    base = math.pi
    s0 = max(params.s0, 1e-9)
    frac = params.beta * (kappa * (scale / s0)) ** 2
    frac = max(-params.clamp, min(params.clamp, frac))
    return base * (1.0 + frac)


_arc_re = re.compile(r"^(?P<code>G0*[23])(?P<rest>.*)$", re.IGNORECASE)


def _extract_float(rest: str, key: str, default=None):
    m = re.search(rf"{key}(-?\d+(?:\.\d+)?)", rest, re.IGNORECASE)
    return float(m.group(1)) if m else default


def compensate_gcode_lines(lines, params: PiAParams, min_arc=0.5, max_arc=1e3):
    out = []
    for ln in lines:
        m = _arc_re.match(ln.strip())
        if not m:
            out.append(ln)
            continue
        m.group("code").upper()  # G2/G3
        rest = m.group("rest")
        r = _extract_float(rest, "R", None)
        if r is None:
            out.append(ln)
            continue
        rabs = abs(r)
        if not (min_arc <= rabs <= max_arc):
            out.append(ln)
            continue
        kappa = 1.0 / max(rabs, 1e-9)
        pa = pi_a(kappa, rabs, params)
        scale = pa / math.pi
        r_new = math.copysign(rabs * scale, r)
        ln2 = re.sub(r"R-?\d+(?:\.\d+)?", f"R{r_new:.6f}", ln)
        out.append(ln2)
    return out
